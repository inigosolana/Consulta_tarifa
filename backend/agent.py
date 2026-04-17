"""
backend/agent.py
================
Agente de voz LiveKit para atención al cliente sobre tarifas telefónicas.

Flujo completo:
  1. El cliente llama → el agente saluda y ofrece ayuda sobre tarifas.
  2. El cliente pregunta por una tarifa → el LLM responde EXCLUSIVAMENTE
     con el catálogo definido en backend/prompts.py.
  3. Si el LLM no puede responder con el catálogo, o si el cliente pide
     un agente humano → se invoca `transferir_a_cola_humana`.
  4. `transferir_a_cola_humana` envía un SIP REFER via LiveKit API hacia
     la extensión de cola configurada en la centralita Yeastar.

Arranque:
    python -m backend.agent dev        # modo desarrollo (log verbose)
    python -m backend.agent start      # producción con registro en LiveKit Cloud
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.plugins import cartesia, deepgram, groq, silero

# Garantizar que el raíz del proyecto esté en sys.path cuando el módulo
# se lanza directamente con `python -m backend.agent` o `python backend/agent.py`.
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.prompts import SYSTEM_PROMPT  # noqa: E402
from backend.services.yeastar_service import YeastarService  # noqa: E402

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: localizar el participante SIP caller en la sala
# ---------------------------------------------------------------------------

def _find_sip_participant_identity(room: rtc.Room) -> str | None:
    """
    Devuelve la identity del participante SIP entrante (el cliente).

    LiveKit añade atributos con prefijo 'sip.' a los participantes SIP
    (p. ej. 'sip.callID', 'sip.trunkPhoneNumber').  Si no se encuentran,
    se devuelve el primer participante remoto disponible como fallback.
    """
    for participant in room.remote_participants.values():
        attrs = participant.attributes or {}
        if any(k.startswith("sip.") for k in attrs):
            logger.debug(
                "Participante SIP identificado por atributos: '%s'",
                participant.identity,
            )
            return participant.identity

    # Fallback: primer participante remoto (válido en salas 1-a-1)
    remote = list(room.remote_participants.values())
    if remote:
        logger.debug(
            "Participante SIP identificado por fallback: '%s'",
            remote[0].identity,
        )
        return remote[0].identity

    return None


# ---------------------------------------------------------------------------
# Factory del agente con la tool de transferencia inyectada
# ---------------------------------------------------------------------------

def build_agent(ctx: JobContext, yeastar: YeastarService) -> Agent:
    """
    Construye el Agent de LiveKit con el tool `transferir_a_cola_humana`
    vinculado al contexto de la llamada en curso.

    El tool se define dentro de la closure para capturar `ctx` y `yeastar`
    sin necesidad de variables globales.
    """

    @function_tool
    async def transferir_a_cola_humana(
        motivo: Annotated[
            str,
            (
                "Motivo por el que se transfiere al agente humano. "
                "Ejemplos: 'el cliente lo solicita', 'tarifa no incluida en el catálogo', "
                "'el cliente está insatisfecho con la información disponible'."
            ),
        ],
    ) -> str:
        """
        Transfiere la llamada en curso a la cola de agentes humanos
        en la centralita Yeastar mediante SIP REFER.

        CUÁNDO INVOCAR ESTA FUNCIÓN:
        - El cliente solicita explícitamente hablar con una persona.
        - La pregunta del cliente no puede responderse con el catálogo de tarifas.
        - El cliente expresa insatisfacción o quiere escalar la consulta.
        - Han pasado más de 2 intentos sin resolver la duda del cliente.

        NO invocar si el cliente simplemente pide que repitas o aclares
        información que sí está en el catálogo.
        """
        logger.info(
            "Invocando transferencia a cola humana. Motivo: '%s'", motivo
        )

        sip_identity = _find_sip_participant_identity(ctx.room)

        if sip_identity is None:
            logger.warning(
                "No se encontró participante SIP en la sala '%s'. "
                "No es posible ejecutar el SIP REFER.",
                ctx.room.name,
            )
            return (
                "Lo siento, no he podido localizar tu línea para realizar "
                "la transferencia automática. Por favor, llama al 900 XXX XXX "
                "y un agente te atenderá directamente."
            )

        ok = await yeastar.transfer_to_queue(
            room_name=ctx.room.name,
            participant_identity=sip_identity,
        )

        if ok:
            return (
                "Perfecto, te estoy pasando ahora mismo con uno de nuestros agentes. "
                "Por favor, mantente en línea un momento."
            )
        else:
            return (
                "Ha ocurrido un problema técnico al intentar la transferencia. "
                "Por favor, llama directamente al 900 XXX XXX y un agente "
                "te atenderá sin espera."
            )

    return Agent(
        instructions=SYSTEM_PROMPT,
        tools=[transferir_a_cola_humana],
    )


# ---------------------------------------------------------------------------
# Punto de entrada del worker LiveKit
# ---------------------------------------------------------------------------

async def entrypoint(ctx: JobContext) -> None:
    """Punto de entrada principal para cada llamada entrante."""
    await ctx.connect(auto_subscribe=False)
    logger.info("Sala conectada: %s", ctx.room.name)

    yeastar = YeastarService()

    try:
        agent = build_agent(ctx, yeastar)

        session = AgentSession(
            stt=deepgram.STT(model="nova-2", language="es"),
            llm=groq.LLM(model="llama-3.3-70b-versatile"),
            tts=cartesia.TTS(
                voice=os.getenv(
                    "VOICE_ID_ESPANOLA",
                    "cefcb124-080b-4655-b31f-932f3ee743de",
                ),
            ),
            vad=silero.VAD.load(),
        )

        await session.start(agent=agent, room=ctx.room)

        await session.generate_reply(
            instructions=(
                "Saluda al cliente de forma breve y amable. Indícale que puedes "
                "ayudarle con cualquier consulta sobre las tarifas telefónicas "
                "disponibles y que también puedes transferirle con un agente si lo necesita."
            )
        )

        logger.info("Agente de tarifas activo, esperando consultas del cliente.")

    except Exception:
        logger.exception("Error fatal durante la sesión del agente.")
        raise
    finally:
        await yeastar.aclose()


# ---------------------------------------------------------------------------
# Arranque del worker
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=os.getenv("AGENT_NAME_DISPATCH", "tarifas-atencion-cliente"),
        )
    )
