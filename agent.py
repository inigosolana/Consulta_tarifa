"""
agent.py
========
Agente de voz LiveKit para consulta de tarifas telefónicas.

Flujo completo:
  1. El comercial pregunta por una tarifa → el agente llama a `buscar_tarifa`.
  2. Si hay información → el agente la lee en voz alta.
  3. Si NO hay información → el agente avisa de la transferencia y entra en
     "modo escucha silenciosa" (STT activo, voz del bot pausada).
  4. Al desconectarse la sala → se extrae el conocimiento de la transcripción
     y se actualiza el CSV automáticamente.
"""

from __future__ import annotations

import asyncio
import logging
import os
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
from livekit.plugins import cartesia, deepgram, openai, silero

from knowledge_feedback import buscar_tarifa, procesar_post_llamada

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Estado compartido de la sesión
# ---------------------------------------------------------------------------

class SessionState:
    """Mantiene el estado mutable de una sesión de llamada."""

    def __init__(self) -> None:
        self.en_escucha_silenciosa: bool = False
        self.ultima_tarifa_buscada: str = ""
        self.transcripcion_post_transfer: list[str] = []


# ---------------------------------------------------------------------------
# Definición del Agente
# ---------------------------------------------------------------------------

INSTRUCCIONES_SISTEMA = """\
Eres un asistente de voz experto en tarifas telefónicas de la empresa.
Tu función es ayudar a los comerciales a consultar información sobre tarifas.

Cuando un comercial pregunte por una tarifa:
1. Llama a la herramienta `buscar_tarifa` con el nombre de la tarifa.
2. Si obtienes información, léela de forma clara y natural.
3. Si el mensaje indica que debes transferir la llamada, dilo en voz alta
   exactamente como te lo indique el mensaje y NO digas nada más.

Sé conciso, profesional y amable.
"""


def build_agent(state: SessionState) -> Agent:
    """Construye el agente con las tools inyectadas."""

    @function_tool
    async def buscar_tarifa_tool(
        nombre_tarifa: Annotated[
            str,
            "Nombre exacto o aproximado de la tarifa que el comercial ha preguntado.",
        ],
    ) -> str:
        """Busca la información comercial de una tarifa en el CRM."""
        state.ultima_tarifa_buscada = nombre_tarifa
        resultado = await buscar_tarifa(nombre_tarifa)

        # Si el resultado indica transferencia, activamos escucha silenciosa
        palabras_clave_transferencia = [
            "transfiero",
            "transfiere",
            "agente humano",
            "paso con un",
            "transferirte",
        ]
        if any(kw in resultado.lower() for kw in palabras_clave_transferencia):
            logger.info(
                "Tarifa '%s' sin info → activando modo escucha silenciosa.",
                nombre_tarifa,
            )
            state.en_escucha_silenciosa = True

        return resultado

    return Agent(
        instructions=INSTRUCCIONES_SISTEMA,
        tools=[buscar_tarifa_tool],
    )


# ---------------------------------------------------------------------------
# Lógica de escucha silenciosa
# ---------------------------------------------------------------------------

def _extraer_transcripcion(session: AgentSession) -> str:
    """
    Extrae el texto plano de todo el historial de chat de la sesión.
    Funciona con la lista de mensajes de `chat_ctx`.
    """
    lineas: list[str] = []

    for msg in session.chat_ctx.messages:
        rol = getattr(msg, "role", "unknown")
        # El contenido puede ser str o lista de bloques
        contenido = getattr(msg, "content", "")
        if isinstance(contenido, list):
            partes = []
            for bloque in contenido:
                if hasattr(bloque, "text"):
                    partes.append(bloque.text)
                elif isinstance(bloque, str):
                    partes.append(bloque)
            contenido = " ".join(partes)

        if contenido and str(contenido).strip():
            lineas.append(f"[{rol.upper()}]: {contenido}")

    return "\n".join(lineas)


# ---------------------------------------------------------------------------
# Punto de entrada del worker LiveKit
# ---------------------------------------------------------------------------

async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect(auto_subscribe=True)
    logger.info("Sala conectada: %s", ctx.room.name)

    state = SessionState()
    agent = build_agent(state)

    session = AgentSession(
        stt=deepgram.STT(model="nova-2", language="es"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(
            voice=os.getenv("VOICE_ID_ESPANOLA", "cefcb124-080b-4655-b31f-932f3ee743de"),
        ),
        vad=silero.VAD.load(),
    )

    # ------------------------------------------------------------------
    # Evento de desconexión → procesamiento post-llamada
    # ------------------------------------------------------------------
    @ctx.room.on("disconnected")
    def on_room_disconnected(*_args: object) -> None:
        logger.info("Sala desconectada. Iniciando procesamiento post-llamada...")

        if not state.ultima_tarifa_buscada:
            logger.info("No se buscó ninguna tarifa. Nada que procesar.")
            return

        transcripcion = _extraer_transcripcion(session)

        if state.en_escucha_silenciosa and state.transcripcion_post_transfer:
            # Añadimos lo capturado durante la escucha silenciosa
            transcripcion += "\n\n--- CONVERSACIÓN CON AGENTE HUMANO ---\n"
            transcripcion += "\n".join(state.transcripcion_post_transfer)

        logger.info(
            "Transcripción total (%d chars) para tarifa '%s'.",
            len(transcripcion),
            state.ultima_tarifa_buscada,
        )

        # Lanzamos la tarea asíncrona en el loop del proceso
        asyncio.ensure_future(
            procesar_post_llamada(state.ultima_tarifa_buscada, transcripcion)
        )

    # ------------------------------------------------------------------
    # Captura de audio/texto en modo escucha silenciosa
    # ------------------------------------------------------------------
    @session.on("user_speech_committed")
    def on_user_speech(event: object) -> None:
        """Captura utterances del usuario durante la escucha silenciosa."""
        if not state.en_escucha_silenciosa:
            return

        texto = getattr(event, "transcript", None) or getattr(event, "text", None)
        if texto and str(texto).strip():
            state.transcripcion_post_transfer.append(f"[HUMANO]: {texto}")
            logger.debug("Escucha silenciosa capturó: %s", texto)

    @session.on("agent_speech_committed")
    def on_agent_speech(event: object) -> None:
        """En modo silencioso, registra pero NO emite voz adicional."""
        if not state.en_escucha_silenciosa:
            return
        texto = getattr(event, "transcript", None) or getattr(event, "text", None)
        if texto and str(texto).strip():
            state.transcripcion_post_transfer.append(f"[AGENTE_FÍSICO]: {texto}")

    # ------------------------------------------------------------------
    # Pausa de la síntesis de voz al entrar en escucha silenciosa
    # ------------------------------------------------------------------
    async def _monitor_modo_silencioso() -> None:
        """
        Monitorea el flag de escucha silenciosa.
        Cuando se activa, interrumpe cualquier respuesta en curso del bot
        y evita que siga generando audio.
        """
        while True:
            await asyncio.sleep(0.5)
            if state.en_escucha_silenciosa:
                # Interrumpir respuesta en curso si existe
                try:
                    await session.interrupt()
                    logger.info("Modo escucha silenciosa activo. Bot silenciado.")
                except Exception:
                    pass
                # Deshabilitar generación de respuestas futuras
                session.input.set_enabled(False)   # silencia el TTS pipeline
                logger.info("Pipeline de respuesta deshabilitado (escucha silenciosa).")
                break

    # ------------------------------------------------------------------
    # Arranque de la sesión
    # ------------------------------------------------------------------
    asyncio.ensure_future(_monitor_modo_silencioso())

    await session.start(
        agent=agent,
        room=ctx.room,
    )

    await session.generate_reply(
        instructions=(
            "Saluda al comercial brevemente y pregúntale sobre qué tarifa "
            "necesita información hoy."
        )
    )

    logger.info("Agente en espera de preguntas del comercial.")


# ---------------------------------------------------------------------------
# Arranque del worker
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=os.getenv("AGENT_NAME_DISPATCH", "consulta-tarifas-agent"),
        )
    )
