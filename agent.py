"""
agent.py
========
Agente de voz LiveKit para consulta de tarifas telefónicas.

Flujo completo:
  1. El comercial pregunta por una tarifa → el agente llama a `buscar_tarifa`.
  2. Si hay información → el agente la lee en voz alta.
  3. Si NO hay información → el agente avisa de la transferencia y entra en
     "modo escucha silenciosa":
       · El bot se silencia (no genera más respuestas de voz).
       · La sala PERMANECE abierta.
       · Se suscribe al audio de TODOS los participantes (comercial + agente humano)
         y corre Deepgram sobre cada uno de forma independiente.
  4. Al desconectarse la sala → el LLM analiza la transcripción capturada,
     extrae la explicación comercial y actualiza el CSV automáticamente.
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
from livekit.plugins import cartesia, deepgram, groq, silero

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
        # Transcripción capturada durante la escucha silenciosa (post-transferencia)
        self.transcripcion_post_transfer: list[str] = []
        # Tasks de STT activas para cada participante (para poder cancelarlas)
        self._tasks_stt: list[asyncio.Task] = []


# ---------------------------------------------------------------------------
# Escucha STT por participante (corazón del sistema de captura)
# ---------------------------------------------------------------------------

async def _transcribir_pista(
    participant: rtc.RemoteParticipant,
    track: rtc.Track,
    state: SessionState,
) -> None:
    """
    Corre Deepgram sobre la pista de audio de UN participante y guarda
    cada frase final en state.transcripcion_post_transfer, etiquetada
    con la identidad del participante (COMERCIAL / AGENTE_HUMANO).

    Esta función se ejecuta como una Task asíncrona independiente por cada
    participante que se une a la sala durante la escucha silenciosa.
    """
    # Heurística de etiquetado: si el identity contiene "agent" se asume
    # que es el agente humano físico. Ajusta según tu naming en LiveKit.
    es_agente_humano = any(
        kw in participant.identity.lower()
        for kw in ("agent", "agente", "human", "support", "asesor")
    )
    label = "AGENTE_HUMANO" if es_agente_humano else "COMERCIAL"

    logger.info(
        "Iniciando STT para participante '%s' → etiqueta [%s]",
        participant.identity,
        label,
    )

    stt_instance = deepgram.STT(model="nova-2", language="es")
    # Resamplea a 16 kHz mono, que es lo que necesita Deepgram
    audio_stream = rtc.AudioStream(track, sample_rate=16_000, num_channels=1)

    try:
        async with stt_instance.stream() as stt_stream:

            async def push_frames() -> None:
                async for audio_event in audio_stream:
                    stt_stream.push_frame(audio_event.frame)

            async def read_events() -> None:
                async for stt_event in stt_stream:
                    # Solo procesamos transcripciones finales (no parciales)
                    if stt_event.type.name != "FINAL_TRANSCRIPT":
                        continue
                    if not stt_event.alternatives:
                        continue
                    texto = stt_event.alternatives[0].text.strip()
                    if texto:
                        entrada = f"[{label}]: {texto}"
                        state.transcripcion_post_transfer.append(entrada)
                        logger.info("Capturado %s", entrada)

            await asyncio.gather(push_frames(), read_events())

    except asyncio.CancelledError:
        logger.info("STT cancelado para participante '%s'.", participant.identity)
    except Exception as exc:
        logger.error(
            "Error en STT del participante '%s': %s",
            participant.identity,
            exc,
        )


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

# Palabras clave que indican que hay que transferir la llamada
_KEYWORDS_TRANSFERENCIA = (
    "transfiero",
    "transfiere",
    "agente humano",
    "paso con un",
    "transferirte",
    "no tengo información",
    "no dispongo",
)


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

        if any(kw in resultado.lower() for kw in _KEYWORDS_TRANSFERENCIA):
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
# Helpers
# ---------------------------------------------------------------------------

def _extraer_transcripcion_bot(session: AgentSession) -> str:
    """Extrae el historial de chat del bot (la parte antes de transferir)."""
    lineas: list[str] = []
    for msg in session.chat_ctx.messages:
        rol = getattr(msg, "role", "unknown")
        contenido = getattr(msg, "content", "")
        if isinstance(contenido, list):
            partes: list[str] = []
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
        llm=groq.LLM(model="llama-3.3-70b-versatile"),
        tts=cartesia.TTS(
            voice=os.getenv("VOICE_ID_ESPANOLA", "cefcb124-080b-4655-b31f-932f3ee743de"),
        ),
        vad=silero.VAD.load(),
    )

    # ------------------------------------------------------------------
    # Silenciado del bot cuando entra en modo escucha silenciosa
    # ------------------------------------------------------------------
    async def _monitor_silencio() -> None:
        """
        Cada 500 ms comprueba si hay que silenciar el bot.
        Cuando se activa:
          · Interrumpe la respuesta en curso.
          · Deshabilita el pipeline de entrada para que no genere más respuestas.
        """
        while True:
            await asyncio.sleep(0.5)
            if state.en_escucha_silenciosa:
                try:
                    await session.interrupt()
                except Exception:
                    pass
                try:
                    session.input.set_audio_enabled(False)
                except Exception:
                    pass
                logger.info("Bot silenciado. Modo escucha silenciosa activo.")
                break   # ya no necesitamos seguir monitoreando

    # ------------------------------------------------------------------
    # Suscripción a audio de cada participante durante escucha silenciosa
    # ------------------------------------------------------------------
    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        _publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        """
        Se dispara cuando se publica una pista de audio de cualquier participante.
        Si estamos en modo escucha silenciosa, lanzamos un STT independiente
        sobre esa pista para capturar lo que dice ese participante.
        """
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        if not state.en_escucha_silenciosa:
            logger.debug(
                "Pista de '%s' ignorada (no estamos en escucha silenciosa aún).",
                participant.identity,
            )
            return

        logger.info(
            "Nueva pista de audio detectada: participante '%s'. Iniciando STT.",
            participant.identity,
        )
        task = asyncio.ensure_future(
            _transcribir_pista(participant, track, state)
        )
        state._tasks_stt.append(task)

    # ------------------------------------------------------------------
    # También capturamos el audio del comercial si la transferencia ocurre
    # ANTES de que llegue el agente humano (suscripción retroactiva)
    # ------------------------------------------------------------------
    async def _suscribir_participantes_existentes() -> None:
        """
        Cuando se activa el modo silencioso, suscribe retroactivamente las
        pistas de audio de los participantes que ya estaban en la sala.
        """
        while True:
            await asyncio.sleep(0.5)
            if not state.en_escucha_silenciosa:
                continue
            for participant in ctx.room.remote_participants.values():
                for pub in participant.track_publications.values():
                    if (
                        pub.track is not None
                        and pub.track.kind == rtc.TrackKind.KIND_AUDIO
                    ):
                        already_running = any(
                            not t.done() for t in state._tasks_stt
                        )
                        if not already_running:
                            task = asyncio.ensure_future(
                                _transcribir_pista(participant, pub.track, state)
                            )
                            state._tasks_stt.append(task)
            break   # lo hacemos una sola vez al activarse el modo

    # ------------------------------------------------------------------
    # Evento de desconexión → procesamiento post-llamada
    # ------------------------------------------------------------------
    @ctx.room.on("disconnected")
    def on_room_disconnected(*_args: object) -> None:
        logger.info("Sala desconectada. Iniciando procesamiento post-llamada...")

        # Cancelar tasks STT activas
        for task in state._tasks_stt:
            task.cancel()

        if not state.ultima_tarifa_buscada:
            logger.info("No se buscó ninguna tarifa. Nada que procesar.")
            return

        # Construir transcripción completa: conversación con el bot + post-transferencia
        transcripcion_bot = _extraer_transcripcion_bot(session)
        lineas_post = "\n".join(state.transcripcion_post_transfer)

        transcripcion_total = transcripcion_bot
        if lineas_post.strip():
            transcripcion_total += (
                "\n\n--- CONVERSACIÓN CON AGENTE HUMANO (post-transferencia) ---\n"
                + lineas_post
            )

        logger.info(
            "Transcripción total: %d chars para tarifa '%s'.\nPost-transferencia: %d líneas.",
            len(transcripcion_total),
            state.ultima_tarifa_buscada,
            len(state.transcripcion_post_transfer),
        )

        asyncio.ensure_future(
            procesar_post_llamada(state.ultima_tarifa_buscada, transcripcion_total)
        )

    # ------------------------------------------------------------------
    # Arranque
    # ------------------------------------------------------------------
    asyncio.ensure_future(_monitor_silencio())
    asyncio.ensure_future(_suscribir_participantes_existentes())

    await session.start(agent=agent, room=ctx.room)

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
