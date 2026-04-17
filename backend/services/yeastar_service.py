"""
yeastar_service.py
==================
Servicio de transferencia de llamadas hacia colas de Yeastar P-Series
mediante SIP REFER a través de la API de LiveKit SIP.

Flujo técnico:
  Agente IA detecta necesidad de transferir
      ↓
  YeastarService.transfer_to_queue()
      ↓
  LiveKit SIP API → TransferSIPParticipantRequest
      ↓
  LiveKit envía SIP REFER al participante (caller)
      ↓
  El teléfono/SIP del cliente redirige al URI indicado
      ↓
  Yeastar (trunk_citelia_entrante_inigo) → Cola de agentes humanos

Requisitos en Yeastar P-Series:
  - Crear una Cola (Queue) y anotar su número de extensión (p. ej. 6100).
  - El trunk SIP 'trunk_citelia_entrante_inigo' debe tener una Inbound Route
    que enrute las llamadas al número de extensión de la cola.
  - Si se usa SIP REFER directo al host de Yeastar, asegúrate de que el
    trunk/IP pool de Yeastar acepte el REFER entrante desde la IP de LiveKit.
    (En Yeastar: PBX → Trunks → <trunk> → Advanced → Allow Remote Party REFER)
"""

from __future__ import annotations

import logging
import os

from livekit import api as lkapi

logger = logging.getLogger(__name__)


class YeastarService:
    """
    Encapsula las operaciones de transferencia SIP hacia la centralita Yeastar.

    Variables de entorno requeridas
    --------------------------------
    LIVEKIT_URL            URL WSS del servidor LiveKit
    LIVEKIT_API_KEY        API Key de LiveKit
    LIVEKIT_API_SECRET     API Secret de LiveKit
    YEASTAR_IP             IP o hostname de la centralita Yeastar (p. ej. 192.168.1.100)
    YEASTAR_QUEUE_EXT      Número de extensión de la cola de agentes (p. ej. 6100)
    """

    def __init__(self) -> None:
        self._lk = lkapi.LiveKitAPI(
            url=os.environ["LIVEKIT_URL"],
            api_key=os.environ["LIVEKIT_API_KEY"],
            api_secret=os.environ["LIVEKIT_API_SECRET"],
        )
        self._yeastar_ip: str = os.getenv("YEASTAR_IP", "")
        self._queue_ext: str = os.getenv("YEASTAR_QUEUE_EXT", "6100")

    # ------------------------------------------------------------------
    # Propiedad helper: URI SIP de la cola
    # ------------------------------------------------------------------

    @property
    def queue_sip_uri(self) -> str:
        """
        Devuelve la URI SIP de destino para la cola humana.

        Formato: sip:<extension>@<yeastar_ip>

        Yeastar P-Series acepta llamadas entrantes SIP a sus extensiones
        directamente cuando la IP origen está autorizada en el trunk.
        """
        if not self._yeastar_ip:
            raise ValueError(
                "YEASTAR_IP no está configurado en las variables de entorno."
            )
        return f"sip:{self._queue_ext}@{self._yeastar_ip}"

    # ------------------------------------------------------------------
    # Transferencia principal
    # ------------------------------------------------------------------

    async def transfer_to_queue(
        self,
        room_name: str,
        participant_identity: str,
        queue_ext: str | None = None,
    ) -> bool:
        """
        Envía un SIP REFER al participante SIP indicado para redirigirlo
        a la cola de agentes humanos en Yeastar.

        Args:
            room_name:             Nombre de la sala LiveKit activa.
            participant_identity:  Identity del participante SIP (el cliente).
            queue_ext:             Extensión de cola a usar; si es None, usa
                                   la configurada en YEASTAR_QUEUE_EXT.

        Returns:
            True si el REFER se envió sin error, False en caso contrario.
        """
        ext = queue_ext or self._queue_ext
        transfer_uri = f"sip:{ext}@{self._yeastar_ip}" if self._yeastar_ip else None

        if not transfer_uri:
            logger.error(
                "No se puede transferir: YEASTAR_IP no configurado. "
                "Define la variable de entorno YEASTAR_IP."
            )
            return False

        logger.info(
            "Iniciando SIP REFER | sala='%s' participante='%s' destino='%s'",
            room_name,
            participant_identity,
            transfer_uri,
        )

        try:
            await self._lk.sip.transfer_sip_participant(
                lkapi.TransferSIPParticipantRequest(
                    room_name=room_name,
                    participant_identity=participant_identity,
                    transfer_to=transfer_uri,
                    play_dialtone=True,  # reproduce tono mientras se establece la redirección
                )
            )
            logger.info(
                "SIP REFER enviado correctamente → participante '%s' redirigido a cola %s",
                participant_identity,
                ext,
            )
            return True

        except Exception as exc:
            logger.error(
                "Error al enviar SIP REFER para '%s': %s",
                participant_identity,
                exc,
                exc_info=True,
            )
            return False

    # ------------------------------------------------------------------
    # Limpieza
    # ------------------------------------------------------------------

    async def aclose(self) -> None:
        """Cierra el cliente LiveKit API de forma limpia."""
        await self._lk.aclose()
