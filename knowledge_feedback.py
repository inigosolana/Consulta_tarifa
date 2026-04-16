"""
knowledge_feedback.py
=====================
Módulo de "Bucle de Retroalimentación de Conocimiento" para el agente de voz.

Responsabilidades:
  1. Tool de búsqueda de tarifas en el CSV (pandas).
  2. Extracción de información comercial de la transcripción via LLM.
  3. Actualización del CSV con el nuevo conocimiento.
"""

from __future__ import annotations

import logging
import os
from typing import Annotated

import pandas as pd
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

CSV_PATH = os.path.join(
    os.path.dirname(__file__),
    "informe_servicios 20260313.xlsx - informe CRM.csv",
)

COL_NOMBRE = "Nombre"
COL_INFO   = "Información comercial"

_openai_client: AsyncOpenAI | None = None


def _get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI()   # usa OPENAI_API_KEY del entorno
    return _openai_client


# ---------------------------------------------------------------------------
# Helpers de CSV
# ---------------------------------------------------------------------------

def _load_csv() -> pd.DataFrame:
    """Carga el CSV preservando encoding y tipos."""
    return pd.read_csv(CSV_PATH, dtype=str, encoding="utf-8-sig")


def _save_csv(df: pd.DataFrame) -> None:
    """Guarda el DataFrame en el mismo CSV."""
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# 1. TOOL: buscar_tarifa
# ---------------------------------------------------------------------------

async def buscar_tarifa(
    nombre_tarifa: Annotated[
        str,
        "Nombre exacto o aproximado de la tarifa que el comercial ha preguntado.",
    ],
) -> str:
    """
    Busca una tarifa en el CRM por su nombre.

    Devuelve la información comercial si existe, o una instrucción para
    transferir la llamada si el campo está vacío.
    """
    try:
        df = _load_csv()
    except FileNotFoundError:
        logger.error("CSV no encontrado en: %s", CSV_PATH)
        return (
            "No puedo acceder al CRM en este momento. "
            "Por favor, transfiere la llamada a un agente humano."
        )

    # Búsqueda flexible: coincidencia exacta → parcial (case-insensitive)
    mask_exacta = df[COL_NOMBRE].str.strip().str.lower() == nombre_tarifa.strip().lower()
    fila = df[mask_exacta]

    if fila.empty:
        mask_parcial = df[COL_NOMBRE].str.contains(nombre_tarifa, case=False, na=False)
        fila = df[mask_parcial]

    if fila.empty:
        logger.info("Tarifa no encontrada en CRM: '%s'", nombre_tarifa)
        return (
            f"No encontré la tarifa '{nombre_tarifa}' en el CRM. "
            "Voy a transferirte con un agente humano que podrá ayudarte."
        )

    info = fila.iloc[0][COL_INFO]

    # Campo vacío, NaN o sólo espacios → transferir
    if pd.isna(info) or str(info).strip() == "":
        logger.info("Tarifa '%s' encontrada pero sin información comercial.", nombre_tarifa)
        return (
            f"Tengo registrada la tarifa '{nombre_tarifa}' pero aún no "
            "dispongo de su información comercial detallada. "
            "Voy a transferirte con un agente humano para que te lo explique."
        )

    logger.info("Tarifa '%s' encontrada con información. Devolviendo al agente.", nombre_tarifa)
    return str(info).strip()


# ---------------------------------------------------------------------------
# 2. Extractor LLM: analizar_transcripcion
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """\
Eres un asistente especializado en extraer información comercial de transcripciones de llamadas.

Se te proporcionará la transcripción completa de una llamada en la que un agente humano \
explicó a un comercial los detalles de una tarifa telefónica específica.

Tu única tarea es extraer, de forma concisa y fiel, la explicación técnico-comercial que \
el agente humano dio sobre la tarifa. Incluye precios, condiciones, velocidades, permanencias \
y cualquier dato relevante que el agente haya mencionado.

REGLAS ESTRICTAS:
- Si la transcripción NO contiene una explicación clara de la tarifa (llamada cortada, \
  tema diferente, conversación incompleta), responde únicamente con: N/A
- No añadas introducciones, despedidas ni comentarios propios.
- Responde en español.
- Máximo 400 palabras.

TARIFA CONSULTADA: {nombre_tarifa}

TRANSCRIPCIÓN:
{transcripcion}
"""


async def analizar_transcripcion(
    nombre_tarifa: str,
    transcripcion: str,
) -> str:
    """
    Usa GPT para extraer la información comercial de la transcripción.

    Devuelve el texto extraído o "N/A" si no hay información válida.
    """
    prompt = _EXTRACTION_PROMPT.format(
        nombre_tarifa=nombre_tarifa,
        transcripcion=transcripcion,
    )

    client = _get_openai_client()
    logger.info("Enviando transcripción al LLM para extracción de tarifa '%s'...", nombre_tarifa)

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=600,
    )

    resultado = response.choices[0].message.content.strip()
    logger.info("LLM devolvió: %.80s...", resultado)
    return resultado


# ---------------------------------------------------------------------------
# 3. Actualizar CSV con nuevo conocimiento
# ---------------------------------------------------------------------------

async def actualizar_crm(nombre_tarifa: str, nueva_info: str) -> bool:
    """
    Escribe `nueva_info` en la celda `Información comercial` de `nombre_tarifa`.

    Devuelve True si la actualización fue exitosa.
    """
    if not nueva_info or nueva_info.strip().upper() == "N/A":
        logger.info(
            "LLM no extrajo información válida para '%s'. CRM no modificado.",
            nombre_tarifa,
        )
        return False

    try:
        df = _load_csv()
    except FileNotFoundError:
        logger.error("No se puede actualizar el CRM: archivo no encontrado.")
        return False

    mask = df[COL_NOMBRE].str.strip().str.lower() == nombre_tarifa.strip().lower()

    if not mask.any():
        logger.warning(
            "No se encontró '%s' en el CRM al intentar actualizar.", nombre_tarifa
        )
        return False

    df.loc[mask, COL_INFO] = nueva_info
    _save_csv(df)

    logger.info(
        "=== CRM ACTUALIZADO === Tarifa: '%s' | Nuevo texto (%d chars): %.120s...",
        nombre_tarifa,
        len(nueva_info),
        nueva_info,
    )
    return True


# ---------------------------------------------------------------------------
# 4. Función orquestadora post-llamada
# ---------------------------------------------------------------------------

async def procesar_post_llamada(
    nombre_tarifa: str,
    transcripcion: str,
) -> None:
    """
    Punto de entrada principal para el evento de desconexión.

    Extrae la información de la transcripción y actualiza el CRM.
    """
    logger.info(
        "--- POST-LLAMADA: procesando transcripción para tarifa '%s' ---",
        nombre_tarifa,
    )

    if not transcripcion.strip():
        logger.info("Transcripción vacía. Nada que procesar.")
        return

    nueva_info = await analizar_transcripcion(nombre_tarifa, transcripcion)
    actualizado = await actualizar_crm(nombre_tarifa, nueva_info)

    if actualizado:
        print(
            f"\n[CRM] ✔ Tarifa '{nombre_tarifa}' actualizada con nueva información comercial.\n"
        )
    else:
        print(
            f"\n[CRM] ✘ No se actualizó el CRM para '{nombre_tarifa}' "
            f"(sin información válida o tarifa no encontrada).\n"
        )
