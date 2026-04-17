"""Scraper: temperatura y precipitaciones (Open-Meteo).

Descarga datos meteorologicos diarios para Santiago de Chile desde la
API historica de Open-Meteo (gratuita, sin clave API requerida) y los
carga en la tabla weather de PostgreSQL.

API utilizada: https://archive-api.open-meteo.com/v1/archive
Coordenadas: Santiago de Chile (-33.45, -70.67)
Variables: temperatura maxima diaria (°C) y precipitacion acumulada (mm)
"""

import logging
import sys
import time
from datetime import date
from pathlib import Path
from typing import Union

import pandas as pd
import requests
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import DB_URL  # noqa: E402

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────
_API_URL: str = "https://archive-api.open-meteo.com/v1/archive"
_LATITUDE: float = -33.45
_LONGITUDE: float = -70.67
_TIMEZONE: str = "America/Santiago"
_REGION: str = "Santiago"
_TIMEOUT_S: int = 30

_BACKFILL_START: str = "2019-01-01"
_BACKFILL_END: str = "2024-12-31"


# =============================================================================
# FUNCIONES PUBLICAS
# =============================================================================

def fetch_weather(
    start_date: Union[str, date],
    end_date: Union[str, date],
) -> pd.DataFrame:
    """Descargar datos meteorologicos diarios desde la API de Open-Meteo.

    Realiza un GET a la API historica de Open-Meteo para Santiago de Chile
    y retorna temperatura maxima y precipitacion acumulada por dia.

    Args:
        start_date: Fecha de inicio del rango, formato 'YYYY-MM-DD' o
            objeto datetime.date.
        end_date: Fecha de fin del rango, formato 'YYYY-MM-DD' o
            objeto datetime.date.

    Returns:
        DataFrame con columnas [date, temp_max_c, precip_mm].
        - date: datetime.date sin hora ni zona horaria.
        - temp_max_c: temperatura maxima diaria en grados Celsius.
        - precip_mm: precipitacion acumulada diaria en milimetros.

    Raises:
        requests.HTTPError: Si la API responde con un codigo de error HTTP.
        requests.Timeout: Si la peticion supera el timeout de 30 segundos.
        ValueError: Si la respuesta JSON no tiene la estructura esperada.
    """
    params = {
        "latitude": _LATITUDE,
        "longitude": _LONGITUDE,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "daily": "temperature_2m_max,precipitation_sum",
        "timezone": _TIMEZONE,
    }

    logger.debug(
        "GET %s  start=%s  end=%s", _API_URL, start_date, end_date
    )

    try:
        response = requests.get(_API_URL, params=params, timeout=_TIMEOUT_S)
        response.raise_for_status()
    except requests.Timeout:
        raise requests.Timeout(
            f"La API de Open-Meteo no respondio en {_TIMEOUT_S}s. "
            f"Rango solicitado: {start_date} → {end_date}"
        )
    except requests.HTTPError as exc:
        raise requests.HTTPError(
            f"Error HTTP {response.status_code} al consultar Open-Meteo: "
            f"{response.text[:200]}"
        ) from exc

    payload = response.json()

    # Validar estructura minima de la respuesta
    if "daily" not in payload:
        raise ValueError(
            f"Respuesta de Open-Meteo sin campo 'daily'. "
            f"Claves recibidas: {list(payload.keys())}"
        )
    daily = payload["daily"]
    for campo in ("time", "temperature_2m_max", "precipitation_sum"):
        if campo not in daily:
            raise ValueError(
                f"Campo '{campo}' ausente en 'daily'. "
                f"Claves recibidas: {list(daily.keys())}"
            )

    df = pd.DataFrame({
        "date": pd.to_datetime(daily["time"], format="%Y-%m-%d").date,
        "temp_max_c": pd.to_numeric(
            pd.Series(daily["temperature_2m_max"]), errors="coerce"
        ),
        "precip_mm": pd.to_numeric(
            pd.Series(daily["precipitation_sum"]), errors="coerce"
        ),
    })

    logger.debug("Registros descargados: %d", len(df))
    return df


def load_to_db(df: pd.DataFrame, engine: Engine) -> int:
    """Insertar DataFrame en la tabla weather usando upsert.

    Asigna la region 'Santiago' a todos los registros e inserta usando
    ON CONFLICT (date, region) DO NOTHING para idempotencia.

    Args:
        df: DataFrame con columnas [date, temp_max_c, precip_mm].
            Debe provenir de fetch_weather().
        engine: Engine de SQLAlchemy conectado a la base de datos cen_data.

    Returns:
        Numero de filas efectivamente insertadas.
    """
    if df.empty:
        logger.debug("DataFrame vacio, sin filas que insertar.")
        return 0

    records = [
        {
            "date": row["date"],
            "region": _REGION,
            "temp_max_c": row["temp_max_c"],
            "precip_mm": row["precip_mm"],
        }
        for row in df.to_dict(orient="records")
    ]

    sql = text("""
        INSERT INTO weather (date, region, temp_max_c, precip_mm)
        VALUES (:date, :region, :temp_max_c, :precip_mm)
        ON CONFLICT (date, region) DO NOTHING
    """)

    with engine.begin() as conn:
        result = conn.execute(sql, records)
        insertadas = result.rowcount

    logger.debug("Filas insertadas en weather: %d", insertadas)
    return insertadas


def run_backfill() -> None:
    """Descargar datos meteorologicos historicos y cargarlos en PostgreSQL.

    Descarga el rango completo 2019-01-01 → 2024-12-31 en una sola
    peticion a la API de Open-Meteo y los inserta en la tabla weather.
    """
    try:
        logger.info(
            "Descargando datos meteorologicos: %s → %s",
            _BACKFILL_START,
            _BACKFILL_END,
        )
        df = fetch_weather(_BACKFILL_START, _BACKFILL_END)

        logger.info("Registros descargados : %d", len(df))
        logger.info(
            "Rango de fechas       : %s → %s",
            df["date"].min(),
            df["date"].max(),
        )
        logger.info(
            "Temperatura max media : %.1f °C",
            df["temp_max_c"].mean(),
        )
        logger.info(
            "Precipitacion total   : %.1f mm",
            df["precip_mm"].sum(),
        )

        engine = create_engine(DB_URL, pool_pre_ping=True)
        insertados = load_to_db(df, engine)
        engine.dispose()

        logger.info("Insertados en BD      : %d", insertados)

    except Exception as exc:
        logger.error("Error en backfill de weather: %s", exc, exc_info=True)
        raise


# =============================================================================
# EJECUCION DIRECTA
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    inicio = time.perf_counter()
    run_backfill()
    duracion = time.perf_counter() - inicio

    logger.info("Tiempo total de ejecucion: %.1f segundos", duracion)
