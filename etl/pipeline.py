"""Orquestador del flujo ETL completo.

Coordina scrapers, limpieza y carga en PostgreSQL para todos los
datasets del proyecto CMG Chile Predictor.

Modos de ejecucion:
  backfill    — carga historica completa desde archivos en data/raw/
  incremental — actualizacion de un dia especifico (futuro, con Selenium)

Uso tipico:
    python etl/pipeline.py --mode backfill
    python etl/pipeline.py --mode incremental --date 2025-01-15
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import DB_URL                                        # noqa: E402
from etl.scrapers.cen_marginal import parse_tsv                  # noqa: E402
from etl.scrapers.cen_generation import parse_generation         # noqa: E402
from etl.scrapers.cen_reservoirs import parse_reservoirs         # noqa: E402
from etl.scrapers.weather import fetch_weather                   # noqa: E402
from etl.transform import clean_marginal_costs, clean_generation # noqa: E402
from etl.load import upsert_dataframe                            # noqa: E402

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────
DATA_RAW: Path = Path("data/raw")
BARRA: str = "Quillota 220kV"

_BACKFILL_WEATHER_START: str = "2019-01-01"
_BACKFILL_WEATHER_END: str = "2024-12-31"


# =============================================================================
# FUNCIONES PUBLICAS
# =============================================================================

def run_backfill(engine: Engine) -> dict[str, int]:
    """Ejecutar el backfill completo de todos los datasets.

    Pipeline en orden:
      1. CMG marginal_costs    (6 archivos TSV anuales)
      2. Generacion generation_by_tech (1 TSV multi-anio)
      3. Embalses reservoir_levels    (1 CSV historico mensual)
      4. Clima weather                (Open-Meteo API, 2019-2024)

    Cada paso es independiente: si uno falla, los restantes se ejecutan
    igualmente y el error queda registrado en el log.

    Args:
        engine: Engine de SQLAlchemy con conexion activa a cen_data.

    Returns:
        Diccionario con filas insertadas por tabla:
        {"cmg": N, "generation": N, "reservoirs": N, "weather": N}
    """
    resumen: dict[str, int] = {
        "cmg": 0,
        "generation": 0,
        "reservoirs": 0,
        "weather": 0,
    }

    # ── 1. CMG ────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PASO 1/4 — CMG (marginal_costs)")
    logger.info("=" * 60)

    archivos_cmg = sorted(DATA_RAW.glob("cmg_quillota_*.tsv"))
    if not archivos_cmg:
        logger.warning(
            "No se encontraron archivos 'cmg_quillota_*.tsv' en '%s'. "
            "Saltando CMG.",
            DATA_RAW.resolve(),
        )
    else:
        dfs_cmg: list[pd.DataFrame] = []
        for archivo in archivos_cmg:
            try:
                df = parse_tsv(archivo)
                dfs_cmg.append(df)
                logger.info("  Parseado: %s  (%d filas)", archivo.name, len(df))
            except Exception as exc:
                logger.error(
                    "  Error parseando '%s': %s", archivo.name, exc, exc_info=True
                )

        if dfs_cmg:
            df_cmg = pd.concat(dfs_cmg, ignore_index=True)
            logger.info("Total filas crudas CMG     : %d", len(df_cmg))

            df_cmg_limpio = clean_marginal_costs(df_cmg)
            logger.info("Filas tras limpieza        : %d", len(df_cmg_limpio))

            n = upsert_dataframe(
                df=df_cmg_limpio,
                table="marginal_costs",
                conflict_cols=["datetime", "barra"],
                engine=engine,
            )
            resumen["cmg"] = n

    # ── 2. Generacion ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PASO 2/4 — Generacion (generation_by_tech)")
    logger.info("=" * 60)

    filepath_gen = DATA_RAW / "gen_real_2021-2024.tsv"
    if not filepath_gen.exists():
        logger.warning(
            "Archivo '%s' no encontrado. Saltando generacion.", filepath_gen
        )
    else:
        try:
            df_gen = parse_generation(filepath_gen)
            logger.info("Filas crudas generacion    : %d", len(df_gen))

            df_gen_limpio = clean_generation(df_gen)
            logger.info("Filas tras limpieza        : %d", len(df_gen_limpio))

            n = upsert_dataframe(
                df=df_gen_limpio,
                table="generation_by_tech",
                conflict_cols=["datetime"],
                engine=engine,
            )
            resumen["generation"] = n
        except Exception as exc:
            logger.error(
                "Error en pipeline de generacion: %s", exc, exc_info=True
            )

    # ── 3. Embalses ───────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PASO 3/4 — Embalses (reservoir_levels)")
    logger.info("=" * 60)

    filepath_res = DATA_RAW / "reservoirs_historical.csv"
    if not filepath_res.exists():
        logger.warning(
            "Archivo '%s' no encontrado. Saltando embalses.", filepath_res
        )
    else:
        try:
            df_res = parse_reservoirs(filepath_res)
            logger.info("Registros mensuales parseados: %d", len(df_res))

            # avg_cota_msnm se almacena en energy_gwh como proxy del nivel
            # de embalses. La tabla no tiene columna separada para cota.
            df_res_db = df_res[["date"]].copy()
            df_res_db["energy_gwh"] = df_res["avg_cota_msnm"]

            n = upsert_dataframe(
                df=df_res_db,
                table="reservoir_levels",
                conflict_cols=["date"],
                engine=engine,
            )
            resumen["reservoirs"] = n
        except Exception as exc:
            logger.error(
                "Error en pipeline de embalses: %s", exc, exc_info=True
            )

    # ── 4. Clima ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PASO 4/4 — Clima (weather)")
    logger.info("=" * 60)

    try:
        df_weather = fetch_weather(_BACKFILL_WEATHER_START, _BACKFILL_WEATHER_END)
        logger.info("Registros descargados      : %d", len(df_weather))

        # La tabla weather tiene UNIQUE(date, region). La columna region no
        # viene de la API; se agrega aqui con el valor fijo "Santiago".
        df_weather["region"] = "Santiago"

        n = upsert_dataframe(
            df=df_weather,
            table="weather",
            conflict_cols=["date", "region"],
            engine=engine,
        )
        resumen["weather"] = n
    except Exception as exc:
        logger.error("Error en pipeline de clima: %s", exc, exc_info=True)

    return resumen


def run_incremental(fecha: str, engine: Engine) -> dict[str, int]:
    """Ejecutar actualizacion incremental para un dia especifico.

    Modo pendiente de implementacion. Se activara en la fase posterior
    cuando cen_demand.py con Selenium este disponible y el scheduler
    este configurado para ejecuciones diarias.

    Args:
        fecha: Fecha a actualizar, formato 'YYYY-MM-DD'.
        engine: Engine de SQLAlchemy con conexion activa a cen_data.

    Returns:
        Diccionario vacio (sin inserciones en este modo aun).
    """
    logger.warning(
        "Modo incremental no implementado aun. "
        "Se implementara en fase posterior con Selenium (cen_demand.py). "
        "Fecha solicitada: %s",
        fecha,
    )
    return {}


# =============================================================================
# EJECUCION DIRECTA
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Orquestador ETL — CMG Chile Predictor"
    )
    parser.add_argument(
        "--mode",
        choices=["backfill", "incremental"],
        default="backfill",
        help="Modo de ejecucion: 'backfill' (historico) o 'incremental' (un dia).",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Fecha para modo incremental (formato YYYY-MM-DD).",
    )
    args = parser.parse_args()

    if args.mode == "incremental" and args.date is None:
        parser.error("--date es requerido para el modo incremental.")

    _engine = create_engine(DB_URL, pool_pre_ping=True)
    _inicio = time.perf_counter()

    try:
        if args.mode == "backfill":
            logger.info("Iniciando pipeline ETL — modo BACKFILL")
            _resumen = run_backfill(_engine)
        else:
            logger.info(
                "Iniciando pipeline ETL — modo INCREMENTAL  fecha=%s", args.date
            )
            _resumen = run_incremental(args.date, _engine)

        _duracion = time.perf_counter() - _inicio

        logger.info("=" * 60)
        logger.info("RESUMEN FINAL")
        logger.info("=" * 60)
        for _tabla, _n in _resumen.items():
            logger.info("  %-20s %d filas insertadas", _tabla + ":", _n)
        logger.info("Tiempo total: %.1f segundos", _duracion)

    except KeyboardInterrupt:
        logger.warning("Pipeline interrumpido por el usuario (KeyboardInterrupt).")
        sys.exit(0)
    finally:
        _engine.dispose()
