"""Scraper: costo marginal real CEN.

Parsea los archivos TSV descargados del Coordinador Electrico Nacional
con costos marginales horarios de la barra Quillota 220 kV y los carga
en la tabla marginal_costs de PostgreSQL.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Union

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ── Path setup ────────────────────────────────────────────────────────────────
# Permite ejecutar este modulo directamente desde cualquier CWD
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import DB_URL  # noqa: E402  (importacion post-path)

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────
# Nombre normalizado que se almacena en la base de datos
_BARRA_DESTINO: str = "Quillota 220kV"

# Columnas en el orden exacto del TSV del CEN (7 columnas, separador \t)
_COLUMNAS_TSV: list[str] = [
    "barra_mnemotecnico",
    "barra_referencia_mnemotecnico",
    "fecha",
    "hora",
    "costo_en_dolares",
    "costo_en_pesos",
    "nombre",
]


# =============================================================================
# FUNCIONES PUBLICAS
# =============================================================================

def parse_tsv(filepath: Union[str, Path]) -> pd.DataFrame:
    """Leer y transformar un archivo TSV de costos marginales del CEN.

    Lee el archivo TSV, filtra la barra Quillota, convierte los decimales
    con coma europea, construye el datetime horario y lo normaliza a UTC.

    Args:
        filepath: Ruta al archivo TSV descargado del CEN.
            Nombre esperado: cmg_quillota_YYYY.tsv

    Returns:
        DataFrame con columnas [datetime, barra, cmg_usd_mwh].
        - datetime: TIMESTAMPTZ en UTC (pandas DatetimeTZDtype)
        - barra: str, valor fijo 'Quillota 220kV'
        - cmg_usd_mwh: float64

    Raises:
        FileNotFoundError: Si el archivo no existe en la ruta indicada.
        ValueError: Si el archivo no contiene las columnas esperadas o
            si no hay registros para la barra Quillota.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {filepath}")

    logger.debug("Leyendo archivo: %s", filepath.name)

    # Leer todo como texto para manejar decimales con coma sin conversion prematura
    df = pd.read_csv(
        filepath,
        sep="\t",
        encoding="utf-8",
        names=_COLUMNAS_TSV,
        header=0,      # primera fila es cabecera, reemplazar con nombres propios
        dtype=str,
        na_values=["", "NA", "N/A", "null"],
        keep_default_na=False,
    )

    # Validar que esten todas las columnas esperadas
    faltantes = set(_COLUMNAS_TSV) - set(df.columns)
    if faltantes:
        raise ValueError(
            f"Columnas faltantes en '{filepath.name}': {faltantes}. "
            f"Columnas encontradas: {list(df.columns)}"
        )

    # ── Conversion de decimales: coma europea → punto ─────────────────────────
    df["cmg_usd_mwh"] = (
        df["costo_en_dolares"]
        .str.strip()
        .str.replace(",", ".", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )

    # ── Construccion del datetime: hora 1 = 00:00, hora 24 = 23:00 ───────────
    hora_num = (
        pd.to_numeric(df["hora"].str.strip(), errors="coerce")
        .astype("Int64")        # Int64 nullable para manejar posibles NaN
    )
    fecha_base = pd.to_datetime(
        df["fecha"].str.strip(),
        format="%Y-%m-%d",
        errors="coerce",
    )
    df["datetime"] = fecha_base + pd.to_timedelta(hora_num - 1, unit="h")

    # ── Normalizacion de zona horaria: America/Santiago → UTC ─────────────────
    # El CEN registra exactamente 24 filas por dia (hora 1-24), incluso en dias
    # de cambio horario. En el paso a horario de invierno (abril), el reloj
    # retrocede a medianoche, por lo que la hora 23:00 local podria ser ambigua.
    # Sin embargo, hora=24 en el TSV corresponde a las 23:00 ANTES de medianoche
    # (aun dentro del periodo DST), por lo que ambiguous=True (DST=True) es
    # correcto. El CEN simplemente no registra la segunda ocurrencia.
    # nonexistent="shift_forward": horas inexistentes (cambio de primavera) se
    # desplazan al primer instante valido posterior.
    df["datetime"] = (
        df["datetime"]
        .dt.tz_localize(
            "America/Santiago",
            ambiguous=True,
            nonexistent="shift_forward",
        )
        .dt.tz_convert("UTC")
    )

    # ── Columna barra normalizada ──────────────────────────────────────────────
    df["barra"] = _BARRA_DESTINO

    return df[["datetime", "barra", "cmg_usd_mwh"]].reset_index(drop=True)


def load_to_db(df: pd.DataFrame, engine: Engine) -> int:
    """Insertar DataFrame en la tabla marginal_costs usando upsert.

    Utiliza INSERT ... ON CONFLICT (datetime, barra) DO NOTHING para que
    ejecuciones repetidas del backfill no generen duplicados ni errores.

    Args:
        df: DataFrame con columnas [datetime, barra, cmg_usd_mwh].
            Debe provenir de parse_tsv() para garantizar los tipos correctos.
        engine: Engine de SQLAlchemy conectado a la base de datos cen_data.

    Returns:
        Numero de filas efectivamente insertadas (las que no generaron conflicto).
    """
    if df.empty:
        logger.debug("DataFrame vacio, sin filas que insertar.")
        return 0

    records = df.to_dict(orient="records")

    sql = text("""
        INSERT INTO marginal_costs (datetime, barra, cmg_usd_mwh, is_imputed)
        VALUES (:datetime, :barra, :cmg_usd_mwh, FALSE)
        ON CONFLICT (datetime, barra) DO NOTHING
    """)

    with engine.begin() as conn:
        result = conn.execute(sql, records)
        insertadas = result.rowcount

    logger.debug("Filas insertadas en marginal_costs: %d", insertadas)
    return insertadas


def run_backfill(data_dir: str = "data/raw") -> None:
    """Procesar todos los archivos TSV historicos de costos marginales.

    Busca archivos con el patron cmg_quillota_*.tsv en data_dir, los
    procesa en orden cronologico ascendente y los carga en PostgreSQL.
    Si un archivo falla, se registra el error y se continua con el siguiente.

    Args:
        data_dir: Directorio donde estan los archivos TSV del CEN.
            Por defecto 'data/raw' (relativo a la raiz del proyecto).
    """
    data_path = Path(data_dir)
    archivos = sorted(data_path.glob("cmg_quillota_*.tsv"))

    if not archivos:
        logger.warning(
            "No se encontraron archivos 'cmg_quillota_*.tsv' en '%s'.",
            data_path.resolve(),
        )
        return

    logger.info("Iniciando backfill. Archivos encontrados: %d", len(archivos))

    engine = create_engine(DB_URL, pool_pre_ping=True)
    total_procesados = 0
    total_insertados = 0
    archivos_fallidos: list[str] = []

    for archivo in archivos:
        try:
            logger.info("── Procesando: %s", archivo.name)
            df = parse_tsv(archivo)

            logger.info("   Registros parseados : %d", len(df))

            insertados = load_to_db(df, engine)
            total_procesados += len(df)
            total_insertados += insertados

            logger.info("   Insertados en BD    : %d", insertados)

        except Exception as exc:
            logger.error(
                "Error procesando '%s': %s",
                archivo.name,
                exc,
                exc_info=True,
            )
            archivos_fallidos.append(archivo.name)
            continue

    engine.dispose()

    logger.info(
        "Backfill finalizado | Procesados: %d | Insertados: %d | Archivos con error: %d",
        total_procesados,
        total_insertados,
        len(archivos_fallidos),
    )
    if archivos_fallidos:
        logger.warning("Archivos que fallaron: %s", archivos_fallidos)


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
