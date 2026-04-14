"""Scraper: generacion horaria por tecnologia CEN.

Parsea el archivo TSV de generacion real del Coordinador Electrico Nacional,
agrega la potencia por tecnologia y hora, y carga el resultado en la tabla
generation_by_tech de PostgreSQL.

Formato de entrada: WIDE — una fila por central y dia, columnas Hora 1..Hora 24.
Formato de salida: una fila por hora con MW agregados por tecnologia.
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
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import DB_URL  # noqa: E402

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────
# Columnas de hora en el archivo WIDE del CEN
_COLUMNAS_HORA: list[str] = [f"Hora {i}" for i in range(1, 25)]

# Columnas de tecnologia en la tabla generation_by_tech
_COLUMNAS_TECH: list[str] = [
    "gen_solar_mw",
    "gen_wind_mw",
    "gen_hydro_reservoir_mw",
    "gen_hydro_runofriver_mw",
    "gen_gas_mw",
    "gen_coal_mw",
    "gen_diesel_mw",
]

# Columnas minimas requeridas en el TSV (ademas de Hora 1..24)
_COLUMNAS_REQUERIDAS: list[str] = ["Tipo", "Subtipo", "Fecha"]


# =============================================================================
# HELPERS PRIVADOS
# =============================================================================

def _asignar_tech_col(df: pd.DataFrame) -> pd.DataFrame:
    """Agregar columna 'tech_col' con el nombre de columna BD segun Tipo/Subtipo.

    Usa asignacion vectorizada. Las filas que no corresponden a ninguna
    tecnologia de interes quedan con tech_col=None y se filtran despues.

    Args:
        df: DataFrame con columnas 'Tipo' y 'Subtipo' ya strip-peadas.

    Returns:
        El mismo DataFrame con la columna 'tech_col' agregada.
    """
    tipo = df["Tipo"]
    sub = df["Subtipo"]

    df["tech_col"] = None

    df.loc[tipo == "Solar",   "tech_col"] = "gen_solar_mw"
    df.loc[tipo == "Eólicas", "tech_col"] = "gen_wind_mw"

    df.loc[(tipo == "Hidroeléctricas") & (sub == "Embalse"), "tech_col"] = "gen_hydro_reservoir_mw"
    df.loc[(tipo == "Hidroeléctricas") & (sub == "Pasada"),  "tech_col"] = "gen_hydro_runofriver_mw"

    df.loc[(tipo == "Termoeléctricas") & (sub == "Gas Natural"), "tech_col"] = "gen_gas_mw"
    df.loc[(tipo == "Termoeléctricas") & (sub == "Carbón"),      "tech_col"] = "gen_coal_mw"
    df.loc[(tipo == "Termoeléctricas") & (sub == "Diésel"),      "tech_col"] = "gen_diesel_mw"

    return df


# =============================================================================
# FUNCIONES PUBLICAS
# =============================================================================

def parse_generation(filepath: Union[str, Path]) -> pd.DataFrame:
    """Leer y transformar el TSV de generacion real del CEN.

    Pipeline:
      1. Lee el TSV (encoding utf-8-sig para BOM, dtype=str).
      2. Filtra las filas de las tecnologias de interes.
      3. Convierte decimales con coma europea en columnas Hora 1..24.
      4. Melt wide→long: (central, fecha, hora, mw).
      5. Construye datetime en UTC desde fecha + hora.
      6. Agrega (suma) MW por (datetime, tecnologia).
      7. Pivota a una fila por datetime con columnas por tecnologia.
      8. Calcula gen_total_mw como suma de las 7 columnas de tecnologia.
      9. Rellena NaN con 0.0.

    Args:
        filepath: Ruta al archivo TSV descargado del CEN.
            Nombre esperado: gen_real_2021-2024.tsv

    Returns:
        DataFrame con columnas:
        [datetime, gen_solar_mw, gen_wind_mw, gen_hydro_reservoir_mw,
         gen_hydro_runofriver_mw, gen_gas_mw, gen_coal_mw,
         gen_diesel_mw, gen_total_mw].
        - datetime: TIMESTAMPTZ en UTC, una fila por hora.
        - MW: float64, suma de todas las centrales de esa tecnologia.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si faltan columnas requeridas en el archivo.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {filepath}")

    logger.debug("Leyendo archivo: %s", filepath.name)

    # ── 1. Lectura ─────────────────────────────────────────────────────────────
    # utf-8-sig elimina el BOM automaticamente; dtype=str evita conversion
    # prematura de decimales con coma
    df = pd.read_csv(
        filepath,
        sep="\t",
        encoding="utf-8-sig",
        dtype=str,
        na_values=["", "NA", "N/A", "null"],
        keep_default_na=False,
    )

    # Normalizar nombres de columna (quitar espacios extra)
    df.columns = df.columns.str.strip()

    # Validar columnas minimas requeridas
    faltantes = set(_COLUMNAS_REQUERIDAS + _COLUMNAS_HORA) - set(df.columns)
    if faltantes:
        raise ValueError(
            f"Columnas faltantes en '{filepath.name}': {sorted(faltantes)}. "
            f"Columnas encontradas: {list(df.columns)}"
        )

    # ── 2. Filtro de tecnologias ───────────────────────────────────────────────
    df["Tipo"]    = df["Tipo"].str.strip()
    df["Subtipo"] = df["Subtipo"].str.strip()
    df["Fecha"]   = df["Fecha"].str.strip()

    total_archivo = len(df)
    df = _asignar_tech_col(df)
    df = df[df["tech_col"].notna()].copy()

    if df.empty:
        logger.warning(
            "No se encontraron filas con tecnologias de interes en '%s'.",
            filepath.name,
        )
        return pd.DataFrame(columns=["datetime"] + _COLUMNAS_TECH + ["gen_total_mw"])

    logger.debug(
        "Filas con tecnologias de interes: %d (de %d totales)",
        len(df),
        total_archivo,
    )

    # ── 3. Conversion de decimales en columnas de hora ─────────────────────────
    for col in _COLUMNAS_HORA:
        df[col] = (
            df[col]
            .str.strip()
            .str.replace(",", ".", regex=False)
            .pipe(pd.to_numeric, errors="coerce")
        )

    # ── 4. Melt: WIDE → LONG ───────────────────────────────────────────────────
    df_long = df.melt(
        id_vars=["Fecha", "tech_col"],
        value_vars=_COLUMNAS_HORA,
        var_name="hora_col",   # "Hora 1", "Hora 2", ..., "Hora 24"
        value_name="mw",
    )

    # ── 5. Construccion del datetime: hora 1 = 00:00, hora 24 = 23:00 ─────────
    hora_num = (
        df_long["hora_col"]
        .str.extract(r"(\d+)", expand=False)
        .astype(int)
    )
    fecha_base = pd.to_datetime(
        df_long["Fecha"],
        format="%Y-%m-%d",
        errors="coerce",
    )
    df_long["datetime"] = fecha_base + pd.to_timedelta(hora_num - 1, unit="h")

    # Mismo tratamiento DST que cen_marginal.py:
    # ambiguous=True → toda hora ambigua es DST (UTC-3).
    # El CEN no registra la hora repetida del cambio horario de abril.
    df_long["datetime"] = (
        df_long["datetime"]
        .dt.tz_localize(
            "America/Santiago",
            ambiguous=True,
            nonexistent="shift_forward",
        )
        .dt.tz_convert("UTC")
    )

    # Descartar filas con datetime invalido (fechas malformadas en el TSV)
    df_long = df_long.dropna(subset=["datetime", "mw"])

    # ── 6. Agregacion: suma de MW por (datetime, tecnologia) ──────────────────
    agg = (
        df_long
        .groupby(["datetime", "tech_col"], observed=True)["mw"]
        .sum()
        .reset_index()
    )

    # ── 7. Pivot: una fila por datetime, una columna por tecnologia ───────────
    pivot = (
        agg
        .pivot(index="datetime", columns="tech_col", values="mw")
        .reset_index()
    )
    pivot.columns.name = None

    # Garantizar que todas las columnas de tecnologia existen
    for col in _COLUMNAS_TECH:
        if col not in pivot.columns:
            pivot[col] = 0.0

    # ── 8 & 9. Total y relleno de NaN ─────────────────────────────────────────
    pivot[_COLUMNAS_TECH] = pivot[_COLUMNAS_TECH].fillna(0.0)
    pivot["gen_total_mw"] = pivot[_COLUMNAS_TECH].sum(axis=1)

    col_orden = ["datetime"] + _COLUMNAS_TECH + ["gen_total_mw"]
    return pivot[col_orden].sort_values("datetime").reset_index(drop=True)


def load_to_db(df: pd.DataFrame, engine: Engine) -> int:
    """Insertar DataFrame en la tabla generation_by_tech usando upsert.

    Utiliza INSERT ... ON CONFLICT (datetime) DO NOTHING para evitar
    duplicados en ejecuciones repetidas del backfill.

    Args:
        df: DataFrame con columnas [datetime, gen_solar_mw, ...].
            Debe provenir de parse_generation() para garantizar tipos correctos.
        engine: Engine de SQLAlchemy conectado a la base de datos cen_data.

    Returns:
        Numero de filas efectivamente insertadas.
    """
    if df.empty:
        logger.debug("DataFrame vacio, sin filas que insertar.")
        return 0

    records = df.to_dict(orient="records")

    sql = text("""
        INSERT INTO generation_by_tech (
            datetime,
            gen_solar_mw,
            gen_wind_mw,
            gen_hydro_reservoir_mw,
            gen_hydro_runofriver_mw,
            gen_gas_mw,
            gen_coal_mw,
            gen_diesel_mw,
            gen_total_mw
        )
        VALUES (
            :datetime,
            :gen_solar_mw,
            :gen_wind_mw,
            :gen_hydro_reservoir_mw,
            :gen_hydro_runofriver_mw,
            :gen_gas_mw,
            :gen_coal_mw,
            :gen_diesel_mw,
            :gen_total_mw
        )
        ON CONFLICT (datetime) DO NOTHING
    """)

    with engine.begin() as conn:
        result = conn.execute(sql, records)
        insertadas = result.rowcount

    logger.debug("Filas insertadas en generation_by_tech: %d", insertadas)
    return insertadas


def run_backfill(filepath: str = "data/raw/gen_real_2021-2024.tsv") -> None:
    """Parsear el archivo de generacion historica y cargarlo en PostgreSQL.

    Args:
        filepath: Ruta al archivo TSV de generacion real del CEN.
            Por defecto 'data/raw/gen_real_2021-2024.tsv'.
    """
    try:
        logger.info("Parseando: %s", filepath)
        df = parse_generation(filepath)

        if df.empty:
            logger.warning("El archivo no produjo registros. Abortando carga.")
            return

        fecha_min = df["datetime"].min()
        fecha_max = df["datetime"].max()
        logger.info("Registros parseados : %d", len(df))
        logger.info("Rango de fechas     : %s → %s", fecha_min, fecha_max)

        engine = create_engine(DB_URL, pool_pre_ping=True)
        insertados = load_to_db(df, engine)
        engine.dispose()

        logger.info("Insertados en BD    : %d", insertados)

    except Exception as exc:
        logger.error("Error en backfill de generacion: %s", exc, exc_info=True)
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
