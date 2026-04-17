"""Scraper: cota diaria de embalses CEN.

Parsea el archivo CSV historico de niveles de embalses del Coordinador
Electrico Nacional, agrega por mes calculando el promedio de cota de los
embalses principales, y carga el resultado en la tabla reservoir_levels
de PostgreSQL.

Resolucion de los datos: MENSUAL (una fila por embalse por mes).
En el feature engineering se propagaran a resolucion diaria/horaria
mediante forward fill antes de unirlos con los datos horarios.

Nota: el campo energy_gwh de la tabla reservoir_levels almacena
avg_cota_msnm como proxy, ya que este dataset no incluye energia_gwh
directamente. La cota es el mejor indicador disponible del nivel de
los embalses en esta fuente.
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
# Embalses principales del SEN a incluir en la agregacion
_EMBALSES: set[str] = {
    "Colbún",
    "Laja",
    "Maule",
    "Rapel",
    "Ralco",
    "Chapo",
    "Melado",
    "La Invernada",
}

# Año minimo a incluir (alineado con el inicio del backfill del proyecto)
_ANIO_INICIO: int = 2019


# =============================================================================
# FUNCIONES PUBLICAS
# =============================================================================

def parse_reservoirs(filepath: Union[str, Path]) -> pd.DataFrame:
    """Leer y transformar el CSV historico de niveles de embalses.

    Pipeline:
      1. Lee el CSV (latin-1, separador ;, dtype=str).
      2. Limpia posible BOM residual en nombres de columna.
      3. Renombra la columna de fecha a 'fecha'.
      4. Filtra los 8 embalses de interes.
      5. Convierte cota_msnm a float (decimales con coma europea).
      6. Parsea fecha con format='%d/%m/%Y'.
      7. Filtra registros desde 2019 en adelante.
      8. Agrega: promedio de cota_msnm por mes, contando embalses distintos.
      9. Construye 'date' como el primer dia del mes.

    Args:
        filepath: Ruta al archivo CSV descargado del CEN.
            Nombre esperado: reservoirs_historical.csv

    Returns:
        DataFrame con columnas [date, avg_cota_msnm, n_embalses].
        - date: fecha del primer dia del mes (sin zona horaria).
        - avg_cota_msnm: promedio de cota en metros sobre nivel del mar.
        - n_embalses: cantidad de embalses que contribuyeron ese mes.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si faltan columnas requeridas en el archivo.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {filepath}")

    logger.debug("Leyendo archivo: %s", filepath.name)

    # ── 1. Lectura ─────────────────────────────────────────────────────────────
    df = pd.read_csv(
        filepath,
        sep=";",
        encoding="latin-1",
        dtype=str,
        na_values=["", "NA", "N/A", "null"],
        keep_default_na=False,
    )

    # ── 2. Limpiar BOM residual en nombres de columna ─────────────────────────
    # latin-1 no elimina el BOM automaticamente; puede quedar como '\ufeff'
    # o como los bytes EF BB BF mal interpretados en la primera columna
    df.columns = (
        df.columns
        .str.strip()
        .str.replace("\ufeff", "", regex=False)
        .str.replace("\xef\xbb\xbf", "", regex=False)
    )

    # ── 3. Renombrar columna de fecha ─────────────────────────────────────────
    # El archivo puede llamarla 'fecha', 'Fecha' u otras variantes
    col_fecha_original = next(
        (c for c in df.columns if c.lower() == "fecha"),
        None,
    )
    if col_fecha_original is None:
        raise ValueError(
            f"No se encontro columna 'fecha' en '{filepath.name}'. "
            f"Columnas disponibles: {list(df.columns)}"
        )
    df = df.rename(columns={col_fecha_original: "fecha"})

    # Validar columnas minimas requeridas
    _requeridas = {"fecha", "central", "cota_msnm"}
    faltantes = _requeridas - set(df.columns)
    if faltantes:
        raise ValueError(
            f"Columnas faltantes en '{filepath.name}': {sorted(faltantes)}. "
            f"Columnas encontradas: {list(df.columns)}"
        )

    # ── 4. Filtrar embalses de interes ─────────────────────────────────────────
    df["central"] = df["central"].str.strip()
    total_archivo = len(df)
    df = df[df["central"].isin(_EMBALSES)].copy()

    logger.debug(
        "Filas de embalses de interes: %d (de %d totales)",
        len(df),
        total_archivo,
    )

    if df.empty:
        logger.warning(
            "Ninguna fila coincide con los embalses de interes en '%s'. "
            "Verifica que los nombres coincidan exactamente con: %s",
            filepath.name,
            sorted(_EMBALSES),
        )
        return pd.DataFrame(columns=["date", "avg_cota_msnm", "n_embalses"])

    # ── 5. Conversion de cota_msnm: coma europea → float ─────────────────────
    df["cota_msnm"] = (
        df["cota_msnm"]
        .str.strip()
        .str.replace(",", ".", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )

    # ── 6. Parseo de fecha ────────────────────────────────────────────────────
    df["fecha"] = pd.to_datetime(
        df["fecha"].str.strip(),
        format="%d/%m/%Y",
        errors="coerce",
    )

    # ── 7. Filtrar desde 2019 en adelante ─────────────────────────────────────
    antes = len(df)
    df = df[df["fecha"].dt.year >= _ANIO_INICIO].copy()
    logger.debug(
        "Filas desde %d en adelante: %d (descartadas: %d)",
        _ANIO_INICIO,
        len(df),
        antes - len(df),
    )

    if df.empty:
        logger.warning("No hay registros desde %d en el archivo.", _ANIO_INICIO)
        return pd.DataFrame(columns=["date", "avg_cota_msnm", "n_embalses"])

    # ── 8. Agregacion mensual: promedio de cota y conteo de embalses ──────────
    # Construir clave de mes para agrupar antes de crear 'date'
    df["date"] = df["fecha"].dt.to_period("M").dt.to_timestamp()

    agg = (
        df.groupby("date", sort=True)
        .agg(
            avg_cota_msnm=("cota_msnm", "mean"),
            n_embalses=("central", "nunique"),
        )
        .reset_index()
    )

    # ── 9. date como primer dia del mes (ya garantizado por to_timestamp) ─────
    agg["avg_cota_msnm"] = agg["avg_cota_msnm"].round(4)
    agg["n_embalses"] = agg["n_embalses"].astype(int)

    return agg[["date", "avg_cota_msnm", "n_embalses"]].reset_index(drop=True)


def load_to_db(df: pd.DataFrame, engine: Engine) -> int:
    """Insertar DataFrame en la tabla reservoir_levels usando upsert.

    Almacena avg_cota_msnm en el campo energy_gwh como proxy del nivel
    de los embalses, dado que este dataset no incluye energia directamente.
    Se usa ON CONFLICT (date) DO NOTHING para idempotencia.

    Args:
        df: DataFrame con columnas [date, avg_cota_msnm, n_embalses].
            Debe provenir de parse_reservoirs().
        engine: Engine de SQLAlchemy conectado a la base de datos cen_data.

    Returns:
        Numero de filas efectivamente insertadas.
    """
    if df.empty:
        logger.debug("DataFrame vacio, sin filas que insertar.")
        return 0

    records = [
        {"date": row["date"], "energy_gwh": row["avg_cota_msnm"]}
        for row in df.to_dict(orient="records")
    ]

    sql = text("""
        INSERT INTO reservoir_levels (date, energy_gwh)
        VALUES (:date, :energy_gwh)
        ON CONFLICT (date) DO NOTHING
    """)

    with engine.begin() as conn:
        result = conn.execute(sql, records)
        insertadas = result.rowcount

    logger.debug("Filas insertadas en reservoir_levels: %d", insertadas)
    return insertadas


def run_backfill(filepath: str = "data/raw/reservoirs_historical.csv") -> None:
    """Parsear el CSV historico de embalses y cargarlo en PostgreSQL.

    Args:
        filepath: Ruta al archivo CSV de niveles de embalses.
            Por defecto 'data/raw/reservoirs_historical.csv'.
    """
    try:
        logger.info("Parseando: %s", filepath)
        df = parse_reservoirs(filepath)

        if df.empty:
            logger.warning("El archivo no produjo registros. Abortando carga.")
            return

        logger.info("Registros mensuales parseados : %d", len(df))
        logger.info(
            "Rango de fechas               : %s → %s",
            df["date"].min().strftime("%Y-%m-%d"),
            df["date"].max().strftime("%Y-%m-%d"),
        )
        logger.info(
            "Embalses promedio por mes     : %.1f",
            df["n_embalses"].mean(),
        )

        engine = create_engine(DB_URL, pool_pre_ping=True)
        insertados = load_to_db(df, engine)
        engine.dispose()

        logger.info("Insertados en BD              : %d", insertados)

    except Exception as exc:
        logger.error("Error en backfill de embalses: %s", exc, exc_info=True)
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
