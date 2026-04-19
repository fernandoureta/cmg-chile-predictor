"""Carga de datos limpios en PostgreSQL.

Provee una funcion de upsert generica que reemplaza las funciones
load_to_db duplicadas en cada scraper. Cualquier modulo ETL puede
importar upsert_dataframe en lugar de reimplementar la logica de
INSERT ... ON CONFLICT DO NOTHING.

Uso tipico:
    from etl.load import upsert_dataframe

    insertadas = upsert_dataframe(
        df=df_limpio,
        table="marginal_costs",
        conflict_cols=["datetime", "barra"],
        engine=engine,
    )

Seguridad: los nombres de tabla y columna se interpolan directamente
en el SQL. Esta funcion es de uso interno exclusivo — los nombres
de tabla y columna provienen del codigo del proyecto, nunca de
entradas de usuario, por lo que no existe riesgo de SQL injection.
"""

import logging
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# =============================================================================
# FUNCIONES PUBLICAS
# =============================================================================

def log_insert_summary(
    table: str,
    intentadas: int,
    insertadas: int,
) -> None:
    """Loggear un resumen de insercion con formato consistente.

    Args:
        table: Nombre de la tabla PostgreSQL.
        intentadas: Numero de filas que se intentaron insertar.
        insertadas: Numero de filas efectivamente insertadas.
    """
    duplicados = intentadas - insertadas
    logger.info(
        "[%s] intentadas: %d | insertadas: %d | duplicados ignorados: %d",
        table,
        intentadas,
        insertadas,
        duplicados,
    )


def upsert_dataframe(
    df: pd.DataFrame,
    table: str,
    conflict_cols: Sequence[str],
    engine: Engine,
) -> int:
    """Insertar un DataFrame en una tabla PostgreSQL con upsert idempotente.

    Construye dinamicamente el SQL de INSERT a partir de los nombres de
    columnas del DataFrame y ejecuta ON CONFLICT (...) DO NOTHING para
    que ejecuciones repetidas no generen duplicados ni errores.

    Args:
        df: DataFrame con los datos a insertar. Los nombres de columna
            deben coincidir exactamente con los de la tabla destino.
        table: Nombre de la tabla PostgreSQL. Debe ser un identificador
            valido (uso interno, nunca proviene de entradas de usuario).
        conflict_cols: Lista de columnas que forman el UNIQUE constraint
            de la tabla. Se usan en la clausula ON CONFLICT.
        engine: Engine de SQLAlchemy con conexion activa a la BD.

    Returns:
        Numero de filas efectivamente insertadas (excluye conflictos).

    Raises:
        ValueError: Si conflict_cols esta vacio o contiene columnas que
            no existen en df.
    """
    if df.empty:
        logger.debug("[%s] DataFrame vacio, sin filas que insertar.", table)
        return 0

    # Validar que las columnas de conflicto existen en el DataFrame
    cols_faltantes = [c for c in conflict_cols if c not in df.columns]
    if not conflict_cols:
        raise ValueError(
            f"[{table}] conflict_cols no puede estar vacio."
        )
    if cols_faltantes:
        raise ValueError(
            f"[{table}] conflict_cols contiene columnas ausentes en el DataFrame: "
            f"{cols_faltantes}. Columnas disponibles: {list(df.columns)}"
        )

    # ── Construccion dinamica del SQL ─────────────────────────────────────────
    columnas = list(df.columns)
    cols_sql         = ", ".join(columnas)
    placeholders_sql = ", ".join(f":{c}" for c in columnas)
    conflict_sql     = ", ".join(conflict_cols)

    sql = text(f"""
        INSERT INTO {table} ({cols_sql})
        VALUES ({placeholders_sql})
        ON CONFLICT ({conflict_sql}) DO NOTHING
    """)

    records = df.to_dict(orient="records")
    intentadas = len(records)

    with engine.begin() as conn:
        result = conn.execute(sql, records)
        insertadas = result.rowcount

    log_insert_summary(table, intentadas, insertadas)
    return insertadas
