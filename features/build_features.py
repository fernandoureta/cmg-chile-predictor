"""Feature engineering para el modelo de prediccion CMG.

Construye el DataFrame de features listo para entrenar, validar y evaluar
los modelos de prediccion del costo marginal (cmg_usd_mwh) de la barra
Quillota 220kV.

Features generadas:
  - Target y lags del target: 1h, 2h, 3h, 6h, 12h, 24h, 168h
  - Rolling stats sobre cmg: media y std con ventanas 24h y 168h
  - Encoding ciclico: hora del dia, dia de semana, mes del anio
  - Variables de calendario: is_weekend, is_holiday (feriados Chile)
  - Temperatura diaria maxima: temp_max_c desde tabla weather
  - Generacion por tecnologia: 7 columnas + gen_total_mw
  - Nivel de embalses: energy_gwh (propagado forward-fill a resolucion horaria)

Split cronologico (definido en CLAUDE.md):
  Train      : 2021-01-01 → 2023-06-30
  Validacion : 2023-07-01 → 2023-12-31
  Test       : 2024-01-01 → 2024-12-31

Nota sobre cobertura: CMG tiene datos desde 2019, pero generacion solo
desde 2021. El DataFrame final solo incluye filas donde TODOS los joins
tienen datos (inner join efectivo desde 2021).
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import holidays
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import DB_URL  # noqa: E402

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────
BARRA: str = "Quillota 220kV"
TIMEZONE_LOCAL: str = "America/Santiago"

# Lags del target en horas
_LAGS_H: list[int] = [1, 2, 3, 6, 12, 24, 168]

# Ventanas para rolling stats en horas
_ROLLING_WINDOWS_H: list[int] = [24, 168]

# Split cronologico (alineado con disponibilidad de generacion desde 2021)
TRAIN_START: str = "2021-01-01"
TRAIN_END: str   = "2023-06-30"
VAL_START: str   = "2023-07-01"
VAL_END: str     = "2023-12-31"
TEST_START: str  = "2024-01-01"
TEST_END: str    = "2024-12-31"

# Columnas de generacion a incluir en el feature set
_COLS_GEN: list[str] = [
    "gen_solar_mw",
    "gen_wind_mw",
    "gen_hydro_reservoir_mw",
    "gen_hydro_runofriver_mw",
    "gen_gas_mw",
    "gen_coal_mw",
    "gen_diesel_mw",
    "gen_total_mw",
]


# =============================================================================
# HELPERS PRIVADOS — CARGA DE DATOS
# =============================================================================

def _to_utc(df: pd.DataFrame, col: str = "datetime") -> pd.DataFrame:
    """Normalizar una columna datetime a UTC puro.

    psycopg2 puede devolver TIMESTAMPTZ con offsets variables (UTC-3/UTC-4)
    segun la zona horaria del servidor PostgreSQL. Esta funcion convierte
    cualquier representacion timezone-aware a UTC, y localiza a UTC si la
    columna es naive.

    Args:
        df: DataFrame con la columna a normalizar.
        col: Nombre de la columna datetime.

    Returns:
        El mismo DataFrame con la columna normalizada a UTC.
    """
    serie = pd.to_datetime(df[col], utc=True)
    df[col] = serie
    return df


def _load_marginal_costs(engine: Engine) -> pd.DataFrame:
    """Cargar costos marginales de la BD para la barra Quillota.

    Args:
        engine: Engine de SQLAlchemy conectado a cen_data.

    Returns:
        DataFrame con columnas [cmg_usd_mwh, is_imputed], indexado por
        datetime (DatetimeIndex UTC).
    """
    sql = text("""
        SELECT datetime, cmg_usd_mwh, is_imputed
        FROM marginal_costs
        WHERE barra = :barra
        ORDER BY datetime ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"barra": BARRA})

    df = _to_utc(df)
    df = df.set_index("datetime").sort_index()
    logger.info("marginal_costs cargados: %d filas", len(df))
    return df


def _load_generation(engine: Engine) -> pd.DataFrame:
    """Cargar generacion por tecnologia de la BD.

    Args:
        engine: Engine de SQLAlchemy conectado a cen_data.

    Returns:
        DataFrame con columnas de generacion, indexado por datetime (UTC).
    """
    cols = ", ".join(_COLS_GEN)
    sql = text(f"SELECT datetime, {cols} FROM generation_by_tech ORDER BY datetime ASC")
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)

    df = _to_utc(df)
    df = df.set_index("datetime").sort_index()
    logger.info("generation_by_tech cargados: %d filas", len(df))
    return df


def _load_reservoirs(engine: Engine) -> pd.DataFrame:
    """Cargar niveles de embalses de la BD (resolucion mensual).

    Los datos mensuales se propagan con forward-fill al unirse con el
    DataFrame horario en build_feature_matrix().

    Args:
        engine: Engine de SQLAlchemy conectado a cen_data.

    Returns:
        DataFrame con columna [energy_gwh], indexado por date (DATE).
    """
    sql = text("SELECT date, energy_gwh FROM reservoir_levels ORDER BY date ASC")
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, parse_dates=["date"])

    df = df.set_index("date")
    logger.info("reservoir_levels cargados: %d filas (mensuales)", len(df))
    return df


def _load_weather(engine: Engine) -> pd.DataFrame:
    """Cargar temperatura diaria de Santiago de la BD.

    Args:
        engine: Engine de SQLAlchemy conectado a cen_data.

    Returns:
        DataFrame con columna [temp_max_c], indexado por date (DATE).
    """
    sql = text("""
        SELECT date, temp_max_c
        FROM weather
        WHERE region = 'Santiago'
        ORDER BY date ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, parse_dates=["date"])

    df = df.set_index("date")
    logger.info("weather cargados: %d filas (diarios)", len(df))
    return df


# =============================================================================
# HELPERS PRIVADOS — TRANSFORMACIONES
# =============================================================================

def _add_lag_features(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Agregar columnas de lags del target.

    Args:
        df: DataFrame con indice DatetimeIndex horario.
        col: Nombre de la columna sobre la que calcular los lags.

    Returns:
        El mismo DataFrame con columnas lag_{col}_{h}h agregadas.
    """
    for h in _LAGS_H:
        df[f"lag_{col}_{h}h"] = df[col].shift(h)
    return df


def _add_rolling_features(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Agregar media y desviacion estandar con ventanas rolling.

    Usa shift(1) antes del rolling para que la estadistica de la hora T
    no incluya T, evitando data leakage. min_periods=1 evita NaN en
    ventanas parciales al inicio de la serie.

    Args:
        df: DataFrame con indice DatetimeIndex horario.
        col: Nombre de la columna sobre la que calcular las stats.

    Returns:
        El mismo DataFrame con columnas rolling_mean_{w}h y rolling_std_{w}h.
    """
    for w in _ROLLING_WINDOWS_H:
        shifted = df[col].shift(1)
        df[f"rolling_mean_{col}_{w}h"] = (
            shifted.rolling(window=w, min_periods=1).mean()
        )
        df[f"rolling_std_{col}_{w}h"] = (
            shifted.rolling(window=w, min_periods=1).std()
        )
    return df


def _add_cyclic_encoding(df: pd.DataFrame, dt_santiago: pd.DatetimeIndex) -> pd.DataFrame:
    """Agregar encoding ciclico de hora, dia de semana y mes.

    Transforma variables periodicas a (sin, cos) para que el modelo
    perciba la continuidad del ciclo (e.g., hora 23 es adyacente a hora 0).

    Args:
        df: DataFrame al que agregar las columnas.
        dt_santiago: DatetimeIndex convertido a zona horaria America/Santiago.

    Returns:
        El mismo DataFrame con 6 columnas de encoding ciclico.
    """
    hora = dt_santiago.hour
    dow  = dt_santiago.dayofweek   # 0=lunes, 6=domingo
    mes  = dt_santiago.month

    df["hora_sin"] = np.sin(2 * np.pi * hora      / 24)
    df["hora_cos"] = np.cos(2 * np.pi * hora      / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * dow       / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * dow       / 7)
    df["mes_sin"]  = np.sin(2 * np.pi * (mes - 1) / 12)
    df["mes_cos"]  = np.cos(2 * np.pi * (mes - 1) / 12)

    return df


def _add_calendar_features(
    df: pd.DataFrame,
    dt_santiago: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Agregar indicadores de fin de semana y feriado en Chile.

    Usa la libreria holidays con pais CL para obtener los feriados
    nacionales. Los feriados regionales no se incluyen.

    Args:
        df: DataFrame al que agregar las columnas.
        dt_santiago: DatetimeIndex convertido a zona horaria America/Santiago.

    Returns:
        El mismo DataFrame con columnas is_weekend (bool) e is_holiday (bool).
    """
    years = sorted(set(dt_santiago.year))
    cl_holidays = holidays.Chile(years=years)

    fechas_date = dt_santiago.date   # numpy array de datetime.date
    df["is_weekend"] = (dt_santiago.dayofweek >= 5)
    df["is_holiday"] = [d in cl_holidays for d in fechas_date]

    return df


def _merge_daily_to_hourly(
    df_hourly: pd.DataFrame,
    df_daily: pd.DataFrame,
    col: str,
    fill_method: str = "ffill",
) -> pd.DataFrame:
    """Unir una columna de resolucion diaria/mensual a un DataFrame horario.

    Hace reindex del DataFrame diario al indice horario del DataFrame
    principal y propaga con forward-fill (o el metodo indicado).
    Si tras el ffill quedan NaN al inicio (datos diarios posteriores al
    inicio del horario), aplica bfill como fallback.

    Args:
        df_hourly: DataFrame con indice DatetimeIndex UTC (resolucion horaria).
        df_daily: DataFrame con indice de tipo DatetimeIndex o DateIndex.
        col: Nombre de la columna a unir desde df_daily.
        fill_method: Metodo de relleno tras el reindex ('ffill' por defecto).

    Returns:
        df_hourly con la columna col agregada.
    """
    if not isinstance(df_daily.index, pd.DatetimeIndex):
        idx_utc = pd.to_datetime(df_daily.index).tz_localize("UTC")
    elif df_daily.index.tz is None:
        idx_utc = df_daily.index.tz_localize("UTC")
    else:
        idx_utc = df_daily.index.tz_convert("UTC")

    serie = pd.Series(df_daily[col].values, index=idx_utc, name=col)
    serie_horaria = serie.reindex(df_hourly.index, method=fill_method)

    if serie_horaria.isna().any():
        serie_horaria = serie_horaria.bfill()

    df_hourly[col] = serie_horaria.values
    return df_hourly


# =============================================================================
# FUNCION PUBLICA PRINCIPAL
# =============================================================================

def build_feature_matrix(
    engine: Engine,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Construir el DataFrame de features completo desde la base de datos.

    Pipeline:
      1. Carga CMG, generacion, embalses y clima desde PostgreSQL.
      2. Inner join CMG x generacion (solo filas con datos en ambas tablas).
      3. Left join embalses y clima con forward-fill a resolucion horaria.
      4. Filtra por el rango de fechas solicitado.
      5. Agrega lags del target (1h, 2h, 3h, 6h, 12h, 24h, 168h).
      6. Agrega rolling stats (media y std, ventanas 24h y 168h).
      7. Agrega encoding ciclico (hora, dia de semana, mes).
      8. Agrega variables de calendario (is_weekend, is_holiday).
      9. Elimina filas donde el target es NaN.

    Args:
        engine: Engine de SQLAlchemy conectado a cen_data.
        start: Fecha de inicio opcional, formato 'YYYY-MM-DD'.
            Si se omite, se usa TRAIN_START (2021-01-01).
        end: Fecha de fin opcional, formato 'YYYY-MM-DD'.
            Si se omite, se usa TEST_END (2024-12-31).

    Returns:
        DataFrame con indice DatetimeIndex UTC y todas las features.
        La columna target es 'cmg_usd_mwh'. Los NaN en features se
        conservan para que los modelos los manejen segun su naturaleza.

    Raises:
        ValueError: Si el DataFrame resultante esta vacio tras el join.
    """
    # ── 1. Carga de tablas ────────────────────────────────────────────────────
    df_cmg  = _load_marginal_costs(engine)
    df_gen  = _load_generation(engine)
    df_res  = _load_reservoirs(engine)
    df_wthr = _load_weather(engine)

    # ── 2. Inner join CMG x generacion ───────────────────────────────────────
    # Se usa merge explicito (no join) para evitar el producto cartesiano que
    # ocurre cuando pandas detecta indices no-unicos por mezcla de offsets DST.
    df = pd.merge(
        df_cmg.reset_index(),
        df_gen.reset_index(),
        on="datetime",
        how="inner",
    ).set_index("datetime").sort_index()

    logger.info(
        "Tras inner join CMG x generacion: %d filas (%s → %s)",
        len(df),
        df.index.min(),
        df.index.max(),
    )

    if df.empty:
        raise ValueError(
            "El join CMG x generacion produjo un DataFrame vacio. "
            "Verifica que ambas tablas tienen datos en el rango solicitado."
        )

    # ── 3. Left join embalses y clima (daily → forward-fill horario) ──────────
    df = _merge_daily_to_hourly(df, df_res,  col="energy_gwh", fill_method="ffill")
    df = _merge_daily_to_hourly(df, df_wthr, col="temp_max_c", fill_method="ffill")

    # ── 4. Filtro de rango de fechas ──────────────────────────────────────────
    _start = pd.Timestamp(start or TRAIN_START, tz="UTC")
    _end   = pd.Timestamp(end   or TEST_END,    tz="UTC") + pd.Timedelta(days=1)
    df = df.loc[_start:_end]
    logger.info("Tras filtro de fechas: %d filas", len(df))

    # ── 5-8. Features temporales y de calendario ──────────────────────────────
    dt_santiago = df.index.tz_convert(TIMEZONE_LOCAL)

    df = _add_lag_features(df, col="cmg_usd_mwh")
    df = _add_rolling_features(df, col="cmg_usd_mwh")
    df = _add_cyclic_encoding(df, dt_santiago)
    df = _add_calendar_features(df, dt_santiago)

    # ── 9. Eliminar filas sin target ──────────────────────────────────────────
    n_antes = len(df)
    df = df.dropna(subset=["cmg_usd_mwh"])
    n_eliminadas = n_antes - len(df)
    if n_eliminadas:
        logger.warning("Eliminadas %d filas con NaN en el target.", n_eliminadas)

    logger.info(
        "Feature matrix construida: %d filas x %d columnas",
        len(df),
        len(df.columns),
    )
    return df


def get_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Dividir el DataFrame en train, validacion y test segun split cronologico.

    Los limites estan definidos en CLAUDE.md y en las constantes de este
    modulo. No se aplica shuffle ni estratificacion.

    Args:
        df: DataFrame con indice DatetimeIndex UTC, producido por
            build_feature_matrix().

    Returns:
        Tupla (df_train, df_val, df_test).
    """
    train = df.loc[TRAIN_START:TRAIN_END]
    val   = df.loc[VAL_START:VAL_END]
    test  = df.loc[TEST_START:TEST_END]

    logger.info(
        "Split | train: %d  val: %d  test: %d",
        len(train), len(val), len(test),
    )
    return train, val, test


# =============================================================================
# EJECUCION DIRECTA — PRUEBA RAPIDA
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    engine = create_engine(DB_URL, pool_pre_ping=True)

    logger.info("Construyendo feature matrix completa...")
    df_features = build_feature_matrix(engine)
    engine.dispose()

    logger.info("=" * 60)
    logger.info("RESULTADO")
    logger.info("=" * 60)
    logger.info("Shape          : %d filas x %d columnas", *df_features.shape)
    logger.info(
        "Rango          : %s → %s",
        df_features.index.min(),
        df_features.index.max(),
    )

    nan_summary = df_features.isna().sum()
    nan_cols = nan_summary[nan_summary > 0]
    if nan_cols.empty:
        logger.info("NaN por columna : (ninguno)")
    else:
        logger.info("NaN por columna :")
        for col, n in nan_cols.items():
            logger.info("  %-45s %d", col, n)

    logger.info("=" * 60)
    logger.info("COLUMNAS GENERADAS (%d):", len(df_features.columns))
    for col in df_features.columns:
        logger.info("  %s", col)

    logger.info("=" * 60)
    logger.info("PRIMERAS 3 FILAS:")
    logger.info("\n%s", df_features.head(3).to_string())
