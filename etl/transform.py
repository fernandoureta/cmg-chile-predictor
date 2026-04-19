"""Limpieza, imputacion y validacion de datos.

Recibe DataFrames crudos producidos por los scrapers y devuelve
DataFrames limpios listos para cargarse en PostgreSQL.

Pipeline de limpieza para CMG:
  valores negativos → NaN
  outliers extremos (>Q99.9×3) → NaN
  gaps <=3h → interpolacion lineal
  gaps >3h → LightGBM (o mediana si hay pocos datos de entrenamiento)

Pipeline de limpieza para generacion:
  valores negativos → NaN
  solar nocturno → 0.0
  gaps <=3h → interpolacion lineal
  recalculo de gen_total_mw
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────
# Columnas de tecnologia (excluye gen_total_mw, que se recalcula)
_COLUMNAS_TECH: list[str] = [
    "gen_solar_mw",
    "gen_wind_mw",
    "gen_hydro_reservoir_mw",
    "gen_hydro_runofriver_mw",
    "gen_gas_mw",
    "gen_coal_mw",
    "gen_diesel_mw",
]

# Hora local (America/Santiago) a partir de la cual solar = 0
_HORA_NOCHE_INICIO: int = 21   # 21:00 en adelante
_HORA_NOCHE_FIN: int = 6       # hasta las 06:59

# Minimo de filas validas para entrenar LightGBM
_MIN_TRAIN_ROWS: int = 100

# Multiplicador sobre Q99.9 para considerar un valor outlier extremo
_OUTLIER_FACTOR: float = 3.0

# Maximo de horas consecutivas de NaN imputables con interpolacion lineal
_MAX_GAP_INTERPOLACION: int = 3


# =============================================================================
# HELPERS PRIVADOS
# =============================================================================

def _build_temporal_features(dt_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Construir features temporales desde un DatetimeIndex.

    Args:
        dt_index: Indice de tipo DatetimeIndex (puede ser tz-aware).

    Returns:
        DataFrame con columnas [hour_of_day, day_of_week, month].
    """
    return pd.DataFrame(
        {
            "hour_of_day": dt_index.hour,
            "day_of_week": dt_index.dayofweek,
            "month":       dt_index.month,
        },
        index=dt_index,
    )


def _impute_lgbm(series: pd.Series) -> pd.Series:
    """Imputar NaN con LightGBM usando features temporales y lags.

    Usa como features: hour_of_day, day_of_week, month, lag_24h, lag_168h.
    Si hay menos de _MIN_TRAIN_ROWS filas validas, usa la mediana.
    Para NaN que no pueden ser predichos (features con NaN), usa la mediana.

    Args:
        series: Serie con indice DatetimeIndex y valores float con NaN.

    Returns:
        Serie con los NaN imputados. El indice no cambia.
    """
    import lightgbm as lgb  # importacion diferida: solo se usa si hay gaps >3h

    nan_mask = series.isna()
    if not nan_mask.any():
        return series

    # ── Construir matriz de features ──────────────────────────────────────────
    feats = _build_temporal_features(series.index)
    feats["lag_24h"]  = series.shift(24)
    feats["lag_168h"] = series.shift(168)

    # Filas donde todas las features estan disponibles
    feats_completas = feats.notna().all(axis=1)

    train_mask = ~nan_mask & feats_completas
    pred_mask  =  nan_mask & feats_completas

    median_val: float = float(series.dropna().median())

    if train_mask.sum() < _MIN_TRAIN_ROWS:
        logger.warning(
            "Solo %d filas validas para entrenar LightGBM. "
            "Usando mediana (%.4f) para imputar.",
            train_mask.sum(),
            median_val,
        )
        result = series.copy()
        result[nan_mask] = median_val
        return result

    X_train = feats[train_mask].values
    y_train = series[train_mask].values
    X_pred  = feats[pred_mask].values

    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    result = series.copy()
    if len(X_pred) > 0:
        result[pred_mask] = model.predict(X_pred)
        logger.debug(
            "LightGBM imputó %d valores (predichos) en '%s'.",
            int(pred_mask.sum()),
            series.name or "serie",
        )

    # NaN restantes: features con NaN ellas mismas (inicio de la serie, lags)
    aun_nan = result.isna()
    if aun_nan.any():
        logger.debug(
            "Imputando %d NaN residuales con mediana (%.4f).",
            int(aun_nan.sum()),
            median_val,
        )
        result[aun_nan] = median_val

    return result


# =============================================================================
# FUNCIONES PUBLICAS
# =============================================================================

def clean_marginal_costs(df: pd.DataFrame) -> pd.DataFrame:
    """Limpiar e imputar el DataFrame de costos marginales.

    Pipeline:
      1. Establece datetime como indice, ordena cronologicamente.
      2. Reemplaza valores negativos con NaN.
      3. Detecta outliers extremos (>Q99.9 x 3) y los reemplaza con NaN.
      4. Imputa gaps cortos (<=3h consecutivos) con interpolacion lineal.
      5. Imputa gaps largos con LightGBM (o mediana si datos insuficientes).
      6. Marca con is_imputed=True las filas que eran NaN originalmente.
      7. Resetea el indice y retorna el DataFrame limpio.

    Args:
        df: DataFrame crudo con columnas [datetime, barra, cmg_usd_mwh].
            datetime puede ser columna o indice.

    Returns:
        DataFrame con columnas [datetime, barra, cmg_usd_mwh, is_imputed].
        Ninguna fila tendra NaN en cmg_usd_mwh tras la imputacion.
    """
    df = df.copy()

    # ── 1. Datetime como indice ordenado ──────────────────────────────────────
    if "datetime" in df.columns:
        df = df.set_index("datetime")
    df = df.sort_index()

    # ── 2. Valores negativos → NaN ────────────────────────────────────────────
    negativos = df["cmg_usd_mwh"] < 0
    if negativos.any():
        logger.debug("Valores negativos reemplazados: %d", int(negativos.sum()))
        df.loc[negativos, "cmg_usd_mwh"] = np.nan

    # ── 3. Outliers extremos → NaN ────────────────────────────────────────────
    q999 = df["cmg_usd_mwh"].quantile(0.999)
    umbral = q999 * _OUTLIER_FACTOR
    outliers = df["cmg_usd_mwh"] > umbral
    if outliers.any():
        logger.debug(
            "Outliers extremos (>%.2f) reemplazados: %d",
            umbral,
            int(outliers.sum()),
        )
        df.loc[outliers, "cmg_usd_mwh"] = np.nan

    # ── 4. Marcar NaN originales ANTES de imputar ─────────────────────────────
    era_nan = df["cmg_usd_mwh"].isna()
    total_nan_original = int(era_nan.sum())
    logger.info(
        "NaN a imputar (negativos + outliers + originales): %d (%.2f%%)",
        total_nan_original,
        100 * total_nan_original / len(df) if len(df) else 0,
    )

    # ── 5. Interpolacion lineal para gaps cortos (<=3h) ───────────────────────
    df["cmg_usd_mwh"] = df["cmg_usd_mwh"].interpolate(
        method="linear",
        limit=_MAX_GAP_INTERPOLACION,
        limit_direction="forward",
    )

    # ── 6. LightGBM para gaps largos residuales ───────────────────────────────
    aun_nan = df["cmg_usd_mwh"].isna()
    if aun_nan.any():
        logger.info(
            "Gaps largos (>%dh) a imputar con LightGBM: %d",
            _MAX_GAP_INTERPOLACION,
            int(aun_nan.sum()),
        )
        df["cmg_usd_mwh"] = _impute_lgbm(df["cmg_usd_mwh"])

    # ── 7. Columna is_imputed ─────────────────────────────────────────────────
    df["is_imputed"] = era_nan

    df = df.reset_index()
    return df[["datetime", "barra", "cmg_usd_mwh", "is_imputed"]]


def clean_generation(df: pd.DataFrame) -> pd.DataFrame:
    """Limpiar el DataFrame de generacion por tecnologia.

    Pipeline:
      1. Reemplaza valores negativos con NaN en columnas de MW.
      2. Fuerza gen_solar_mw = 0.0 en horas nocturnas (21:00-06:59 Santiago).
      3. Interpola gaps cortos (<=3h) en cada columna de tecnologia.
      4. Recalcula gen_total_mw como suma de las 6 tecnologias.

    Args:
        df: DataFrame con columnas [datetime, gen_solar_mw, gen_wind_mw,
            gen_hydro_reservoir_mw, gen_hydro_runofriver_mw, gen_gas_mw,
            gen_coal_mw, gen_diesel_mw, gen_total_mw].
            datetime puede ser columna o indice.

    Returns:
        DataFrame limpio con las mismas columnas. gen_total_mw recalculado.
    """
    df = df.copy()

    # Asegurar indice ordenado para interpolacion
    tiene_col_datetime = "datetime" in df.columns
    if tiene_col_datetime:
        df = df.set_index("datetime")
    df = df.sort_index()

    # ── 1. Valores negativos → NaN en columnas de tecnologia ─────────────────
    for col in _COLUMNAS_TECH:
        if col not in df.columns:
            continue
        negativos = df[col] < 0
        if negativos.any():
            logger.debug(
                "Valores negativos en '%s': %d reemplazados con NaN.",
                col,
                int(negativos.sum()),
            )
            df.loc[negativos, col] = np.nan

    # ── 2. Solar nocturno → 0.0 ──────────────────────────────────────────────
    if "gen_solar_mw" in df.columns:
        # Convertir indice UTC a hora local Santiago para la mascara
        if df.index.tz is None:
            raise ValueError(
                "El índice datetime de generation debe ser tz-aware (UTC). "
                "Verificar que proviene de cen_generation.py."
            )
        dt_santiago = df.index.tz_convert("America/Santiago")
        hora_local = dt_santiago.hour
        noche_mask = (hora_local >= _HORA_NOCHE_INICIO) | (hora_local < _HORA_NOCHE_FIN)
        n_solar_cero = int(noche_mask.sum())
        if n_solar_cero > 0:
            df.loc[noche_mask, "gen_solar_mw"] = 0.0
            logger.debug(
                "gen_solar_mw forzado a 0.0 en %d filas nocturnas.", n_solar_cero
            )

    # ── 3. Interpolacion lineal para gaps cortos ──────────────────────────────
    for col in _COLUMNAS_TECH:
        if col not in df.columns:
            continue
        nan_antes = int(df[col].isna().sum())
        if nan_antes > 0:
            df[col] = df[col].interpolate(
                method="linear",
                limit=_MAX_GAP_INTERPOLACION,
                limit_direction="forward",
            )
            nan_despues = int(df[col].isna().sum())
            if nan_antes != nan_despues:
                logger.debug(
                    "'%s': %d NaN → %d tras interpolacion.",
                    col,
                    nan_antes,
                    nan_despues,
                )

    # ── 4. Recalcular gen_total_mw ────────────────────────────────────────────
    cols_presentes = [c for c in _COLUMNAS_TECH if c in df.columns]
    df["gen_total_mw"] = df[cols_presentes].sum(axis=1, min_count=1)

    if tiene_col_datetime:
        df = df.reset_index()

    return df


def validate_dataframe(df: pd.DataFrame, name: str) -> bool:
    """Validar un DataFrame y loggear advertencias si hay problemas.

    Verificaciones:
      - Ninguna columna completamente vacia.
      - Sin duplicados en la columna o indice datetime.
      - Porcentaje de NaN por columna no supera el 5%.

    No lanza excepciones: solo loggea advertencias y retorna False
    si se encontro alguna condicion anormal.

    Args:
        df: DataFrame a validar. Puede tener datetime como columna o indice.
        name: Nombre descriptivo del DataFrame para los mensajes de log.

    Returns:
        True si paso todas las validaciones sin advertencias.
        False si se encontro al menos una condicion anormal.
    """
    ok = True

    # ── Columnas completamente vacias ─────────────────────────────────────────
    vacias = [col for col in df.columns if df[col].isna().all()]
    if vacias:
        logger.warning(
            "[%s] Columnas completamente vacias: %s", name, vacias
        )
        ok = False

    # ── Duplicados en datetime ────────────────────────────────────────────────
    dt_col: Optional[pd.Series] = None
    if "datetime" in df.columns:
        dt_col = df["datetime"]
    elif isinstance(df.index, pd.DatetimeIndex):
        dt_col = df.index.to_series()

    if dt_col is not None:
        n_duplicados = int(dt_col.duplicated().sum())
        if n_duplicados > 0:
            logger.warning(
                "[%s] %d timestamps duplicados encontrados.", name, n_duplicados
            )
            ok = False
    else:
        logger.warning(
            "[%s] No se encontro columna o indice 'datetime' para verificar duplicados.",
            name,
        )

    # ── Porcentaje de NaN por columna ─────────────────────────────────────────
    n = len(df)
    if n > 0:
        for col in df.columns:
            pct_nan = 100 * df[col].isna().sum() / n
            if pct_nan > 5.0:
                logger.warning(
                    "[%s] Columna '%s' tiene %.1f%% de NaN.", name, col, pct_nan
                )
                ok = False

    if ok:
        logger.debug("[%s] Validacion OK (%d filas, %d columnas).", name, n, len(df.columns))

    return ok
