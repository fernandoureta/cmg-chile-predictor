"""Modelo SARIMA — linea base estadistica para prediccion CMG.

Entrena un modelo SARIMAX sobre los ultimos 6 meses del conjunto de
train para eficiencia computacional. Evalua sobre el conjunto de test
(2024) y guarda las predicciones en la tabla predictions de PostgreSQL.

Estacionalidad: diaria (s=24), captura el patron intradiario tipico
del CMG que cae al mediodia (generacion solar) y sube en punta vespertina.

Seleccion de orden: grid search sobre p,q in {0,1,2} con d=0 usando AIC.
El componente estacional se fija en (1,1,1,24) para todos los candidatos.
"""

import itertools
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import DB_URL  # noqa: E402

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────
BARRA: str = "Quillota 220kV"
TRAIN_END: str = "2023-06-30"
TEST_START: str = "2024-01-01"
SAMPLE_MONTHS: int = 6
SEASONAL_PERIOD: int = 24

_FALLBACK_ORDER: tuple = (1, 0, 1)
_FALLBACK_SEASONAL: tuple = (1, 1, 1, SEASONAL_PERIOD)


# =============================================================================
# CARGA DE DATOS
# =============================================================================

def _to_utc(series: pd.Series) -> pd.Series:
    """Normalizar una Serie de timestamps a UTC.

    psycopg2 puede devolver TIMESTAMPTZ con offsets mixtos segun el timezone
    del servidor PostgreSQL. pd.to_datetime(..., utc=True) normaliza todo.

    Args:
        series: Serie con timestamps (posiblemente timezone-aware con offsets
            variables como UTC-3/UTC-4 por DST de America/Santiago).

    Returns:
        Serie con dtype datetime64[ns, UTC].
    """
    return pd.to_datetime(series, utc=True)


def load_train_series(engine: Engine) -> pd.Series:
    """Cargar la serie de train para SARIMA desde PostgreSQL.

    Consulta marginal_costs para la barra Quillota, filtra el conjunto de
    train completo (hasta TRAIN_END) y retorna solo los ultimos SAMPLE_MONTHS
    meses para mantener el entrenamiento de SARIMA computacionalmente viable.

    Args:
        engine: Engine de SQLAlchemy conectado a cen_data.

    Returns:
        Serie con indice DatetimeIndex UTC, frecuencia horaria ('h'),
        nombre 'cmg_usd_mwh', y exactamente SAMPLE_MONTHS meses de datos
        del final del periodo de train.
    """
    sql = text("""
        SELECT datetime, cmg_usd_mwh
        FROM marginal_costs
        WHERE barra = :barra
          AND datetime <= :train_end
        ORDER BY datetime ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(
            sql,
            conn,
            params={
                "barra": BARRA,
                "train_end": TRAIN_END + " 23:59:59",
            },
        )

    df["datetime"] = _to_utc(df["datetime"])
    df = df.set_index("datetime").sort_index()

    # Tomar solo los ultimos SAMPLE_MONTHS meses del train
    cutoff = df.index.max() - pd.DateOffset(months=SAMPLE_MONTHS)
    serie = df.loc[cutoff:, "cmg_usd_mwh"]

    # Fijar frecuencia horaria para SARIMAX
    serie = serie.asfreq("h")

    logger.info(
        "Serie de train cargada: %d filas (%s → %s)",
        len(serie),
        serie.index.min(),
        serie.index.max(),
    )
    return serie


def load_test_series(engine: Engine) -> pd.Series:
    """Cargar la serie de test desde PostgreSQL.

    Consulta marginal_costs para la barra Quillota desde TEST_START
    en adelante (2024-01-01 → fin de datos).

    Args:
        engine: Engine de SQLAlchemy conectado a cen_data.

    Returns:
        Serie con indice DatetimeIndex UTC, frecuencia horaria ('h'),
        nombre 'cmg_usd_mwh'.
    """
    sql = text("""
        SELECT datetime, cmg_usd_mwh
        FROM marginal_costs
        WHERE barra = :barra
          AND datetime >= :test_start
        ORDER BY datetime ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(
            sql,
            conn,
            params={
                "barra": BARRA,
                "test_start": TEST_START,
            },
        )

    df["datetime"] = _to_utc(df["datetime"])
    serie = df.set_index("datetime").sort_index()["cmg_usd_mwh"]
    serie = serie.asfreq("h")

    logger.info(
        "Serie de test cargada: %d filas (%s → %s)",
        len(serie),
        serie.index.min(),
        serie.index.max(),
    )
    return serie


# =============================================================================
# SELECCION DE ORDEN
# =============================================================================

def select_sarima_order(series: pd.Series) -> tuple[tuple, tuple]:
    """Seleccionar el orden ARIMA optimo por grid search sobre AIC.

    Prueba todas las combinaciones de p in [0,1,2] y q in [0,1,2] con d=0.
    El componente estacional se fija en (1,1,1,SEASONAL_PERIOD) para todos.
    Selecciona la combinacion con menor AIC.

    El grid es pequeno (9 combinaciones) para mantener tiempos razonables
    con una serie de 6 meses y estacionalidad s=24.

    Args:
        series: Serie temporal con frecuencia horaria y sin NaN.

    Returns:
        Tupla (best_order, best_seasonal_order) donde:
        - best_order = (p, d, q)
        - best_seasonal_order = (P, D, Q, s)
        Si todas las combinaciones fallan, retorna el fallback
        ((1,0,1), (1,1,1,24)).
    """
    p_values = [0, 1, 2]
    q_values = [0, 1, 2]
    seasonal_order = (1, 1, 1, SEASONAL_PERIOD)

    best_aic = np.inf
    best_order: tuple = _FALLBACK_ORDER
    best_seasonal: tuple = _FALLBACK_SEASONAL
    exitos = 0

    logger.info(
        "Grid search SARIMA: %d combinaciones (p∈%s, d=0, q∈%s, P=1,D=1,Q=1,s=%d)",
        len(p_values) * len(q_values),
        p_values,
        q_values,
        SEASONAL_PERIOD,
    )

    for p, q in itertools.product(p_values, q_values):
        order = (p, 0, q)
        try:
            t0 = time.perf_counter()
            model = SARIMAX(
                series,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result = model.fit(disp=False)
            dt = time.perf_counter() - t0
            aic = result.aic
            exitos += 1
            logger.info(
                "  SARIMA%s x %s  AIC=%.2f  (%.1fs)%s",
                order,
                seasonal_order,
                aic,
                dt,
                "  ← MEJOR" if aic < best_aic else "",
            )
            if aic < best_aic:
                best_aic = aic
                best_order = order
                best_seasonal = seasonal_order

        except Exception as exc:
            logger.warning(
                "  SARIMA%s x %s  FALLO: %s",
                order,
                seasonal_order,
                exc,
            )

    if exitos == 0:
        logger.error(
            "Todas las combinaciones fallaron. Usando fallback %s x %s.",
            _FALLBACK_ORDER,
            _FALLBACK_SEASONAL,
        )
        return _FALLBACK_ORDER, _FALLBACK_SEASONAL

    logger.info(
        "Orden seleccionado: SARIMA%s x %s  AIC=%.2f",
        best_order,
        best_seasonal,
        best_aic,
    )
    return best_order, best_seasonal


# =============================================================================
# ENTRENAMIENTO
# =============================================================================

def train_sarima(
    series: pd.Series,
    order: tuple,
    seasonal_order: tuple,
) -> SARIMAXResults:
    """Entrenar el modelo SARIMAX con los parametros seleccionados.

    Args:
        series: Serie de train con frecuencia horaria y sin NaN.
        order: Orden ARIMA (p, d, q).
        seasonal_order: Orden estacional (P, D, Q, s).

    Returns:
        Objeto SARIMAXResults con el modelo ajustado.
    """
    logger.info(
        "Entrenando SARIMA%s x %s sobre %d observaciones...",
        order,
        seasonal_order,
        len(series),
    )
    t0 = time.perf_counter()

    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False)

    dt = time.perf_counter() - t0
    logger.info(
        "Entrenamiento completado en %.1fs | AIC=%.2f | BIC=%.2f",
        dt,
        result.aic,
        result.bic,
    )
    logger.info("Parametros ajustados:\n%s", result.summary().tables[1])
    return result


# =============================================================================
# EVALUACION
# =============================================================================

def evaluate_sarima(
    model_result: SARIMAXResults,
    test_series: pd.Series,
) -> dict:
    """Generar predicciones y calcular metricas sobre el conjunto de test.

    Usa forecast() con steps=len(test_series) para prediccion out-of-sample.
    Las metricas calculadas son: MAE, RMSE, MAPE, R2.
    MAPE ignora los timesteps donde el valor real es cero para evitar
    division por cero (ocurre al mediodia cuando el CMG cae a 0 por solar).

    Args:
        model_result: Resultado del entrenamiento SARIMAX.
        test_series: Serie real del conjunto de test.

    Returns:
        Diccionario con claves:
        - 'mae': float
        - 'rmse': float
        - 'mape': float (porcentaje, 0-100)
        - 'r2': float
        - 'predictions': pd.Series con las predicciones alineadas al indice
          de test_series
    """
    logger.info("Generando forecast para %d pasos...", len(test_series))
    t0 = time.perf_counter()

    forecast = model_result.forecast(steps=len(test_series))
    forecast.index = test_series.index
    forecast.name = "cmg_predicted"

    dt = time.perf_counter() - t0
    logger.info("Forecast generado en %.1fs", dt)

    y_true = test_series.values
    y_pred = forecast.values

    # Alinear sobre indices donde ambos son finitos
    mask_validos = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[mask_validos]
    yp = y_pred[mask_validos]

    mae  = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    r2   = float(1 - np.sum((yt - yp) ** 2) / np.sum((yt - np.mean(yt)) ** 2))

    # MAPE: excluir ceros en el target para evitar division por cero
    mask_no_cero = mask_validos & (y_true != 0)
    yt_nz = y_true[mask_no_cero]
    yp_nz = y_pred[mask_no_cero]
    mape = float(np.mean(np.abs((yt_nz - yp_nz) / yt_nz)) * 100)

    metrics = {
        "mae":  mae,
        "rmse": rmse,
        "mape": mape,
        "r2":   r2,
        "predictions": forecast,
    }

    logger.info("Metricas sobre test:")
    logger.info("  MAE   = %.4f USD/MWh", mae)
    logger.info("  RMSE  = %.4f USD/MWh", rmse)
    logger.info("  MAPE  = %.2f%%", mape)
    logger.info("  R2    = %.4f", r2)
    return metrics


# =============================================================================
# GUARDADO DE RESULTADOS
# =============================================================================

def save_results(
    metrics: dict,
    test_series: pd.Series,
    engine: Engine,
) -> None:
    """Guardar predicciones en la tabla predictions de PostgreSQL.

    Inserta cada prediccion con model_name='SARIMA', horizon_h=1 (prediccion
    de un paso hacia adelante en modo forecast out-of-sample), usando
    ON CONFLICT DO NOTHING para idempotencia.

    Args:
        metrics: Diccionario producido por evaluate_sarima(), que incluye
            la clave 'predictions' con la Serie de predicciones.
        test_series: Serie real del conjunto de test (para actual_cmg).
        engine: Engine de SQLAlchemy conectado a cen_data.
    """
    predictions: pd.Series = metrics["predictions"]

    records = [
        {
            "datetime":      ts,
            "barra":         BARRA,
            "model_name":    "SARIMA",
            "model_version": "1.0",
            "predicted_cmg": float(pred),
            "actual_cmg":    float(actual) if pd.notna(actual) else None,
            "horizon_h":     1,
        }
        for ts, pred, actual in zip(
            predictions.index,
            predictions.values,
            test_series.reindex(predictions.index).values,
        )
        if np.isfinite(pred)
    ]

    if not records:
        logger.warning("No hay predicciones validas para guardar.")
        return

    sql = text("""
        INSERT INTO predictions (
            datetime, barra, model_name, model_version,
            predicted_cmg, actual_cmg, horizon_h
        )
        VALUES (
            :datetime, :barra, :model_name, :model_version,
            :predicted_cmg, :actual_cmg, :horizon_h
        )
        ON CONFLICT (datetime, barra, model_name, model_version, horizon_h)
        DO NOTHING
    """)

    with engine.begin() as conn:
        result = conn.execute(sql, records)
        insertadas = result.rowcount

    logger.info(
        "Predicciones guardadas en BD: %d insertadas / %d totales",
        insertadas,
        len(records),
    )

    # Loggear metricas en formato tabla
    logger.info("=" * 50)
    logger.info("RESUMEN METRICAS SARIMA")
    logger.info("=" * 50)
    logger.info("  %-10s %12.4f %s", "MAE",  metrics["mae"],  "USD/MWh")
    logger.info("  %-10s %12.4f %s", "RMSE", metrics["rmse"], "USD/MWh")
    logger.info("  %-10s %11.2f%%",  "MAPE", metrics["mape"])
    logger.info("  %-10s %12.4f",    "R2",   metrics["r2"])
    logger.info("=" * 50)


# =============================================================================
# EJECUCION DIRECTA
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    engine = create_engine(DB_URL, pool_pre_ping=True)
    t_total = time.perf_counter()

    try:
        # ── Carga de datos ─────────────────────────────────────────────────────
        logger.info("Cargando series desde PostgreSQL...")
        train_series = load_train_series(engine)
        test_series  = load_test_series(engine)

        logger.info(
            "Train: %d obs  |  Test: %d obs",
            len(train_series),
            len(test_series),
        )

        # ── Imputar NaN residuales en train antes de modelar ──────────────────
        nan_train = train_series.isna().sum()
        if nan_train:
            logger.warning(
                "%d NaN en serie de train. Imputando con interpolacion lineal.",
                nan_train,
            )
            train_series = train_series.interpolate(method="linear").ffill().bfill()

        # ── Seleccion de orden ─────────────────────────────────────────────────
        logger.info("Seleccionando orden optimo SARIMA...")
        t0 = time.perf_counter()
        order, seasonal_order = select_sarima_order(train_series)
        logger.info(
            "Seleccion completada en %.1fs", time.perf_counter() - t0
        )

        # ── Entrenamiento ──────────────────────────────────────────────────────
        model_result = train_sarima(train_series, order, seasonal_order)

        # ── Evaluacion ────────────────────────────────────────────────────────
        metrics = evaluate_sarima(model_result, test_series)

        # ── Guardado ──────────────────────────────────────────────────────────
        save_results(metrics, test_series, engine)

        # ── Resumen final ──────────────────────────────────────────────────────
        logger.info(
            "Pipeline SARIMA completado en %.1fs",
            time.perf_counter() - t_total,
        )

    except KeyboardInterrupt:
        logger.warning("Interrumpido por el usuario.")
        sys.exit(0)
    finally:
        engine.dispose()
