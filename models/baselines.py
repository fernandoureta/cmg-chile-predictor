"""Lineas base ingenuas para prediccion CMG (horizonte 1 hora).

Dos modelos de referencia sin entrenamiento:
  - Persistencia : pred(T) = CMG(T-1)
  - Estacional   : pred(T) = CMG(T-24)  (misma hora del dia anterior)

Cualquier modelo con horizonte de 1 hora debe superar a la persistencia
para aportar valor: en series horarias muy autocorrelacionadas como el
CMG, "la proxima hora sera igual a la actual" ya es un pronostico fuerte.

Las predicciones se calculan sobre exactamente las mismas filas de test
que XGBoost y LSTM: se reutiliza la feature matrix y split_data() de
models/xgboost_model.py, y se toman las columnas lag_cmg_usd_mwh_1h y
lag_cmg_usd_mwh_24h ya presentes en la matriz.
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import DB_URL, PREDICTIONS_VERSION                      # noqa: E402
from models.xgboost_model import load_feature_matrix, split_data    # noqa: E402

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────
BARRA: str = "Quillota 220kV"

# model_name en la tabla predictions -> columna de lag usada como prediccion
BASELINES: dict[str, str] = {
    "Persistencia": "lag_cmg_usd_mwh_1h",
    "Estacional_24h": "lag_cmg_usd_mwh_24h",
}


# =============================================================================
# PREDICCION Y METRICAS
# =============================================================================

def predict_baselines(X_test: pd.DataFrame) -> dict[str, pd.Series]:
    """Generar las predicciones de las lineas base sobre el test.

    Args:
        X_test: Features del test producidas por split_data(); debe incluir
            las columnas de lag listadas en BASELINES.

    Returns:
        Diccionario {model_name: Serie de predicciones indexada como X_test}.
    """
    return {
        name: X_test[col].astype(float).rename("cmg_predicted")
        for name, col in BASELINES.items()
    }


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Calcular MAE, RMSE, MAPE y R2 (MAPE excluye CMG real == 0).

    Args:
        y_true: Valores reales.
        y_pred: Predicciones alineadas con y_true.

    Returns:
        Diccionario con claves 'mae', 'rmse', 'mape', 'r2'.
    """
    yt = y_true.to_numpy(dtype=float)
    yp = y_pred.reindex(y_true.index).to_numpy(dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]

    nz = yt != 0
    return {
        "mae":  float(np.mean(np.abs(yt - yp))),
        "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))),
        "mape": float(np.mean(np.abs((yt[nz] - yp[nz]) / yt[nz])) * 100),
        "r2":   float(1 - np.sum((yt - yp) ** 2) / np.sum((yt - yt.mean()) ** 2)),
    }


# =============================================================================
# GUARDADO
# =============================================================================

def save_results(
    model_name: str,
    predictions: pd.Series,
    y_test: pd.Series,
    engine: Engine,
) -> None:
    """Guardar predicciones de una linea base en la tabla predictions.

    Inserta con model_version=PREDICTIONS_VERSION y horizon_h=1.
    Usa ON CONFLICT DO NOTHING para idempotencia.

    Args:
        model_name: Nombre del modelo (clave de BASELINES).
        predictions: Serie de predicciones indexada por datetime UTC.
        y_test: Target real del test (para actual_cmg).
        engine: Engine de SQLAlchemy conectado a cen_data.
    """
    records = [
        {
            "datetime":      ts,
            "barra":         BARRA,
            "model_name":    model_name,
            "model_version": PREDICTIONS_VERSION,
            "predicted_cmg": float(pred),
            "actual_cmg":    float(actual) if pd.notna(actual) else None,
            "horizon_h":     1,
        }
        for ts, pred, actual in zip(
            predictions.index,
            predictions.values,
            y_test.reindex(predictions.index).values,
        )
        if np.isfinite(pred)
    ]

    if not records:
        logger.warning("%s: no hay predicciones validas para guardar.", model_name)
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
        insertadas = conn.execute(sql, records).rowcount

    logger.info(
        "%s: %d insertadas / %d totales",
        model_name,
        insertadas,
        len(records),
    )


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
        df = load_feature_matrix(engine)
        _, _, _, _, X_test, y_test = split_data(df)

        for name, preds in predict_baselines(X_test).items():
            m = compute_metrics(y_test, preds)
            logger.info(
                "%-15s MAE=%.4f  RMSE=%.4f  MAPE=%.2f%%  R2=%.4f  (n=%d)",
                name, m["mae"], m["rmse"], m["mape"], m["r2"], len(preds),
            )
            save_results(name, preds, y_test, engine)

        logger.info(
            "Lineas base completadas en %.1fs",
            time.perf_counter() - t_total,
        )

    except KeyboardInterrupt:
        logger.warning("Interrumpido por el usuario.")
        sys.exit(0)
    finally:
        engine.dispose()
