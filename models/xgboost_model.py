"""Modelo XGBoost con feature engineering completo para prediccion CMG.

Usa la feature matrix de 29 columnas construida por build_features.py:
lags del target, rolling stats, encoding ciclico, calendario, generacion
por tecnologia, nivel de embalses y temperatura.

Early stopping sobre el conjunto de validacion (2023-07-01 a 2023-12-31)
con eval_metric=mae y patience=50 estimadores.
"""

import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend no interactivo para entornos sin pantalla
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import DB_URL                               # noqa: E402
from features.build_features import build_feature_matrix  # noqa: E402

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────
BARRA: str = "Quillota 220kV"
TRAIN_END: str  = "2023-06-30"
VAL_START: str  = "2023-07-01"
VAL_END: str    = "2023-12-31"
TEST_START: str = "2024-01-01"
TARGET: str = "cmg_usd_mwh"

_SAVED_DIR = _ROOT / "models" / "saved"

FEATURE_COLS: list[str] = [
    "gen_solar_mw",
    "gen_wind_mw",
    "gen_hydro_reservoir_mw",
    "gen_hydro_runofriver_mw",
    "gen_gas_mw",
    "gen_coal_mw",
    "gen_diesel_mw",
    "gen_total_mw",
    "energy_gwh",
    "temp_max_c",
    "lag_cmg_usd_mwh_1h",
    "lag_cmg_usd_mwh_2h",
    "lag_cmg_usd_mwh_3h",
    "lag_cmg_usd_mwh_6h",
    "lag_cmg_usd_mwh_12h",
    "lag_cmg_usd_mwh_24h",
    "lag_cmg_usd_mwh_168h",
    "rolling_mean_cmg_usd_mwh_24h",
    "rolling_std_cmg_usd_mwh_24h",
    "rolling_mean_cmg_usd_mwh_168h",
    "rolling_std_cmg_usd_mwh_168h",
    "hora_sin",
    "hora_cos",
    "dow_sin",
    "dow_cos",
    "mes_sin",
    "mes_cos",
    "is_weekend",
    "is_holiday",
]


# =============================================================================
# CARGA Y SPLIT
# =============================================================================

def load_feature_matrix(engine: Engine) -> pd.DataFrame:
    """Construir la feature matrix completa desde PostgreSQL.

    Delega a build_feature_matrix() de features/build_features.py, que
    une marginal_costs, generation_by_tech, reservoir_levels y weather,
    y agrega todos los features de ingenieria.

    Args:
        engine: Engine de SQLAlchemy conectado a cen_data.

    Returns:
        DataFrame con indice DatetimeIndex UTC y 31 columnas de features.
    """
    logger.info("Construyendo feature matrix desde PostgreSQL...")
    t0 = time.perf_counter()
    df = build_feature_matrix(engine)
    logger.info(
        "Feature matrix lista: %d filas x %d columnas (%.1fs)",
        len(df),
        len(df.columns),
        time.perf_counter() - t0,
    )
    return df


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Dividir la feature matrix en train, validacion y test.

    Aplica el split cronologico definido en CLAUDE.md. Elimina filas con
    NaN en cualquier feature o en el target antes de separar X e y.

    Args:
        df: DataFrame con indice DatetimeIndex UTC producido por
            load_feature_matrix().

    Returns:
        Tupla (X_train, y_train, X_val, y_val, X_test, y_test) donde:
        - X_*: DataFrame con FEATURE_COLS
        - y_*: Serie con TARGET (cmg_usd_mwh)
    """
    cols_necesarias = FEATURE_COLS + [TARGET]
    df_clean = df[cols_necesarias].dropna()

    train = df_clean.loc[:TRAIN_END]
    val   = df_clean.loc[VAL_START:VAL_END]
    test  = df_clean.loc[TEST_START:]

    X_train, y_train = train[FEATURE_COLS], train[TARGET]
    X_val,   y_val   = val[FEATURE_COLS],   val[TARGET]
    X_test,  y_test  = test[FEATURE_COLS],  test[TARGET]

    logger.info(
        "Split | train: %d  val: %d  test: %d",
        len(X_train), len(X_val), len(X_test),
    )
    logger.info(
        "Train rango: %s → %s", X_train.index.min(), X_train.index.max()
    )
    logger.info(
        "Val   rango: %s → %s", X_val.index.min(),   X_val.index.max()
    )
    logger.info(
        "Test  rango: %s → %s", X_test.index.min(),  X_test.index.max()
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


# =============================================================================
# ENTRENAMIENTO
# =============================================================================

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> xgb.XGBRegressor:
    """Entrenar el modelo XGBoost con early stopping sobre validacion.

    Hiperparametros fijos segun CLAUDE.md:
      n_estimators=1000, learning_rate=0.05, max_depth=6,
      subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=50.

    El early stopping detiene el entrenamiento cuando el MAE de validacion
    no mejora durante 50 estimadores consecutivos.

    Args:
        X_train: Features del conjunto de train.
        y_train: Target del conjunto de train.
        X_val: Features del conjunto de validacion.
        y_val: Target del conjunto de validacion.

    Returns:
        Modelo XGBRegressor ajustado con el mejor numero de estimadores.
    """
    logger.info(
        "Entrenando XGBoost: n_estimators=1000, lr=0.05, max_depth=6, "
        "early_stopping=50 | train=%d  val=%d",
        len(X_train),
        len(X_val),
    )
    t0 = time.perf_counter()

    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=50,
        eval_metric="mae",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    dt = time.perf_counter() - t0
    best_iter = model.best_iteration
    best_score = model.best_score

    logger.info(
        "Entrenamiento completado en %.1fs | best_iteration=%d | val_mae=%.4f",
        dt,
        best_iter,
        best_score,
    )
    return model


# =============================================================================
# EVALUACION
# =============================================================================

def evaluate_model(
    model: xgb.XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Generar predicciones y calcular metricas sobre el conjunto de test.

    Calcula MAE, RMSE, MAPE y R2. MAPE excluye timesteps donde el valor
    real es cero para evitar division por cero (mediodia solar).

    Args:
        model: Modelo XGBRegressor entrenado.
        X_test: Features del conjunto de test.
        y_test: Target real del conjunto de test.

    Returns:
        Diccionario con claves 'mae', 'rmse', 'mape', 'r2' y 'predictions'
        (pd.Series con las predicciones alineadas al indice de y_test).
    """
    y_pred_arr = model.predict(X_test)
    predictions = pd.Series(y_pred_arr, index=y_test.index, name="cmg_predicted")

    y_true = y_test.values
    y_pred = y_pred_arr

    mask_validos = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[mask_validos]
    yp = y_pred[mask_validos]

    mae  = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    r2   = float(1 - np.sum((yt - yp) ** 2) / np.sum((yt - np.mean(yt)) ** 2))

    mask_no_cero = mask_validos & (y_true != 0)
    yt_nz = y_true[mask_no_cero]
    yp_nz = y_pred[mask_no_cero]
    mape  = float(np.mean(np.abs((yt_nz - yp_nz) / yt_nz)) * 100)

    metrics = {
        "mae":         mae,
        "rmse":        rmse,
        "mape":        mape,
        "r2":          r2,
        "predictions": predictions,
    }

    logger.info("Metricas sobre test (XGBoost):")
    logger.info("  MAE   = %.4f USD/MWh", mae)
    logger.info("  RMSE  = %.4f USD/MWh", rmse)
    logger.info("  MAPE  = %.2f%%", mape)
    logger.info("  R2    = %.4f", r2)
    return metrics


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

def plot_feature_importance(
    model: xgb.XGBRegressor,
    feature_names: list[str],
) -> None:
    """Loggear y guardar grafico de importancia de features.

    Extrae la importancia por ganancia (gain) del modelo, loggea el
    top-15 y guarda el grafico en models/saved/xgboost_feature_importance.png.

    Args:
        model: Modelo XGBRegressor entrenado.
        feature_names: Lista de nombres de features en el mismo orden que
            las columnas usadas en el entrenamiento.
    """
    importances = model.feature_importances_
    importancia_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    top15 = importancia_df.head(15)
    logger.info("Top 15 features por importancia (gain):")
    for _, row in top15.iterrows():
        logger.info("  %-45s %.6f", row["feature"], row["importance"])

    # Grafico
    _SAVED_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top15["feature"][::-1], top15["importance"][::-1], color="steelblue")
    ax.set_xlabel("Importancia (gain)")
    ax.set_title("XGBoost — Top 15 Features (CMG Quillota 220kV)")
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()

    ruta = _SAVED_DIR / "xgboost_feature_importance.png"
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    logger.info("Grafico de importancia guardado en: %s", ruta)


# =============================================================================
# GUARDADO
# =============================================================================

def save_model(model: xgb.XGBRegressor) -> None:
    """Serializar el modelo en formato JSON de XGBoost.

    Guarda en models/saved/xgboost_model.json. El formato JSON es
    portable y puede cargarse con xgb.XGBRegressor().load_model().

    Args:
        model: Modelo XGBRegressor entrenado.
    """
    _SAVED_DIR.mkdir(parents=True, exist_ok=True)
    ruta = _SAVED_DIR / "xgboost_model.json"
    model.save_model(str(ruta))
    logger.info("Modelo guardado en: %s", ruta)


def save_results(
    metrics: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    engine: Engine,
) -> None:
    """Guardar predicciones en la tabla predictions de PostgreSQL.

    Inserta con model_name='XGBoost', model_version='1.0', horizon_h=1.
    Usa ON CONFLICT DO NOTHING para idempotencia.

    Args:
        metrics: Diccionario producido por evaluate_model(), con clave
            'predictions' conteniendo la Serie de predicciones.
        X_test: Features del test (para obtener el indice datetime).
        y_test: Target real del test (para actual_cmg).
        engine: Engine de SQLAlchemy conectado a cen_data.
    """
    predictions: pd.Series = metrics["predictions"]

    records = [
        {
            "datetime":      ts,
            "barra":         BARRA,
            "model_name":    "XGBoost",
            "model_version": "1.0",
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

    logger.info("=" * 50)
    logger.info("RESUMEN METRICAS XGBoost")
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
        # ── Carga ──────────────────────────────────────────────────────────────
        df = load_feature_matrix(engine)

        # ── Split ──────────────────────────────────────────────────────────────
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(df)

        # ── Entrenamiento ──────────────────────────────────────────────────────
        model = train_xgboost(X_train, y_train, X_val, y_val)

        # ── Evaluacion ─────────────────────────────────────────────────────────
        metrics = evaluate_model(model, X_test, y_test)

        # ── Feature importance ─────────────────────────────────────────────────
        plot_feature_importance(model, FEATURE_COLS)

        # ── Guardar modelo y resultados ────────────────────────────────────────
        save_model(model)
        save_results(metrics, X_test, y_test, engine)

        logger.info(
            "Pipeline XGBoost completado en %.1fs",
            time.perf_counter() - t_total,
        )

    except KeyboardInterrupt:
        logger.warning("Interrumpido por el usuario.")
        sys.exit(0)
    finally:
        engine.dispose()
