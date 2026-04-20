"""Modelo LSTM multivariado para prediccion CMG.

Red neuronal con dos capas LSTM (128 -> 64), Dropout, BatchNormalization
y Dense de salida. Ventana de entrada: 168 horas (7 dias).
Horizonte de salida: 1 hora (prediccion T+1).

Arquitectura segun CLAUDE.md:
  LSTM(128, return_sequences=True) -> Dropout(0.2) -> BatchNorm
  LSTM(64)                         -> Dropout(0.2)
  Dense(64, relu)                  -> Dense(1)

Entrenado con Adam(lr=0.001), loss=MAE, EarlyStopping(patience=15).
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

os_environ_key = "TF_CPP_MIN_LOG_LEVEL"
import os
os.environ.setdefault(os_environ_key, "3")   # suprimir warnings de compilacion TF

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import DB_URL                                  # noqa: E402
from features.build_features import build_feature_matrix   # noqa: E402

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────
BARRA: str = "Quillota 220kV"
WINDOW_SIZE: int = 168
TRAIN_END: str  = "2023-06-30"
VAL_START: str  = "2023-07-01"
VAL_END: str    = "2023-12-31"
TEST_START: str = "2024-01-01"
TARGET: str = "cmg_usd_mwh"
BATCH_SIZE: int = 64
EPOCHS: int = 100
TARGET_COL_IDX: int = 0   # cmg_usd_mwh es la columna 0 del array escalado

_SAVED_DIR = _ROOT / "models" / "saved"

# Mismo orden que xgboost_model.py; el TARGET se antepone en load_and_prepare_data
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

# Orden de columnas en el DataFrame que entra al scaler: TARGET primero
ALL_COLS: list[str] = [TARGET] + FEATURE_COLS


# =============================================================================
# CARGA Y PREPARACION
# =============================================================================

def load_and_prepare_data(
    engine: Engine,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cargar feature matrix y dividir en train, val y test.

    Ordena las columnas con TARGET primero (columna 0) para que la
    inversion del scaler recupere el target por indice 0 sin ambiguedad.
    Elimina filas con NaN en cualquier columna antes del split.

    Args:
        engine: Engine de SQLAlchemy conectado a cen_data.

    Returns:
        Tupla (df_train, df_val, df_test) con columnas en orden ALL_COLS
        y sin NaN.
    """
    logger.info("Construyendo feature matrix...")
    t0 = time.perf_counter()
    df_full = build_feature_matrix(engine)
    logger.info(
        "Feature matrix: %d filas x %d columnas (%.1fs)",
        len(df_full),
        len(df_full.columns),
        time.perf_counter() - t0,
    )

    # Reordenar: TARGET primero, luego features exogenas
    df_full = df_full[ALL_COLS].dropna()
    logger.info("Filas tras dropna: %d", len(df_full))

    df_train = df_full.loc[:TRAIN_END]
    df_val   = df_full.loc[VAL_START:VAL_END]
    df_test  = df_full.loc[TEST_START:]

    logger.info(
        "Split | train: %d  val: %d  test: %d",
        len(df_train), len(df_val), len(df_test),
    )
    return df_train, df_val, df_test


# =============================================================================
# ESCALADO
# =============================================================================

def scale_data(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """Escalar todas las columnas al rango [0, 1] con MinMaxScaler.

    El scaler se ajusta SOLO sobre train para evitar data leakage.
    Se aplica el mismo scaler a val y test.

    Args:
        df_train: DataFrame del conjunto de train.
        df_val: DataFrame del conjunto de validacion.
        df_test: DataFrame del conjunto de test.

    Returns:
        Tupla (train_scaled, val_scaled, test_scaled, scaler) donde
        los arrays tienen shape (n, n_features) con dtype float32.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))

    train_scaled = scaler.fit_transform(df_train.values).astype(np.float32)
    val_scaled   = scaler.transform(df_val.values).astype(np.float32)
    test_scaled  = scaler.transform(df_test.values).astype(np.float32)

    logger.info(
        "Escalado completado | train=%s  val=%s  test=%s",
        train_scaled.shape,
        val_scaled.shape,
        test_scaled.shape,
    )
    return train_scaled, val_scaled, test_scaled, scaler


# =============================================================================
# SECUENCIAS
# =============================================================================

def create_sequences(
    data: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Crear secuencias deslizantes para entrenamiento LSTM.

    Para cada paso t, crea una ventana de entrada con las filas
    [t, t+window_size) y un target con el valor de la columna TARGET_COL_IDX
    en el instante t+window_size (el siguiente paso fuera de la ventana).

    Args:
        data: Array de shape (n_timesteps, n_features) escalado.
        window_size: Numero de pasos de tiempo en cada secuencia.

    Returns:
        Tupla (X, y) donde:
        - X: shape (n_samples, window_size, n_features)
        - y: shape (n_samples,) — target de la hora siguiente a la ventana
    """
    X_list: list[np.ndarray] = []
    y_list: list[float] = []

    for i in range(len(data) - window_size):
        X_list.append(data[i : i + window_size])
        y_list.append(data[i + window_size, TARGET_COL_IDX])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    logger.debug("Secuencias creadas: X=%s  y=%s", X.shape, y.shape)
    return X, y


# =============================================================================
# ARQUITECTURA
# =============================================================================

def build_lstm_model(n_features: int) -> keras.Model:
    """Construir la arquitectura LSTM segun CLAUDE.md.

    Arquitectura:
      Input(WINDOW_SIZE, n_features)
      LSTM(128, return_sequences=True)
      Dropout(0.2)
      BatchNormalization()
      LSTM(64, return_sequences=False)
      Dropout(0.2)
      Dense(64, relu)
      Dense(1)

    Compilado con Adam(lr=0.001), loss='mae', metrics=['mse'].

    Args:
        n_features: Numero de features por timestep (30 en este proyecto).

    Returns:
        Modelo Keras compilado listo para entrenar.
    """
    inputs = keras.Input(shape=(WINDOW_SIZE, n_features), name="input_seq")

    x = layers.LSTM(128, return_sequences=True, name="lstm_1")(inputs)
    x = layers.Dropout(0.2, name="dropout_1")(x)
    x = layers.BatchNormalization(name="batchnorm_1")(x)

    x = layers.LSTM(64, return_sequences=False, name="lstm_2")(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)

    x = layers.Dense(64, activation="relu", name="dense_1")(x)
    output = layers.Dense(1, name="output")(x)

    model = keras.Model(inputs=inputs, outputs=output, name="CMG_LSTM")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mae",
        metrics=["mse"],
    )

    logger.info("Arquitectura LSTM construida:")
    model.summary(print_fn=lambda line: logger.info("  %s", line))
    return model


# =============================================================================
# ENTRENAMIENTO
# =============================================================================

def train_lstm(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[keras.Model, keras.callbacks.History]:
    """Entrenar el modelo LSTM con callbacks de control automatico.

    Callbacks configurados:
      - EarlyStopping: detiene si val_loss no mejora en 15 epocas,
        restaurando los pesos del mejor epoch.
      - ReduceLROnPlateau: reduce lr a la mitad si val_loss no mejora
        en 7 epocas consecutivas.
      - ModelCheckpoint: guarda el mejor modelo en models/saved/lstm_best.keras.

    Args:
        model: Modelo Keras compilado por build_lstm_model().
        X_train: Secuencias de train, shape (n, WINDOW_SIZE, n_features).
        y_train: Targets de train, shape (n,).
        X_val: Secuencias de validacion.
        y_val: Targets de validacion.

    Returns:
        Tupla (model, history) con el modelo con mejores pesos y el
        historial de entrenamiento.
    """
    _SAVED_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(_SAVED_DIR / "lstm_best.keras")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]

    logger.info(
        "Entrenando LSTM: epochs=%d, batch_size=%d | train=%d  val=%d",
        EPOCHS,
        BATCH_SIZE,
        len(X_train),
        len(X_val),
    )
    t0 = time.perf_counter()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    epochs_ejecutados = len(history.history["loss"])
    mejor_val_loss = min(history.history["val_loss"])
    dt = time.perf_counter() - t0

    logger.info(
        "Entrenamiento completado en %.1fs | epochs=%d | mejor val_loss=%.6f",
        dt,
        epochs_ejecutados,
        mejor_val_loss,
    )
    logger.info("Mejor modelo guardado en: %s", checkpoint_path)
    return model, history


# =============================================================================
# EVALUACION
# =============================================================================

def evaluate_lstm(
    model: keras.Model,
    X_test: np.ndarray,
    y_test_raw: np.ndarray,
    scaler: MinMaxScaler,
    n_features: int,
    test_index: pd.DatetimeIndex,
) -> dict:
    """Generar predicciones, invertir escala y calcular metricas.

    Las predicciones del modelo estan en escala [0,1] (aplicada por el
    scaler). Para invertir, se construye un array auxiliar con la prediccion
    en la columna TARGET_COL_IDX y ceros en el resto, luego se aplica
    scaler.inverse_transform() y se extrae la columna 0.

    Args:
        model: Modelo LSTM entrenado (pesos del mejor epoch).
        X_test: Secuencias de test, shape (n, WINDOW_SIZE, n_features).
        y_test_raw: Targets de test en escala normalizada, shape (n,).
        scaler: MinMaxScaler ajustado sobre el train.
        n_features: Numero total de features (30).
        test_index: DatetimeIndex del conjunto de test, alineado con y_test_raw.

    Returns:
        Diccionario con claves 'mae', 'rmse', 'mape', 'r2' y 'predictions'
        (pd.Series con valores en escala real, indice=test_index).
    """
    logger.info("Generando predicciones LSTM sobre %d secuencias...", len(X_test))
    t0 = time.perf_counter()

    y_pred_scaled = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).flatten()
    logger.info("Prediccion completada en %.1fs", time.perf_counter() - t0)

    # Invertir escala: reconstruir array completo con prediccion en col 0
    def _inverse_col0(scaled_col: np.ndarray) -> np.ndarray:
        dummy = np.zeros((len(scaled_col), n_features), dtype=np.float32)
        dummy[:, TARGET_COL_IDX] = scaled_col
        return scaler.inverse_transform(dummy)[:, TARGET_COL_IDX]

    y_pred_real = _inverse_col0(y_pred_scaled)
    y_true_real = _inverse_col0(y_test_raw)

    predictions = pd.Series(y_pred_real, index=test_index, name="cmg_predicted")

    mask_validos = np.isfinite(y_true_real) & np.isfinite(y_pred_real)
    yt = y_true_real[mask_validos]
    yp = y_pred_real[mask_validos]

    mae  = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    r2   = float(1 - np.sum((yt - yp) ** 2) / np.sum((yt - np.mean(yt)) ** 2))

    mask_no_cero = mask_validos & (y_true_real != 0)
    yt_nz = y_true_real[mask_no_cero]
    yp_nz = y_pred_real[mask_no_cero]
    mape  = float(np.mean(np.abs((yt_nz - yp_nz) / yt_nz)) * 100)

    metrics = {
        "mae":         mae,
        "rmse":        rmse,
        "mape":        mape,
        "r2":          r2,
        "predictions": predictions,
        "y_true_real": y_true_real,
    }

    logger.info("Metricas sobre test (LSTM):")
    logger.info("  MAE   = %.4f USD/MWh", mae)
    logger.info("  RMSE  = %.4f USD/MWh", rmse)
    logger.info("  MAPE  = %.2f%%", mape)
    logger.info("  R2    = %.4f", r2)
    return metrics


# =============================================================================
# GUARDADO
# =============================================================================

def save_results(
    metrics: dict,
    test_index: pd.DatetimeIndex,
    y_test_real: np.ndarray,
    engine: Engine,
) -> None:
    """Guardar predicciones LSTM en la tabla predictions de PostgreSQL.

    Inserta con model_name='LSTM', model_version='1.0', horizon_h=1.
    Usa ON CONFLICT DO NOTHING para idempotencia.

    Args:
        metrics: Diccionario de evaluate_lstm(), con clave 'predictions'.
        test_index: DatetimeIndex del conjunto de test.
        y_test_real: Valores reales en escala original (USD/MWh).
        engine: Engine de SQLAlchemy conectado a cen_data.
    """
    predictions: pd.Series = metrics["predictions"]

    records = [
        {
            "datetime":      ts,
            "barra":         BARRA,
            "model_name":    "LSTM",
            "model_version": "1.0",
            "predicted_cmg": float(pred),
            "actual_cmg":    float(actual) if np.isfinite(actual) else None,
            "horizon_h":     1,
        }
        for ts, pred, actual in zip(
            predictions.index,
            predictions.values,
            y_test_real,
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

    logger.info("=" * 55)
    logger.info("RESUMEN METRICAS LSTM")
    logger.info("=" * 55)
    logger.info("  %-10s %12.4f %s", "MAE",  metrics["mae"],  "USD/MWh")
    logger.info("  %-10s %12.4f %s", "RMSE", metrics["rmse"], "USD/MWh")
    logger.info("  %-10s %11.2f%%",  "MAPE", metrics["mape"])
    logger.info("  %-10s %12.4f",    "R2",   metrics["r2"])
    logger.info("=" * 55)


def _log_comparison(engine: Engine, lstm_metrics: dict) -> None:
    """Comparar metricas LSTM contra SARIMA y XGBoost si existen en BD.

    Args:
        engine: Engine conectado a cen_data.
        lstm_metrics: Metricas del LSTM recien calculadas.
    """
    sql = text("""
        SELECT model_name,
               AVG(ABS(predicted_cmg - actual_cmg))           AS mae,
               SQRT(AVG(POWER(predicted_cmg - actual_cmg, 2))) AS rmse
        FROM predictions
        WHERE actual_cmg IS NOT NULL
          AND barra = :barra
          AND horizon_h = 1
        GROUP BY model_name
        ORDER BY mae ASC
    """)
    try:
        with engine.connect() as conn:
            df_comp = pd.read_sql(sql, conn, params={"barra": BARRA})

        if df_comp.empty:
            return

        logger.info("=" * 55)
        logger.info("COMPARACION DE MODELOS (test, horizon_h=1)")
        logger.info("=" * 55)
        logger.info("  %-12s  %10s  %10s", "Modelo", "MAE", "RMSE")
        logger.info("  %s", "-" * 40)
        for _, row in df_comp.iterrows():
            logger.info(
                "  %-12s  %10.4f  %10.4f",
                row["model_name"],
                row["mae"],
                row["rmse"],
            )
        logger.info("=" * 55)
    except Exception as exc:
        logger.warning("No se pudo generar comparacion de modelos: %s", exc)


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
        # ── 1. Carga y split ───────────────────────────────────────────────────
        df_train, df_val, df_test = load_and_prepare_data(engine)

        # ── 2. Escalado ────────────────────────────────────────────────────────
        train_scaled, val_scaled, test_scaled, scaler = scale_data(
            df_train, df_val, df_test
        )
        n_features = train_scaled.shape[1]

        # ── 3. Secuencias ──────────────────────────────────────────────────────
        logger.info("Creando secuencias (window=%d)...", WINDOW_SIZE)
        X_train, y_train = create_sequences(train_scaled, WINDOW_SIZE)
        X_val,   y_val   = create_sequences(val_scaled,   WINDOW_SIZE)
        X_test,  y_test  = create_sequences(test_scaled,  WINDOW_SIZE)

        logger.info(
            "Secuencias | X_train=%s  X_val=%s  X_test=%s",
            X_train.shape, X_val.shape, X_test.shape,
        )

        # Indice datetime del test alineado con las secuencias
        # create_sequences produce n-window_size samples; el primero
        # corresponde al instante window_size del DataFrame original.
        test_index = df_test.index[WINDOW_SIZE:]

        # ── 4. Arquitectura ────────────────────────────────────────────────────
        model = build_lstm_model(n_features)

        # ── 5. Entrenamiento ───────────────────────────────────────────────────
        model, history = train_lstm(model, X_train, y_train, X_val, y_val)

        # ── 6. Evaluacion ──────────────────────────────────────────────────────
        metrics = evaluate_lstm(
            model, X_test, y_test, scaler, n_features, test_index
        )

        # ── 7. Guardar resultados y comparar ───────────────────────────────────
        save_results(metrics, test_index, metrics["y_true_real"], engine)
        _log_comparison(engine, metrics)

        logger.info(
            "Pipeline LSTM completado en %.1fs",
            time.perf_counter() - t_total,
        )

    except KeyboardInterrupt:
        logger.warning("Interrumpido por el usuario.")
        sys.exit(0)
    finally:
        engine.dispose()
