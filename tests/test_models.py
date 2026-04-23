"""Tests de estructura y artefactos de modelos entrenados.

Verifica que los archivos serializados existen, son cargables y producen
los atributos esperados. No requiere PostgreSQL ni re-entrenamiento.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_SAVED_DIR   = _ROOT / "models" / "saved"
_REPORTS_DIR = _SAVED_DIR / "reports"

_XGBOOST_PATH = _SAVED_DIR / "xgboost_model.json"
_LSTM_PATH    = _SAVED_DIR / "lstm_best.keras"
_METRICS_CSV  = _REPORTS_DIR / "metrics_comparison.csv"

_EXPECTED_FEATURE_COUNT = 29
_EXPECTED_METRIC_COLS   = {"model_name", "mae", "rmse", "mape", "r2"}
_EXPECTED_MODEL_COUNT   = 3
_MAE_THRESHOLD_XGBOOST  = 15.0


# =============================================================================
# TEST 1 — Archivo del modelo XGBoost existe
# =============================================================================

def test_xgboost_modelo_guardado_existe() -> None:
    """Verifica que el archivo serializado del modelo XGBoost existe en disco.

    El modelo se guarda en models/saved/xgboost_model.json al ejecutar
    python models/xgboost_model.py. Este test detecta si el modelo
    nunca fue entrenado o si el archivo fue borrado accidentalmente.
    """
    assert _XGBOOST_PATH.exists(), (
        f"Modelo XGBoost no encontrado en {_XGBOOST_PATH}. "
        "Ejecuta: python models/xgboost_model.py"
    )


# =============================================================================
# TEST 2 — Modelo XGBoost cargable y con 29 features
# =============================================================================

def test_xgboost_modelo_cargable() -> None:
    """Verifica que el modelo XGBoost se puede cargar y tiene 29 features.

    Carga el modelo desde JSON y comprueba que feature_importances_ tiene
    exactamente 29 elementos, coincidiendo con FEATURE_COLS definido en
    models/xgboost_model.py.
    """
    pytest.importorskip("xgboost", reason="xgboost no instalado")
    import xgboost as xgb

    model = xgb.XGBRegressor()
    model.load_model(str(_XGBOOST_PATH))

    assert hasattr(model, "feature_importances_"), (
        "El modelo cargado no tiene atributo feature_importances_"
    )
    n_features = len(model.feature_importances_)
    assert n_features == _EXPECTED_FEATURE_COUNT, (
        f"Se esperaban {_EXPECTED_FEATURE_COUNT} features, obtenidos {n_features}"
    )


# =============================================================================
# TEST 3 — Archivo del modelo LSTM existe
# =============================================================================

def test_lstm_modelo_guardado_existe() -> None:
    """Verifica que el checkpoint del mejor modelo LSTM existe en disco.

    El archivo se guarda durante el entrenamiento via ModelCheckpoint de Keras
    en models/saved/lstm_best.keras. Su ausencia indica que el entrenamiento
    nunca completó al menos una época de mejora.
    """
    assert _LSTM_PATH.exists(), (
        f"Modelo LSTM no encontrado en {_LSTM_PATH}. "
        "Ejecuta: python models/lstm_model.py"
    )


# =============================================================================
# TEST 4 — CSV de métricas existe, tiene estructura correcta y XGBoost < 15
# =============================================================================

def test_metricas_csv_existe() -> None:
    """Verifica que el CSV comparativo de métricas existe y tiene contenido válido.

    Comprueba: existencia del archivo, columnas requeridas, exactamente
    3 filas (una por modelo) y que XGBoost tiene MAE < 15 USD/MWh.
    """
    assert _METRICS_CSV.exists(), (
        f"CSV de métricas no encontrado en {_METRICS_CSV}. "
        "Ejecuta: python models/evaluate.py"
    )

    df = pd.read_csv(_METRICS_CSV)

    cols_faltantes = _EXPECTED_METRIC_COLS - set(df.columns)
    assert not cols_faltantes, (
        f"Columnas faltantes en metrics_comparison.csv: {cols_faltantes}"
    )

    assert len(df) == _EXPECTED_MODEL_COUNT, (
        f"Se esperaban {_EXPECTED_MODEL_COUNT} filas (SARIMA, XGBoost, LSTM), "
        f"obtenidas {len(df)}"
    )

    xgb_row = df[df["model_name"] == "XGBoost"]
    assert len(xgb_row) == 1, "No se encontró la fila de XGBoost en el CSV"

    mae_xgb = float(xgb_row["mae"].iloc[0])
    assert mae_xgb < _MAE_THRESHOLD_XGBOOST, (
        f"MAE de XGBoost ({mae_xgb:.2f}) supera el umbral de {_MAE_THRESHOLD_XGBOOST} USD/MWh"
    )
