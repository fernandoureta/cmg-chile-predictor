"""Tests para el módulo de feature engineering.

Verifica encoding cíclico, lags y rolling stats sin tocar PostgreSQL.
Todas las funciones bajo test son helpers privados de build_features.py,
invocados directamente para testear la lógica matemática.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from features.build_features import (
    _add_cyclic_encoding,
    _add_lag_features,
    _add_rolling_features,
)


# =============================================================================
# TEST 1 — Encoding cíclico produce valores en [-1, 1]
# =============================================================================

def test_cyclic_encoding_rango() -> None:
    """Verifica que sin/cos del encoding cíclico permanecen en [-1, 1] para las 24 horas.

    El encoding cíclico transforma la hora h → (sin(2π·h/24), cos(2π·h/24)).
    Por definición del seno y coseno, ambos valores deben estar en [-1, 1].
    """
    idx_utc = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
    dt_santiago = idx_utc.tz_convert("America/Santiago")

    df = pd.DataFrame(index=idx_utc)
    df = _add_cyclic_encoding(df, dt_santiago)

    for col in ["hora_sin", "hora_cos", "dow_sin", "dow_cos", "mes_sin", "mes_cos"]:
        assert col in df.columns, f"Columna '{col}' no encontrada"
        assert df[col].between(-1.0, 1.0).all(), (
            f"'{col}' tiene valores fuera de [-1, 1]: "
            f"min={df[col].min():.4f}, max={df[col].max():.4f}"
        )


# =============================================================================
# TEST 2 — Continuidad cíclica entre hora 23 y hora 0
# =============================================================================

def test_cyclic_encoding_continuidad() -> None:
    """Verifica que hora 23 y hora 0 están cerca en el espacio (sin, cos).

    El encoding cíclico debe preservar la continuidad del ciclo: la distancia
    euclidiana entre (sin_23, cos_23) y (sin_0, cos_0) debe ser < 0.3,
    reflejando que las 23:00 y las 00:00 son horas adyacentes.
    """
    idx_utc = pd.date_range("2024-01-01 00:00", periods=24, freq="h", tz="UTC")
    dt_santiago = idx_utc.tz_convert("America/Santiago")

    df = pd.DataFrame(index=idx_utc)
    df = _add_cyclic_encoding(df, dt_santiago)

    hora_local = dt_santiago.hour

    sin_0  = df.loc[idx_utc[hora_local == 0][0],  "hora_sin"]
    cos_0  = df.loc[idx_utc[hora_local == 0][0],  "hora_cos"]
    sin_23 = df.loc[idx_utc[hora_local == 23][0], "hora_sin"]
    cos_23 = df.loc[idx_utc[hora_local == 23][0], "hora_cos"]

    distancia = np.sqrt((sin_23 - sin_0) ** 2 + (cos_23 - cos_0) ** 2)

    assert distancia < 0.3, (
        f"Hora 23 y hora 0 deben estar cerca en espacio cíclico "
        f"(distancia={distancia:.4f}, umbral=0.3)"
    )


# =============================================================================
# TEST 3 — Lags sin data leakage: lag_1h[T] == valor[T-1]
# =============================================================================

def test_lags_no_leakage() -> None:
    """Verifica que lag_1h en el tiempo T contiene el valor de T-1, no T.

    El lag debe desplazar la serie 1 posición hacia el futuro, de modo que
    en el tiempo T el modelo solo ve el valor observado en T-1. El primer
    elemento del lag debe ser NaN (sin valor anterior disponible).
    """
    idx = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    valores = [float(i * 10) for i in range(10)]  # 0, 10, 20, ..., 90

    df = pd.DataFrame({"cmg_usd_mwh": valores}, index=idx)
    df = _add_lag_features(df, col="cmg_usd_mwh")

    assert pd.isna(df["lag_cmg_usd_mwh_1h"].iloc[0]), "lag_1h[0] debe ser NaN"

    for i in range(1, 10):
        assert df["lag_cmg_usd_mwh_1h"].iloc[i] == pytest.approx(valores[i - 1], rel=1e-6), (
            f"lag_1h[{i}] = {df['lag_cmg_usd_mwh_1h'].iloc[i]} ≠ valor[{i-1}] = {valores[i-1]}"
        )


# =============================================================================
# TEST 4 — Rolling stats sin data leakage: shift(1) antes del rolling
# =============================================================================

def test_rolling_no_leakage() -> None:
    """Verifica que rolling_mean_24h en T no incluye el valor de T (solo T-1..T-24).

    La función usa shift(1) antes del rolling, por lo que el valor en T está
    excluido de la media. Comprobamos que rolling_mean_24h[T] es la media de
    los valores anteriores, no contaminada por T.
    """
    n = 50
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    # Serie constante = 100 excepto en la última posición = 999
    valores = [100.0] * (n - 1) + [999.0]

    df = pd.DataFrame({"cmg_usd_mwh": valores}, index=idx)
    df = _add_rolling_features(df, col="cmg_usd_mwh")

    rolling_last = df["rolling_mean_cmg_usd_mwh_24h"].iloc[-1]
    assert rolling_last == pytest.approx(100.0, rel=1e-3), (
        f"rolling_mean_24h en T no debe incluir T: "
        f"obtenido {rolling_last:.2f}, esperado ~100.0"
    )
