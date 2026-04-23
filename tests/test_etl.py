"""Tests para scrapers y pipeline ETL.

Verifica parse_tsv(), clean_marginal_costs() y upsert_dataframe()
sin conectarse a la base de datos real (uso de archivos temporales y mocks).
"""

import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from etl.scrapers.cen_marginal import parse_tsv
from etl.transform import clean_marginal_costs
from etl.load import upsert_dataframe


# =============================================================================
# FIXTURES
# =============================================================================

_TSV_HEADER = (
    "barra_mnemotecnico\t"
    "barra_referencia_mnemotecnico\t"
    "fecha\t"
    "hora\t"
    "costo_en_dolares\t"
    "costo_en_pesos\t"
    "nombre\n"
)


def _make_tsv_file(tmp_path: Path, filas: str, nombre: str = "cmg_test.tsv") -> Path:
    """Crear un archivo TSV temporal con cabecera CEN y las filas indicadas."""
    contenido = _TSV_HEADER + filas
    ruta = tmp_path / nombre
    ruta.write_text(contenido, encoding="utf-8")
    return ruta


# =============================================================================
# TEST 1 — Estructura básica del parse
# =============================================================================

def test_parse_tsv_estructura(tmp_path: Path) -> None:
    """Verifica que parse_tsv retorna las 3 columnas correctas con tipos correctos.

    Comprueba: 3 filas, columnas [datetime, barra, cmg_usd_mwh], dtype float64,
    timezone UTC y conversión correcta de decimales con coma europea.
    """
    filas = (
        "BA01\tBA02\t2024-03-10\t1\t59,80\t51234\tQuillota\n"
        "BA01\tBA02\t2024-03-10\t2\t62,50\t53210\tQuillota\n"
        "BA01\tBA02\t2024-03-10\t3\t110,25\t94500\tQuillota\n"
    )
    ruta = _make_tsv_file(tmp_path, filas)

    df = parse_tsv(ruta)

    assert len(df) == 3, f"Esperadas 3 filas, obtenidas {len(df)}"
    assert list(df.columns) == ["datetime", "barra", "cmg_usd_mwh"]
    assert df["cmg_usd_mwh"].dtype == np.float64
    assert str(df["datetime"].dtype) == "datetime64[ns, UTC]"

    # Decimales con coma convertidos correctamente
    assert pytest.approx(df["cmg_usd_mwh"].iloc[0], rel=1e-4) == 59.80
    assert pytest.approx(df["cmg_usd_mwh"].iloc[1], rel=1e-4) == 62.50
    assert pytest.approx(df["cmg_usd_mwh"].iloc[2], rel=1e-4) == 110.25


# =============================================================================
# TEST 2 — Conversión hora 1→00:00 y hora 24→23:00
# =============================================================================

def test_parse_tsv_hora_a_datetime(tmp_path: Path) -> None:
    """Verifica que hora=1 produce 00:00 UTC y hora=24 produce 23:00 UTC.

    El CEN usa horas 1-24 (no 0-23). El parser debe restar 1 a cada
    hora para obtener el offset desde medianoche: hora 1 = 0h, hora 24 = 23h.
    """
    filas = (
        "BA01\tBA02\t2024-01-15\t1\t50,00\t43000\tQuillota\n"
        "BA01\tBA02\t2024-01-15\t24\t75,00\t64000\tQuillota\n"
    )
    ruta = _make_tsv_file(tmp_path, filas)

    df = parse_tsv(ruta).sort_values("datetime").reset_index(drop=True)

    dt_0 = df["datetime"].iloc[0].tz_convert("America/Santiago")
    dt_23 = df["datetime"].iloc[1].tz_convert("America/Santiago")

    assert dt_0.hour == 0,  f"hora=1 debe → 00:00, obtuvo {dt_0.hour:02d}:00"
    assert dt_23.hour == 23, f"hora=24 debe → 23:00, obtuvo {dt_23.hour:02d}:00"


# =============================================================================
# TEST 3 — Valores negativos imputados
# =============================================================================

def test_clean_marginal_costs_negativos() -> None:
    """Verifica que clean_marginal_costs reemplaza negativos y marca is_imputed=True.

    Un CMG negativo es físicamente inválido. El pipeline debe convertirlo
    a NaN y luego imputarlo, dejando is_imputed=True en esas filas.
    """
    idx = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    valores = [50.0, -5.0, 60.0, 55.0, 52.0, 48.0, 53.0, 61.0, 58.0, 49.0]
    df_input = pd.DataFrame({
        "datetime":    idx,
        "barra":       "Quillota 220kV",
        "cmg_usd_mwh": valores,
    })

    df_clean = clean_marginal_costs(df_input)

    assert (df_clean["cmg_usd_mwh"] >= 0).all(), "No deben quedar valores negativos"
    assert df_clean.loc[df_clean["is_imputed"], "cmg_usd_mwh"].notna().all()
    # La fila con valor negativo (índice 1) debe estar marcada como imputada
    fila_neg = df_clean[df_clean["datetime"] == idx[1]]
    assert fila_neg["is_imputed"].iloc[0], "Fila con negativo debe tener is_imputed=True"


# =============================================================================
# TEST 4 — Outliers extremos imputados
# =============================================================================

def test_clean_marginal_costs_outliers() -> None:
    """Verifica que valores extremos (>Q99.9×3) son marcados como imputados.

    Un spike absurdo (999999 USD/MWh) debe ser detectado como outlier y
    reemplazado, quedando is_imputed=True en esa fila.
    """
    # 1.000 valores normales [30, 120] → Q99.9 ≈ 119, umbral ≈ 357.
    # El spike de 1.500 supera el umbral sin distorsionar el percentil.
    np.random.seed(42)
    normales = np.random.uniform(30, 120, size=1_000).tolist()
    idx = pd.date_range("2024-01-01", periods=1_001, freq="h", tz="UTC")
    valores = normales + [1_500.0]

    df_input = pd.DataFrame({
        "datetime":    idx,
        "barra":       "Quillota 220kV",
        "cmg_usd_mwh": valores,
    })

    df_clean = clean_marginal_costs(df_input)

    # El spike debe haber sido imputado
    fila_spike = df_clean[df_clean["datetime"] == idx[-1]]
    assert fila_spike["is_imputed"].iloc[0], "El spike extremo debe tener is_imputed=True"
    assert fila_spike["cmg_usd_mwh"].iloc[0] < 1_500.0, "El spike debe haber sido reemplazado"


# =============================================================================
# TEST 5 — upsert con DataFrame vacío retorna 0 sin error
# =============================================================================

def test_upsert_dataframe_vacio() -> None:
    """Verifica que upsert_dataframe con DataFrame vacío retorna 0 sin lanzar.

    Una ejecución idempotente del pipeline puede producir DataFrames vacíos
    si no hay datos nuevos. La función debe manejar este caso silenciosamente.
    """
    df_vacio = pd.DataFrame(columns=["datetime", "barra", "cmg_usd_mwh"])
    engine_mock = MagicMock()

    resultado = upsert_dataframe(
        df=df_vacio,
        table="marginal_costs",
        conflict_cols=["datetime", "barra"],
        engine=engine_mock,
    )

    assert resultado == 0
    # No debe haber intentado ejecutar SQL
    engine_mock.begin.assert_not_called()


# =============================================================================
# TEST 6 — upsert con conflict_cols inválidas lanza ValueError
# =============================================================================

def test_upsert_dataframe_conflict_cols_invalidas() -> None:
    """Verifica que conflict_cols con columnas inexistentes en el DataFrame lanza ValueError.

    Si el llamador especifica una columna de conflicto que no existe en el
    DataFrame, es un error de programación que debe fallar explícitamente.
    """
    df = pd.DataFrame({"datetime": pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
                        "cmg_usd_mwh": [50.0, 51.0, 52.0]})
    engine_mock = MagicMock()

    with pytest.raises(ValueError, match="conflict_cols"):
        upsert_dataframe(
            df=df,
            table="marginal_costs",
            conflict_cols=["columna_inexistente"],
            engine=engine_mock,
        )
