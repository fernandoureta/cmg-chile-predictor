"""Evaluacion comparativa de modelos CMG Chile Predictor.

Carga predicciones desde la tabla predictions de PostgreSQL,
calcula metricas (MAE, RMSE, MAPE, R2) por modelo y genera
graficos comparativos y reporte CSV.
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
MODELOS: list[str] = ["SARIMA", "XGBoost", "LSTM"]
SAVED_DIR: Path = Path("models/saved")
REPORTS_DIR: Path = SAVED_DIR / "reports"


# =============================================================================
# CARGA
# =============================================================================

def load_predictions(engine: Engine) -> pd.DataFrame:
    """Cargar predicciones desde la tabla predictions de PostgreSQL.

    Filtra por barra Quillota 220kV, horizon_h=1 y actual_cmg NOT NULL.
    Convierte datetime a zona America/Santiago para analisis por hora.

    Args:
        engine: Engine de SQLAlchemy conectado a cen_data.

    Returns:
        DataFrame con columnas: datetime (UTC), model_name, model_version,
        predicted_cmg, actual_cmg, hora_santiago (int 0-23).
    """
    sql = text("""
        SELECT
            datetime,
            model_name,
            model_version,
            predicted_cmg,
            actual_cmg
        FROM predictions
        WHERE barra        = :barra
          AND horizon_h    = 1
          AND actual_cmg   IS NOT NULL
        ORDER BY model_name, datetime
    """)

    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"barra": BARRA})

    if df.empty:
        logger.warning("No hay predicciones en la tabla 'predictions' para %s.", BARRA)
        return df

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["hora_santiago"] = (
        df["datetime"]
        .dt.tz_convert("America/Santiago")
        .dt.hour
    )

    logger.info(
        "Predicciones cargadas: %d filas | modelos: %s",
        len(df),
        df["model_name"].unique().tolist(),
    )
    return df


# =============================================================================
# METRICAS
# =============================================================================

def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calcular MAE, RMSE, MAPE y R2 por modelo.

    MAPE excluye registros donde actual_cmg == 0 para evitar
    division por cero (mediodias solares). Ordena por MAE ascendente.

    Args:
        df: DataFrame producido por load_predictions().

    Returns:
        DataFrame con columnas: model_name, mae, rmse, mape, r2,
        n_predictions; una fila por modelo, ordenado por mae.
    """
    if df.empty:
        logger.warning("DataFrame vacio — no se pueden calcular metricas.")
        return pd.DataFrame()

    rows: list[dict] = []
    for model_name, grp in df.groupby("model_name"):
        yt = grp["actual_cmg"].values.astype(float)
        yp = grp["predicted_cmg"].values.astype(float)

        mask = np.isfinite(yt) & np.isfinite(yp)
        yt_v = yt[mask]
        yp_v = yp[mask]

        if len(yt_v) == 0:
            logger.warning("Modelo %s — sin predicciones validas.", model_name)
            continue

        mae  = float(np.mean(np.abs(yt_v - yp_v)))
        rmse = float(np.sqrt(np.mean((yt_v - yp_v) ** 2)))
        ss_res = np.sum((yt_v - yp_v) ** 2)
        ss_tot = np.sum((yt_v - np.mean(yt_v)) ** 2)
        r2   = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

        mask_nz = mask & (yt != 0)
        yt_nz = yt[mask_nz]
        yp_nz = yp[mask_nz]
        mape = float(np.mean(np.abs((yt_nz - yp_nz) / yt_nz)) * 100) if len(yt_nz) > 0 else float("nan")

        rows.append({
            "model_name":    model_name,
            "mae":           mae,
            "rmse":          rmse,
            "mape":          mape,
            "r2":            r2,
            "n_predictions": int(mask.sum()),
        })

    metrics_df = (
        pd.DataFrame(rows)
        .sort_values("mae")
        .reset_index(drop=True)
    )

    logger.info("Metricas calculadas para %d modelos.", len(metrics_df))
    return metrics_df


# =============================================================================
# GRAFICOS
# =============================================================================

def plot_predictions_vs_real(df: pd.DataFrame, output_dir: Path) -> None:
    """Grafico de lineas — predicciones vs real (primeros 30 dias por modelo).

    Genera un subplot por modelo, muestra la serie real y la predicha
    para los primeros 30 dias disponibles de cada modelo.
    Guarda en output_dir/predictions_vs_real.png.

    Args:
        df: DataFrame producido por load_predictions().
        output_dir: Directorio donde guardar el PNG.
    """
    if df.empty:
        logger.warning("DataFrame vacio — saltando plot predictions_vs_real.")
        return

    modelos_presentes = [m for m in MODELOS if m in df["model_name"].unique()]
    n = len(modelos_presentes)
    if n == 0:
        logger.warning("Ningun modelo conocido en los datos — saltando plot.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(n, 1, figsize=(16, 5 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, modelo in zip(axes, modelos_presentes):
        grp = df[df["model_name"] == modelo].sort_values("datetime")
        inicio = grp["datetime"].min()
        fin    = inicio + pd.Timedelta(days=30)
        grp30  = grp[grp["datetime"] <= fin]

        ax.plot(grp30["datetime"], grp30["actual_cmg"],    label="Real",      color="steelblue",  linewidth=1.2)
        ax.plot(grp30["datetime"], grp30["predicted_cmg"], label="Prediccion", color="darkorange", linewidth=1.0, alpha=0.85)
        ax.set_title(f"{modelo} — Prediccion vs Real (primeros 30 dias)")
        ax.set_ylabel("CMG (USD/MWh)")
        ax.legend(loc="upper right", fontsize=8)
        ax.tick_params(axis="x", rotation=30, labelsize=8)

    fig.suptitle("Comparacion Predicciones vs Real — Barra Quillota 220kV", fontsize=13, y=1.01)
    plt.tight_layout()

    ruta = output_dir / "predictions_vs_real.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Grafico guardado: %s", ruta)


def plot_error_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Histograma de distribucion de errores (predicho - real) por modelo.

    Un subplot por modelo. Guarda en output_dir/error_distribution.png.

    Args:
        df: DataFrame producido por load_predictions().
        output_dir: Directorio donde guardar el PNG.
    """
    if df.empty:
        logger.warning("DataFrame vacio — saltando plot error_distribution.")
        return

    modelos_presentes = [m for m in MODELOS if m in df["model_name"].unique()]
    n = len(modelos_presentes)
    if n == 0:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, modelo in zip(axes, modelos_presentes):
        grp   = df[df["model_name"] == modelo]
        error = grp["predicted_cmg"].values - grp["actual_cmg"].values
        error = error[np.isfinite(error)]

        ax.hist(error, bins=60, color="steelblue", edgecolor="white", linewidth=0.4)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.2)
        ax.set_title(f"{modelo}")
        ax.set_xlabel("Error (USD/MWh)")
        ax.set_ylabel("Frecuencia")

        mae_val = float(np.mean(np.abs(error)))
        ax.text(
            0.97, 0.96,
            f"MAE={mae_val:.2f}",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        )

    fig.suptitle("Distribucion de Errores por Modelo", fontsize=13)
    plt.tight_layout()

    ruta = output_dir / "error_distribution.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Grafico guardado: %s", ruta)


def plot_error_by_hour(df: pd.DataFrame, output_dir: Path) -> None:
    """MAE por hora del dia (0-23, zona America/Santiago) por modelo.

    Un subplot por modelo. Util para detectar el efecto mediodia solar
    (precios cercanos a 0 → picos de error relativo).
    Guarda en output_dir/error_by_hour.png.

    Args:
        df: DataFrame producido por load_predictions().
        output_dir: Directorio donde guardar el PNG.
    """
    if df.empty:
        logger.warning("DataFrame vacio — saltando plot error_by_hour.")
        return

    modelos_presentes = [m for m in MODELOS if m in df["model_name"].unique()]
    n = len(modelos_presentes)
    if n == 0:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, modelo in zip(axes, modelos_presentes):
        grp = df[df["model_name"] == modelo].copy()
        grp["abs_error"] = np.abs(grp["predicted_cmg"] - grp["actual_cmg"])

        mae_hora = (
            grp.groupby("hora_santiago")["abs_error"]
            .mean()
            .reindex(range(24), fill_value=np.nan)
        )

        ax.bar(mae_hora.index, mae_hora.values, color="steelblue", edgecolor="white", linewidth=0.4)
        ax.set_title(f"{modelo}")
        ax.set_xlabel("Hora del dia (Santiago)")
        ax.set_ylabel("MAE (USD/MWh)")
        ax.set_xticks(range(0, 24, 2))

    fig.suptitle("MAE por Hora del Dia — Barra Quillota 220kV", fontsize=13)
    plt.tight_layout()

    ruta = output_dir / "error_by_hour.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Grafico guardado: %s", ruta)


# =============================================================================
# REPORTE
# =============================================================================

def generate_report(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    """Guardar CSV de metricas y loggear tabla comparativa.

    Guarda metrics_df en output_dir/metrics_comparison.csv.
    Loggea tabla formateada con alineacion columnar y mejor modelo
    por cada metrica.

    Args:
        metrics_df: DataFrame producido por compute_metrics().
        output_dir: Directorio donde guardar el CSV.
    """
    if metrics_df.empty:
        logger.warning("metrics_df vacio — no se genera reporte.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    ruta_csv = output_dir / "metrics_comparison.csv"
    metrics_df.to_csv(ruta_csv, index=False)
    logger.info("Reporte CSV guardado: %s", ruta_csv)

    # Tabla formateada en log
    sep = "=" * 75
    logger.info(sep)
    logger.info("COMPARACION DE MODELOS — CMG Quillota 220kV")
    logger.info(sep)
    header = f"{'Modelo':<12} {'MAE':>10} {'RMSE':>10} {'MAPE':>10} {'R2':>8} {'N':>8}"
    logger.info(header)
    logger.info("-" * 75)
    for _, row in metrics_df.iterrows():
        logger.info(
            "%-12s %10.4f %10.4f %9.2f%% %8.4f %8d",
            row["model_name"],
            row["mae"],
            row["rmse"],
            row["mape"],
            row["r2"],
            int(row["n_predictions"]),
        )
    logger.info(sep)

    # Mejor modelo por metrica
    for metrica, ascending in [("mae", True), ("rmse", True), ("mape", True), ("r2", False)]:
        mejor = metrics_df.sort_values(metrica, ascending=ascending).iloc[0]
        val   = mejor[metrica]
        sufijo = "%" if metrica == "mape" else ""
        logger.info(
            "Mejor %s: %-12s (%.4f%s)",
            metrica.upper(),
            mejor["model_name"],
            val,
            sufijo,
        )


# =============================================================================
# EJECUCION DIRECTA
# =============================================================================

if __name__ == "__main__":
    import time

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    engine = create_engine(DB_URL, pool_pre_ping=True)
    t_total = time.perf_counter()

    try:
        # ── Carga ──────────────────────────────────────────────────────────────
        df = load_predictions(engine)

        if df.empty:
            logger.error(
                "No hay predicciones disponibles. "
                "Ejecuta primero sarima.py, xgboost_model.py y lstm_model.py."
            )
            sys.exit(1)

        # ── Metricas ───────────────────────────────────────────────────────────
        metrics_df = compute_metrics(df)

        # ── Graficos ───────────────────────────────────────────────────────────
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        plot_predictions_vs_real(df, REPORTS_DIR)
        plot_error_distribution(df, REPORTS_DIR)
        plot_error_by_hour(df, REPORTS_DIR)

        # ── Reporte ────────────────────────────────────────────────────────────
        generate_report(metrics_df, REPORTS_DIR)

        logger.info(
            "Evaluacion completada en %.1fs",
            time.perf_counter() - t_total,
        )

    except KeyboardInterrupt:
        logger.warning("Interrumpido por el usuario.")
        sys.exit(0)
    finally:
        engine.dispose()
