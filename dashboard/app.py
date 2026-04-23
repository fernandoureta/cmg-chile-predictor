"""Dashboard Streamlit — CMG Chile Predictor.

Tres paginas navegables via sidebar:
  1. Historico CMG   — serie completa + heatmap hora×mes
  2. Predicciones    — comparacion de modelos + error por hora
  3. Analisis drivers — scatter CMG vs generacion solar/gas
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, text

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import DB_URL  # noqa: E402

# ── Constantes ────────────────────────────────────────────────────────────────
BARRA: str = "Quillota 220kV"
MODELOS: list[str] = ["SARIMA", "XGBoost", "LSTM"]

COLOR_REAL:    str = "#111111"
COLOR_SARIMA:  str = "#888888"
COLOR_XGBOOST: str = "#1f77b4"
COLOR_LSTM:    str = "#ff7f0e"

COLOR_MODELO: dict[str, str] = {
    "Real":    COLOR_REAL,
    "SARIMA":  COLOR_SARIMA,
    "XGBoost": COLOR_XGBOOST,
    "LSTM":    COLOR_LSTM,
}

# ── Engine (singleton por sesion) ─────────────────────────────────────────────
@st.cache_resource
def get_engine():
    """Crear engine SQLAlchemy reutilizable entre reruns."""
    return create_engine(DB_URL, pool_pre_ping=True, pool_size=2, max_overflow=2)


# =============================================================================
# CONSULTAS SQL CON CACHE
# =============================================================================

@st.cache_data(ttl=3600)
def load_cmg() -> pd.DataFrame:
    """Cargar serie historica completa del CMG desde marginal_costs.

    Returns:
        DataFrame con columnas datetime (UTC) y cmg_usd_mwh,
        ordenado por datetime. Vacio si no hay datos.
    """
    sql = text("""
        SELECT datetime, cmg_usd_mwh
        FROM marginal_costs
        WHERE barra = :barra
          AND cmg_usd_mwh IS NOT NULL
        ORDER BY datetime
    """)
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"barra": BARRA})

    if df.empty:
        return df

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["datetime_santiago"] = df["datetime"].dt.tz_convert("America/Santiago")
    df["hora"]  = df["datetime_santiago"].dt.hour
    df["mes"]   = df["datetime_santiago"].dt.month
    return df


@st.cache_data(ttl=3600)
def load_predictions_dash() -> pd.DataFrame:
    """Cargar predicciones de los tres modelos desde tabla predictions.

    Returns:
        DataFrame con columnas datetime (UTC), model_name, predicted_cmg,
        actual_cmg, hora_santiago. Vacio si no hay datos.
    """
    sql = text("""
        SELECT datetime, model_name, predicted_cmg, actual_cmg
        FROM predictions
        WHERE barra      = :barra
          AND horizon_h  = 1
          AND actual_cmg IS NOT NULL
        ORDER BY model_name, datetime
    """)
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"barra": BARRA})

    if df.empty:
        return df

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["hora_santiago"] = df["datetime"].dt.tz_convert("America/Santiago").dt.hour
    return df


@st.cache_data(ttl=3600)
def load_generation() -> pd.DataFrame:
    """Cargar generacion por tecnologia desde generation_by_tech.

    Returns:
        DataFrame con datetime UTC, gen_solar_mw y gen_gas_mw.
    """
    sql = text("""
        SELECT datetime, gen_solar_mw, gen_gas_mw
        FROM generation_by_tech
        WHERE gen_solar_mw IS NOT NULL
          AND gen_gas_mw   IS NOT NULL
        ORDER BY datetime
    """)
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)

    if df.empty:
        return df

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df


# =============================================================================
# HELPERS DE METRICAS
# =============================================================================

def _compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calcular MAE, RMSE, MAPE, R2 por modelo sobre df de predicciones."""
    rows: list[dict] = []
    for model_name, grp in df.groupby("model_name"):
        yt = grp["actual_cmg"].values.astype(float)
        yp = grp["predicted_cmg"].values.astype(float)
        mask = np.isfinite(yt) & np.isfinite(yp)
        yt_v, yp_v = yt[mask], yp[mask]
        if len(yt_v) == 0:
            continue
        mae  = float(np.mean(np.abs(yt_v - yp_v)))
        rmse = float(np.sqrt(np.mean((yt_v - yp_v) ** 2)))
        ss_res = np.sum((yt_v - yp_v) ** 2)
        ss_tot = np.sum((yt_v - np.mean(yt_v)) ** 2)
        r2   = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        mask_nz = mask & (yt != 0)
        yt_nz, yp_nz = yt[mask_nz], yp[mask_nz]
        mape = float(np.mean(np.abs((yt_nz - yp_nz) / yt_nz)) * 100) if len(yt_nz) > 0 else float("nan")
        rows.append({
            "Modelo": model_name,
            "MAE (USD/MWh)": round(mae, 2),
            "RMSE (USD/MWh)": round(rmse, 2),
            "MAPE (%)": round(mape, 2),
            "R²": round(r2, 4),
            "N predicciones": int(mask.sum()),
        })
    return pd.DataFrame(rows).sort_values("MAE (USD/MWh)").reset_index(drop=True)


# =============================================================================
# CONFIGURACION GLOBAL
# =============================================================================

st.set_page_config(
    page_title="CMG Chile Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navegacion ────────────────────────────────────────────────────────
st.sidebar.title("⚡ CMG Chile Predictor")
st.sidebar.caption("Barra Quillota 220kV · Fuente: CEN")
st.sidebar.markdown("---")

pagina = st.sidebar.radio(
    "Navegacion",
    ["Historico CMG", "Predicciones y modelos", "Analisis de drivers"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Datos: [CEN](https://www.coordinador.cl) · "
    "[Open-Meteo](https://open-meteo.com)"
)


# =============================================================================
# PAGINA 1 — HISTORICO CMG
# =============================================================================

if pagina == "Historico CMG":
    st.title("Costo Marginal Electrico — Chile")
    st.caption("Barra Quillota 220kV | Fuente: Coordinador Electrico Nacional (CEN)")

    df_cmg = load_cmg()

    if df_cmg.empty:
        st.warning(
            "No hay datos disponibles en la tabla `marginal_costs`. "
            "Ejecuta primero el pipeline ETL: `python etl/pipeline.py --mode backfill`"
        )
        st.stop()

    # ── Metricas ultimas 24h ─────────────────────────────────────────────────
    ultimo_dt  = df_cmg["datetime"].max()
    hace_24h   = ultimo_dt - pd.Timedelta(hours=24)
    df_24h     = df_cmg[df_cmg["datetime"] >= hace_24h]
    ultimo_val = df_cmg.loc[df_cmg["datetime"] == ultimo_dt, "cmg_usd_mwh"].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "CMG mas reciente",
        f"{ultimo_val:.2f} USD/MWh",
        help=f"Dato de {ultimo_dt.tz_convert('America/Santiago').strftime('%Y-%m-%d %H:%M')} (Santiago)",
    )
    col2.metric(
        "Promedio ultimas 24h",
        f"{df_24h['cmg_usd_mwh'].mean():.2f} USD/MWh",
    )
    col3.metric(
        "Maximo ultimas 24h",
        f"{df_24h['cmg_usd_mwh'].max():.2f} USD/MWh",
    )
    col4.metric(
        "Minimo ultimas 24h",
        f"{df_24h['cmg_usd_mwh'].min():.2f} USD/MWh",
    )

    st.markdown("---")

    # ── Serie completa con selector de rango ─────────────────────────────────
    st.subheader("Serie historica del CMG")

    fig_serie = px.line(
        df_cmg,
        x="datetime",
        y="cmg_usd_mwh",
        labels={"datetime": "Fecha", "cmg_usd_mwh": "CMG (USD/MWh)"},
        color_discrete_sequence=[COLOR_XGBOOST],
    )
    fig_serie.update_traces(line_width=0.8)
    fig_serie.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=7,  label="7d",  step="day",  stepmode="backward"),
                    dict(count=1,  label="1m",  step="month", stepmode="backward"),
                    dict(count=3,  label="3m",  step="month", stepmode="backward"),
                    dict(count=6,  label="6m",  step="month", stepmode="backward"),
                    dict(count=1,  label="1a",  step="year",  stepmode="backward"),
                    dict(step="all", label="Todo"),
                ]
            ),
            rangeslider=dict(visible=True),
            type="date",
        ),
        height=420,
        margin=dict(l=0, r=0, t=30, b=0),
        yaxis_title="CMG (USD/MWh)",
    )
    st.plotly_chart(fig_serie, use_container_width=True)

    st.markdown("---")

    # ── Heatmap hora x mes ───────────────────────────────────────────────────
    st.subheader("CMG promedio por hora del dia y mes del año")
    st.caption("Color = USD/MWh promedio historico (2019–2024)")

    heatmap_data = (
        df_cmg.groupby(["hora", "mes"])["cmg_usd_mwh"]
        .mean()
        .reset_index()
        .pivot(index="hora", columns="mes", values="cmg_usd_mwh")
        .reindex(index=range(24), columns=range(1, 13))
    )

    meses_es = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

    fig_heat = go.Figure(
        go.Heatmap(
            z=heatmap_data.values,
            x=meses_es,
            y=[f"{h:02d}:00" for h in range(24)],
            colorscale="RdYlGn_r",
            colorbar=dict(title="USD/MWh"),
            hovertemplate="Mes: %{x}<br>Hora: %{y}<br>CMG: %{z:.2f} USD/MWh<extra></extra>",
        )
    )
    fig_heat.update_layout(
        xaxis_title="Mes",
        yaxis_title="Hora del dia (Santiago)",
        yaxis=dict(autorange="reversed"),
        height=500,
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig_heat, use_container_width=True)


# =============================================================================
# PAGINA 2 — PREDICCIONES Y MODELOS
# =============================================================================

elif pagina == "Predicciones y modelos":
    st.title("Comparacion de Modelos Predictivos")
    st.caption("Conjunto de test: 2024-01-01 → 2024-12-31 | horizon_h=1")

    df_pred = load_predictions_dash()

    if df_pred.empty:
        st.warning(
            "No hay predicciones en la tabla `predictions`. "
            "Ejecuta primero los modelos: `python models/sarima.py`, "
            "`python models/xgboost_model.py`, `python models/lstm_model.py`."
        )
        st.stop()

    # ── Tabla de metricas ────────────────────────────────────────────────────
    st.subheader("Metricas sobre conjunto de test")
    metrics_df = _compute_metrics(df_pred)

    st.dataframe(
        metrics_df.style.highlight_min(
            subset=["MAE (USD/MWh)", "RMSE (USD/MWh)", "MAPE (%)"],
            color="#d4edda",
            axis=0,
        ).highlight_max(
            subset=["R²"],
            color="#d4edda",
            axis=0,
        ).format({
            "MAE (USD/MWh)": "{:.2f}",
            "RMSE (USD/MWh)": "{:.2f}",
            "MAPE (%)": "{:.2f}%",
            "R²": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    # ── Selector de modelos ──────────────────────────────────────────────────
    st.subheader("Predicciones vs Real — ultimos 30 dias del test")

    modelos_disponibles = sorted(df_pred["model_name"].unique().tolist())
    col_checks = st.columns(len(modelos_disponibles))
    modelos_seleccionados: list[str] = []
    for col, modelo in zip(col_checks, modelos_disponibles):
        if col.checkbox(modelo, value=True, key=f"chk_{modelo}"):
            modelos_seleccionados.append(modelo)

    if not modelos_seleccionados:
        st.info("Selecciona al menos un modelo para visualizar.")
        st.stop()

    # Ultimos 30 dias del test
    fecha_max = df_pred["datetime"].max()
    fecha_min = fecha_max - pd.Timedelta(days=30)
    df_30 = df_pred[df_pred["datetime"] >= fecha_min].copy()

    # Construir figura con serie real + modelos seleccionados
    fig_pred = go.Figure()

    # Serie real (una sola vez — igual para todos los modelos)
    grp_ref = df_30[df_30["model_name"] == modelos_seleccionados[0]].sort_values("datetime")
    fig_pred.add_trace(go.Scatter(
        x=grp_ref["datetime"],
        y=grp_ref["actual_cmg"],
        mode="lines",
        name="Real",
        line=dict(color=COLOR_REAL, width=1.5),
    ))

    for modelo in modelos_seleccionados:
        grp = df_30[df_30["model_name"] == modelo].sort_values("datetime")
        fig_pred.add_trace(go.Scatter(
            x=grp["datetime"],
            y=grp["predicted_cmg"],
            mode="lines",
            name=modelo,
            line=dict(color=COLOR_MODELO.get(modelo, "#999"), width=1.2),
            opacity=0.85,
        ))

    fig_pred.update_layout(
        xaxis_title="Fecha",
        yaxis_title="CMG (USD/MWh)",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    st.markdown("---")

    # ── Error absoluto por hora ───────────────────────────────────────────────
    st.subheader("Error absoluto promedio por hora del dia")
    st.caption("Zona horaria: America/Santiago")

    fig_hora = go.Figure()
    horas = list(range(24))

    for modelo in modelos_seleccionados:
        grp = df_pred[df_pred["model_name"] == modelo].copy()
        grp["abs_error"] = np.abs(grp["predicted_cmg"] - grp["actual_cmg"])
        mae_hora = (
            grp.groupby("hora_santiago")["abs_error"]
            .mean()
            .reindex(horas, fill_value=np.nan)
        )
        fig_hora.add_trace(go.Bar(
            x=horas,
            y=mae_hora.values,
            name=modelo,
            marker_color=COLOR_MODELO.get(modelo, "#999"),
            opacity=0.85,
        ))

    fig_hora.update_layout(
        barmode="group",
        xaxis=dict(title="Hora del dia (Santiago)", tickmode="linear", dtick=2),
        yaxis_title="MAE (USD/MWh)",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_hora, use_container_width=True)


# =============================================================================
# PAGINA 3 — ANALISIS DE DRIVERS
# =============================================================================

elif pagina == "Analisis de drivers":
    st.title("Variables que explican el CMG")
    st.caption(
        "Relacion entre el costo marginal y las principales fuentes de generacion. "
        "Datos: 2021–2024, barra Quillota 220kV."
    )

    df_cmg = load_cmg()
    df_gen = load_generation()

    if df_cmg.empty or df_gen.empty:
        st.warning(
            "No hay datos suficientes. Verifica que las tablas `marginal_costs` "
            "y `generation_by_tech` esten cargadas."
        )
        st.stop()

    # Join por datetime UTC
    df_join = pd.merge(
        df_cmg[["datetime", "cmg_usd_mwh"]],
        df_gen[["datetime", "gen_solar_mw", "gen_gas_mw"]],
        on="datetime",
        how="inner",
    ).dropna()

    # Limitar a 20.000 puntos para rendimiento del scatter
    if len(df_join) > 20_000:
        df_join = df_join.sample(20_000, random_state=42).sort_values("datetime")

    # ── Scatter CMG vs Solar ─────────────────────────────────────────────────
    st.subheader("CMG vs Generacion Solar")

    fig_solar = px.scatter(
        df_join,
        x="gen_solar_mw",
        y="cmg_usd_mwh",
        labels={
            "gen_solar_mw": "Generacion solar (MW)",
            "cmg_usd_mwh":  "CMG (USD/MWh)",
        },
        color_discrete_sequence=[COLOR_XGBOOST],
        opacity=0.25,
    )
    fig_solar.update_traces(marker_size=3)

    _x_solar = df_join["gen_solar_mw"].values
    _y_solar  = df_join["cmg_usd_mwh"].values
    _coef_solar = np.polyfit(_x_solar, _y_solar, deg=1)
    _x_solar_sorted = np.linspace(_x_solar.min(), _x_solar.max(), 200)
    fig_solar.add_trace(go.Scatter(
        x=_x_solar_sorted,
        y=np.polyval(_coef_solar, _x_solar_sorted),
        mode="lines",
        name="Tendencia lineal",
        line=dict(color="red", width=2),
    ))

    fig_solar.update_layout(height=420, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_solar, use_container_width=True)

    st.markdown(
        """
        **Interpretacion:** A mayor generacion solar, menor CMG.
        La curva de tendencia muestra la relacion inversa entre penetracion fotovoltaica
        y precio marginal — el efecto "merit order" solar.
        En las horas del mediodia (mayor irradiacion), la energia solar desplaza a las
        plantas de gas y carbon mas caras, hundiendo el precio hasta cero o valores negativos.
        Este fenomeno se intensifico desde 2023 con la entrada masiva de parques solares en el SIC.
        """
    )

    st.markdown("---")

    # ── Scatter CMG vs Gas ───────────────────────────────────────────────────
    st.subheader("CMG vs Generacion Gas")

    fig_gas = px.scatter(
        df_join,
        x="gen_gas_mw",
        y="cmg_usd_mwh",
        labels={
            "gen_gas_mw":  "Generacion gas (MW)",
            "cmg_usd_mwh": "CMG (USD/MWh)",
        },
        color_discrete_sequence=[COLOR_LSTM],
        opacity=0.25,
    )
    fig_gas.update_traces(marker_size=3)

    _x_gas = df_join["gen_gas_mw"].values
    _y_gas  = df_join["cmg_usd_mwh"].values
    _coef_gas = np.polyfit(_x_gas, _y_gas, deg=1)
    _x_gas_sorted = np.linspace(_x_gas.min(), _x_gas.max(), 200)
    fig_gas.add_trace(go.Scatter(
        x=_x_gas_sorted,
        y=np.polyval(_coef_gas, _x_gas_sorted),
        mode="lines",
        name="Tendencia lineal",
        line=dict(color="red", width=2),
    ))

    fig_gas.update_layout(height=420, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_gas, use_container_width=True)

    st.markdown(
        """
        **Interpretacion:** La generacion a gas tiene una relacion positiva con el CMG.
        Las plantas de ciclo abierto a gas natural operan como tecnologia marginal en horas
        de alta demanda o baja disponibilidad hidraulica, fijando el precio del mercado.
        Cuando el gas importado es caro (shocks de GNL), el CMG sube en forma proporcional.
        Esta variable es uno de los features mas importantes en el modelo XGBoost.
        """
    )
