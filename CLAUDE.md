# CLAUDE.md — CMG Chile Predictor

## Contexto del proyecto

Sistema predictivo del costo marginal eléctrico de Chile (USD/MWh).
Comparación de modelos estadísticos y deep learning sobre datos reales del CEN.
Proyecto de portafolio GitHub — Ingeniería en Informática.

**Variable objetivo:** costo marginal real, barra Quillota 220kV
(nombre exacto en BD: "Quillota 220kV"), resolución horaria.
**Horizonte de predicción:** 24 horas (T+1 a T+24).
**Fuente principal:** Coordinador Eléctrico Nacional (coordinador.cl).

---

## Stack tecnológico

- **Lenguaje:** Python 3.11 (entorno virtual en `/venv`)
- **Base de datos:** PostgreSQL 18, puerto 5432, BD: `cen_data`, usuario: `postgres`
- **ORM / driver:** SQLAlchemy 2.x + psycopg2-binary
- **ETL:** pandas, numpy, requests, APScheduler
- **Imputación:** LightGBM (gaps > 3h), interpolación lineal (gaps ≤ 3h)
- **Modelos:** statsmodels (SARIMA), XGBoost, TensorFlow/Keras (LSTM)
- **Dashboard:** Streamlit + Plotly
- **Variables de entorno:** archivo `.env` en la raíz (NUNCA commitear)
- **Configuración centralizada:** `config.py` (carga variables desde `.env`)
- **Dependencias:** `requirements.txt`
- **Base de datos local:** PostgreSQL 18 instalado directamente en Windows
  (no Docker por ahora — docker-compose.yml existe para referencia futura)

---

## Estructura del repositorio

```
CMG_CHILE_PREDICTOR/
├── data/
│   ├── raw/
│   │   ├── cmg_quillota_2019.tsv   ✅ descargado
│   │   ├── cmg_quillota_2020.tsv   ✅ descargado
│   │   ├── cmg_quillota_2021.tsv   ✅ descargado
│   │   ├── cmg_quillota_2022.tsv   ✅ descargado
│   │   ├── cmg_quillota_2023.tsv   ✅ descargado
│   │   ├── cmg_quillota_2024.tsv   ✅ descargado
│   │   └── gen_real_2021-2024.tsv  ✅ descargado
│   ├── processed/
│   └── external/
├── db/
│   ├── schema.sql                  ✅ ejecutado
│   ├── feature_store.sql           ⬜ ejecutar cuando haya todos los datos
│   └── migrations/
├── etl/
│   ├── scrapers/
│   │   ├── cen_marginal.py         ✅ completo y funcionando
│   │   ├── cen_generation.py       ✅ completo y funcionando
│   │   ├── cen_demand.py           ⬜ pendiente (ver nota abajo)
│   │   ├── cen_reservoirs.py       ⬜ pendiente
│   │   └── weather.py              ⬜ pendiente
│   ├── transform.py                ⬜ pendiente
│   ├── load.py                     ⬜ pendiente
│   ├── pipeline.py                 ⬜ pendiente
│   └── scheduler.py                ⬜ pendiente
├── features/
│   └── build_features.py           ⬜ pendiente
├── models/
│   ├── sarima.py                   ⬜ pendiente
│   ├── xgboost_model.py            ⬜ pendiente
│   ├── lstm_model.py               ⬜ pendiente
│   ├── evaluate.py                 ⬜ pendiente
│   └── saved/
├── notebooks/
│   ├── 01_eda.ipynb                ⬜ pendiente
│   ├── 02_feature_engineering.ipynb
│   ├── 03_sarima_baseline.ipynb
│   ├── 04_xgboost.ipynb
│   ├── 05_lstm.ipynb
│   └── 06_comparison.ipynb
├── dashboard/
│   ├── app.py                      ⬜ pendiente
│   └── components/
├── tests/
│   ├── test_etl.py                 ⬜ pendiente
│   ├── test_features.py            ⬜ pendiente
│   └── test_models.py              ⬜ pendiente
├── docs/
│   ├── CMG_Chile_Predictor_Manual.docx
│   └── Guia_Estudio_CMG.docx
├── PROBLEMAS_Y_SOLUCIONES.md       ✅ iniciado
├── config.py                       ✅ completo
├── requirements.txt                ✅ completo
├── docker-compose.yml              (referencia futura, no se usa ahora)
├── .env.example
├── .gitignore
├── README.md
└── CLAUDE.md  ← este archivo
```

---

## Decisiones de arquitectura ya tomadas

### Datos y períodos
- **CMG:** datos disponibles desde 2019. Archivo TSV por año, barra Quillota 220kV.
  - Nombre exacto barra en CEN: `"BA S/E QUILLOTA 220KV BP1-1"` (no se usa para filtrar,
    los archivos ya vienen filtrados desde el formulario de descarga)
  - Código mnemotécnico interno CEN: `BA02T0002SE032T0002` (ignorar)
- **Generación:** datos disponibles desde 2021 (antes no existe en el CEN).
  Archivo único `gen_real_2021-2024.tsv` con todos los años.
- **Demanda horaria:** OMITIDA TEMPORALMENTE. El formulario del CEN solo permite
  descarga mes a mes (48 descargas para 4 años). Los datasets de energiaabierta.cl
  son resolución diaria, insuficiente para el modelo. Pendiente implementar con
  Selenium en fase posterior cuando el proyecto esté más avanzado.
- **Embalses:** pendiente explorar formulario CEN
- **Clima:** Open-Meteo API (gratuita, sin clave) — pendiente implementar

### Split cronológico
El período de entrenamiento está limitado por la variable más restrictiva
(generación disponible desde 2021):

| Período | Rol |
|---|---|
| 2021-01-01 → 2023-06-30 | Train |
| 2023-07-01 → 2023-12-31 | Validación |
| 2024-01-01 → 2024-12-31 | Test (tocar solo al final) |

### Zona horaria — lección crítica aprendida
El CEN genera exactamente 24 filas por día sin registrar la hora repetida
del cambio horario de abril. Por eso `ambiguous="infer"` falla (necesita
ver la hora duplicada). La solución correcta para TODOS los scrapers es:

```python
.dt.tz_localize(
    "America/Santiago",
    ambiguous=True,           # toda hora ambigua = DST (UTC-3)
    nonexistent="shift_forward",
)
.dt.tz_convert("UTC")
```

### Formato de archivos CEN
- **CMG (TSV):** decimales con coma europea (`59,8`), hora del 1 al 24,
  separador tab, encoding utf-8
- **Generación (TSV):** formato WIDE con columnas Hora 1..Hora 24,
  decimales con coma europea, encoding utf-8-sig (tiene BOM),
  fechas en formato ISO `YYYY-MM-DD`

### Modelos
- SARIMA: estacionalidad s=24, selección (p,d,q) por AIC
- XGBoost: 1000 estimadores, early stopping sobre validación
- LSTM: 2 capas (128→64), Dropout 0.2, BatchNorm, ventana 168h,
  horizonte 24h, optimizer Adam lr=0.001, loss=MAE

### Métricas de evaluación
MAE, RMSE, MAPE, R². Precaución: MAPE se infla cuando CMG ≈ 0
(mediodía solar). Métrica adicional: hit rate en picos (CMG > P90).

---

## Estado actual de la base de datos

| Tabla | Registros | Período | Estado |
|---|---|---|---|
| `marginal_costs` | 48.523 | 2019-01 → 2024-07 | ✅ cargada |
| `generation_by_tech` | 35.060 | 2021-01 → 2024-12 | ✅ cargada |
| `demand` | 0 | — | ⬜ pendiente Selenium |
| `reservoir_levels` | 0 | — | ⬜ pendiente |
| `weather` | 0 | — | ⬜ pendiente |
| `predictions` | 0 | — | se llena al evaluar modelos |

---

## Schema de base de datos

**Tablas:**
- `marginal_costs` → (datetime, barra, cmg_usd_mwh, is_imputed)
  UNIQUE(datetime, barra)
- `generation_by_tech` → (datetime, gen_solar_mw, gen_wind_mw,
  gen_hydro_reservoir_mw, gen_hydro_runofriver_mw, gen_gas_mw,
  gen_coal_mw, gen_diesel_mw, gen_total_mw) UNIQUE(datetime)
- `demand` → (datetime, demand_mw) UNIQUE(datetime)
- `reservoir_levels` → (date, energy_gwh) UNIQUE(date)
- `weather` → (date, region, temp_max_c, precip_mm) UNIQUE(date, region)
- `predictions` → (datetime, barra, model_name, model_version,
  predicted_cmg, actual_cmg, horizon_h)

**Vista materializada:** `feature_store` — une todas las tablas,
filtrada para barra Quillota. Ejecutar SOLO cuando estén todas las
tablas con datos. Refrescar con:
`REFRESH MATERIALIZED VIEW CONCURRENTLY feature_store;`

---

## Problemas conocidos en datos del CEN

| Problema | Tratamiento |
|---|---|
| Horas faltantes | Interpolar si gap ≤ 3h; LightGBM si gap > 3h |
| Valores negativos | Reemplazar con NaN antes de imputar |
| Spikes (> Q99.9 × 3) | Reemplazar con NaN antes de imputar |
| Cambio de formato CEN jul-2024 | Detector automático de schema en parser |
| Zona horaria mixta | Normalizar todo a America/Santiago → UTC |
| Hora repetida cambio horario | ambiguous=True en tz_localize (ver arriba) |
| BOM en archivos generación | encoding="utf-8-sig" en read_csv |
| Decimales coma europea | .str.replace(",", ".") antes de to_numeric |
| Hora 1-24 en CMG | datetime = fecha + (hora - 1) horas |
| Hora 0-23 en demanda | datetime directo sin ajuste |

---

## Convenciones de código

- Type hints obligatorios en todas las funciones
- Docstrings en todas las funciones públicas (Google style)
- Usar `logging`, nunca `print()`
- Path setup en scrapers: `_ROOT = Path(__file__).resolve().parent.parent.parent`
- Commits en español: `feat:` / `fix:` / `refactor:` / `docs:`
- Upsert siempre con `ON CONFLICT DO NOTHING`
- Tests en `tests/` para cada módulo ETL

---

## Cómo correr el proyecto

```bash
# Activar entorno virtual (Windows PowerShell)
venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Inicializar schema (solo primera vez)
psql -U postgres -h localhost -d cen_data -f db/schema.sql

# Backfill CMG
python etl/scrapers/cen_marginal.py

# Backfill generación
python etl/scrapers/cen_generation.py

# Construir features (cuando estén todos los datos)
python features/build_features.py

# Entrenar modelos
python models/sarima.py --train
python models/xgboost_model.py --train
python models/lstm_model.py --train

# Evaluar
python models/evaluate.py --report

# Dashboard
streamlit run dashboard/app.py
```

---

## Próximos pasos en orden

1. `cen_reservoirs.py` — explorar formulario CEN y descargar datos
2. `weather.py` — consumir Open-Meteo API para temperatura/precipitaciones
3. `etl/transform.py` — limpieza centralizada
4. `etl/load.py` — función de carga genérica
5. `etl/pipeline.py` — orquestador completo
6. `db/feature_store.sql` — ejecutar cuando estén todos los datos
7. `features/build_features.py` — lags, rolling, encoding cíclico
8. Notebooks EDA y modelos
9. Dashboard Streamlit
10. `cen_demand.py` con Selenium (fase posterior)

Escribe el código directamente en el archivo etl/pipeline.py 
usando tus herramientas de edición de archivos. No me lo 
muestres en el chat, escríbelo en el archivo directamente.
Luego confírmame que el archivo fue creado correctamente.