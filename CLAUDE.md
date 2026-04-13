# CLAUDE.md — CMG Chile Predictor

## Contexto del proyecto

Sistema predictivo del costo marginal eléctrico de Chile (USD/MWh).
Comparación de modelos estadísticos y deep learning sobre datos reales del CEN.
Proyecto de portafolio GitHub — Ingeniería en Informática.

**Variable objetivo:** costo marginal real, barra Quillota 220 kV, resolución horaria.
**Horizonte de predicción:** 24 horas (T+1 a T+24).
**Fuente principal:** Coordinador Eléctrico Nacional (coordinador.cl / portal.api.coordinador.cl).

---

## Stack tecnológico

- **Lenguaje:** Python 3.11 (entorno virtual en `/venv`)
- **Base de datos:** PostgreSQL 16, puerto 5432, BD: `cen_data`, usuario: `postgres`
- **ORM / driver:** SQLAlchemy 2.x + psycopg2-binary
- **ETL:** pandas, numpy, requests, APScheduler
- **Imputación:** LightGBM (gaps > 3h), interpolación lineal (gaps ≤ 3h)
- **Modelos:** statsmodels (SARIMA), XGBoost, TensorFlow/Keras (LSTM)
- **Dashboard:** Streamlit + Plotly
- **Variables de entorno:** archivo `.env` en la raíz (NUNCA commitear)
- **Configuración centralizada:** `config.py` (carga variables desde `.env`)
- **Dependencias:** `requirements.txt`
- **Contenedor BD:** `docker-compose.yml` (imagen postgres:16)

---

## Estructura del repositorio

```
CMG_CHILE_PREDICTOR/
├── data/
│   ├── raw/          # CSVs descargados del CEN, sin modificar
│   ├── processed/    # Datos limpios listos para modelado
│   └── external/     # Temperatura, feriados, precio gas
├── db/
│   ├── schema.sql         # Tablas, índices, restricciones
│   ├── feature_store.sql  # Vista materializada para features
│   └── migrations/
├── etl/
│   ├── scrapers/
│   │   ├── cen_marginal.py    # Costo marginal real CEN
│   │   ├── cen_generation.py  # Generación horaria por tecnología
│   │   ├── cen_reservoirs.py  # Cota diaria de embalses
│   │   ├── cen_demand.py      # Demanda real del SEN
│   │   └── weather.py         # Temperatura/precipitaciones (Open-Meteo)
│   ├── transform.py  # Limpieza, imputación, validación
│   ├── load.py       # Carga en PostgreSQL con upsert
│   ├── pipeline.py   # Orquestador del flujo ETL completo
│   └── scheduler.py  # Ejecución periódica automática
├── features/
│   └── build_features.py  # Lags, rolling stats, encoding cíclico
├── models/
│   ├── sarima.py          # Línea base estadística
│   ├── xgboost_model.py   # Boosting con features ingenierizadas
│   ├── lstm_model.py      # LSTM multivariado (ventana 168h)
│   ├── evaluate.py        # Métricas y comparación de modelos
│   └── saved/             # Modelos serializados (.pkl, .h5)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_sarima_baseline.ipynb
│   ├── 04_xgboost.ipynb
│   ├── 05_lstm.ipynb
│   └── 06_comparison.ipynb
├── dashboard/
│   ├── app.py          # Aplicación Streamlit principal
│   └── components/
├── tests/
│   ├── test_etl.py
│   ├── test_features.py
│   └── test_models.py
├── docs/
│   └── CMG_Chile_Predictor_Manual.docx  # Manual completo del proyecto
├── config.py
├── requirements.txt
├── docker-compose.yml
├── .env.example
├── .gitignore
├── README.md
└── CLAUDE.md  ← este archivo
```

---

## Decisiones de arquitectura ya tomadas

### Datos
- Período de datos: 2018-01-01 en adelante
- Split cronológico estricto (sin fugas):
  - Train: 2018-01-01 → 2023-06-30
  - Validación: 2023-07-01 → 2023-12-31
  - Test: 2024-01-01 → 2024-12-31 (solo para reporte final)
- Zona horaria: todo normalizado a `America/Santiago`, almacenado en UTC en PostgreSQL
- Upsert con `ON CONFLICT DO NOTHING` para evitar duplicados en ejecuciones repetidas

### Features
- Lags del target: 1h, 2h, 3h, 6h, 12h, 24h, 168h (7 días)
- Rolling stats: media y std sobre ventanas de 24h y 168h (con `shift(1)` para evitar leakage)
- Encoding cíclico: seno/coseno de hora (periodo 24), día de semana (7), mes (12)
- Variables exógenas: gen_solar_mw, gen_wind_mw, gen_hydro_reservoir_mw, gen_gas_mw, demand_mw, reservoir_gwh, temp_max_c, is_holiday, is_weekend

### Modelos
- SARIMA: estacionalidad s=24, selección de (p,d,q) por AIC sobre grilla reducida
- XGBoost: 1000 estimadores, early stopping sobre validación, feature importance guardada
- LSTM: 2 capas (128→64 unidades), Dropout 0.2, BatchNormalization, salida Dense(24)
  - Ventana de entrada: 168 horas (7 días)
  - Horizonte de salida: 24 horas
  - Optimizador: Adam lr=0.001, loss=MAE
  - Callbacks: EarlyStopping(patience=15), ReduceLROnPlateau, ModelCheckpoint

### Métricas de evaluación
MAE, RMSE, MAPE, R². Advertencia: MAPE se infla cuando CMG ≈ 0 (mediodía solar).
Métrica adicional: hit rate en picos de precio (CMG > percentil 90).

---

## Schema de base de datos

**Tablas principales:**
- `marginal_costs` → (datetime, barra, cmg_usd_mwh, is_imputed) — UNIQUE(datetime, barra)
- `generation_by_tech` → (datetime, gen_solar_mw, gen_wind_mw, gen_hydro_reservoir_mw, gen_hydro_runofriver_mw, gen_gas_mw, gen_coal_mw, gen_diesel_mw, gen_total_mw)
- `demand` → (datetime, demand_mw)
- `reservoir_levels` → (date, energy_gwh)
- `weather` → (date, region, temp_max_c, precip_mm)
- `predictions` → (datetime, barra, model_name, model_version, predicted_cmg, actual_cmg, horizon_h)

**Vista materializada:** `feature_store` — une todas las tablas con join por datetime/date, filtrada para barra Quillota. Refrescar con `REFRESH MATERIALIZED VIEW CONCURRENTLY feature_store`.

---

## Problemas conocidos en los datos del CEN

| Problema | Tratamiento |
|---|---|
| Horas faltantes | Interpolar si gap ≤ 3h; LightGBM si gap > 3h |
| Valores negativos | Reemplazar con NaN antes de imputar |
| Spikes (> Q99.9 × 3) | Reemplazar con NaN antes de imputar |
| Cambio de formato julio 2024 | Detector automático de schema en el parser |
| Zona horaria mixta | Normalizar todo a America/Santiago, luego UTC |
| Horas duplicadas (cambio horario) | Conservar primer registro, eliminar duplicado |

---

## Convenciones de código

- Type hints obligatorios en todas las funciones
- Docstrings en todas las funciones públicas (formato Google style)
- Nunca usar `print()` en módulos de producción → usar `logging`
- Commits en español: `feat: descripción` / `fix: descripción` / `refactor: descripción`
- Cada módulo ETL debe tener su test correspondiente en `tests/`
- Los notebooks son solo para exploración, el código de producción va en los módulos
- Variables de entorno siempre via `config.py`, nunca hardcodeadas

---

## Cómo correr el proyecto

```bash
# 1. Activar entorno virtual
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Levantar PostgreSQL
docker-compose up -d postgres

# 3. Inicializar schema
psql -U postgres -d cen_data -f db/schema.sql

# 4. Backfill histórico (primera vez, ~30-40 min)
python etl/pipeline.py --mode backfill --start 2018-01-01

# 5. Actualización incremental (uso diario)
python etl/pipeline.py --mode incremental --date $(date +%Y-%m-%d)

# 6. Construir features
python features/build_features.py

# 7. Entrenar modelos
python models/sarima.py --train
python models/xgboost_model.py --train
python models/lstm_model.py --train

# 8. Evaluar
python models/evaluate.py --report

# 9. Dashboard
streamlit run dashboard/app.py
```

---

## Estado actual del proyecto

> **Actualizar esta sección al final de cada sesión de trabajo.**

| Módulo | Estado | Notas |
|---|---|---|
| Estructura de carpetas | ✅ Completa | Todos los archivos vacíos creados |
| docker-compose.yml | ✅ Creado | postgres:16, puerto 5432 |
| db/schema.sql | ⬜ Pendiente | |
| db/feature_store.sql | ⬜ Pendiente | |
| config.py | ⬜ Pendiente | |
| etl/scrapers/cen_marginal.py | ⬜ Pendiente | |
| etl/scrapers/cen_generation.py | ⬜ Pendiente | |
| etl/scrapers/cen_reservoirs.py | ⬜ Pendiente | |
| etl/scrapers/cen_demand.py | ⬜ Pendiente | |
| etl/scrapers/weather.py | ⬜ Pendiente | |
| etl/transform.py | ⬜ Pendiente | |
| etl/load.py | ⬜ Pendiente | |
| etl/pipeline.py | ⬜ Pendiente | |
| features/build_features.py | ⬜ Pendiente | |
| models/sarima.py | ⬜ Pendiente | |
| models/xgboost_model.py | ⬜ Pendiente | |
| models/lstm_model.py | ⬜ Pendiente | |
| models/evaluate.py | ⬜ Pendiente | |
| dashboard/app.py | ⬜ Pendiente | |
| tests/ | ⬜ Pendiente | |
| requirements.txt | ⬜ Pendiente | |

---

## Referencia rápida de fuentes de datos CEN

| Dataset | URL sección CEN | Formato |
|---|---|---|
| Costo marginal real | coordinador.cl → Mercados → Costos Marginales | CSV exportable |
| Generación por tecnología | coordinador.cl → Operación Real → Generación Real | CSV exportable |
| Demanda real | coordinador.cl → Operación Real → Demanda Real | CSV exportable |
| Embalses (cota diaria) | coordinador.cl → Estadísticas → Embalses | CSV exportable |
| API desarrolladores | portal.api.coordinador.cl | JSON REST |

**Variables exógenas externas:**
- Temperatura/precipitaciones: Open-Meteo API (gratuita, sin clave API)
- Feriados Chile: biblioteca `holidays` de Python (`holidays.Chile()`)
- Precio gas natural: FRED API (`fredapi`) — serie PNGASEUUSDM

