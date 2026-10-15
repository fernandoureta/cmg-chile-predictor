# CLAUDE.md — CMG Chile Predictor

## Contexto del proyecto

Sistema predictivo del costo marginal eléctrico de Chile (USD/MWh).
Comparación de modelos estadísticos y deep learning sobre datos reales del CEN.
Proyecto de portafolio GitHub — Ingeniería en Informática.

**Variable objetivo:** costo marginal real, barra Quillota 220kV
(nombre exacto en BD: "Quillota 220kV"), resolución horaria.
**Horizonte de predicción:** objetivo 24 horas (T+1 a T+24); implementado
hoy solo T+1 (horizon_h=1). Ver "Limitaciones" en README.
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
│   └── raw/
│       ├── cmg_quillota_2019.tsv   ✅ descargado
│       ├── cmg_quillota_2020.tsv   ✅ descargado
│       ├── cmg_quillota_2021.tsv   ✅ descargado
│       ├── cmg_quillota_2022.tsv   ✅ descargado
│       ├── cmg_quillota_2023.tsv   ✅ descargado
│       ├── cmg_quillota_2024.tsv   ✅ descargado
│       └── gen_real_2021-2024.tsv  ✅ descargado
├── db/
│   ├── schema.sql                  ✅ ejecutado
│   ├── feature_store.sql           ✅ ejecutado
│   └── migrations/
├── etl/
│   ├── scrapers/
│   │   ├── cen_marginal.py         ✅ completo y funcionando
│   │   ├── cen_generation.py       ✅ completo y funcionando
│   │   ├── cen_demand.py           ⬜ pendiente (solo docstring, ver nota abajo)
│   │   ├── cen_reservoirs.py       ✅ completo y funcionando
│   │   └── weather.py              ✅ completo y funcionando
│   ├── transform.py                ✅ completo y funcionando
│   ├── load.py                     ✅ completo y funcionando
│   ├── pipeline.py                 ✅ completo y funcionando
│   └── scheduler.py                ⬜ pendiente (solo docstring)
├── features/
│   └── build_features.py           ✅ completo y funcionando
├── models/
│   ├── baselines.py                ✅ persistencia (MAE=11.77) y estacional 24h (MAE=19.49)
│   ├── sarima.py                   ✅ paso a paso (MAE=9.09, R2=0.895)
│   ├── xgboost_model.py            ✅ re-entrenado sin fuga (MAE=10.38, R2=0.872)
│   ├── lstm_model.py               ✅ re-entrenado (MAE=9.95, R2=0.864)
│   ├── evaluate.py                 ✅ filas comunes + mejora vs persistencia
│   └── saved/
├── notebooks/
│   └── 01_eda.ipynb                ✅ completo
├── dashboard/
│   └── app.py                      ✅ completo y funcionando
├── tests/
│   ├── test_etl.py                 ✅ 6 tests pasando
│   ├── test_features.py            ✅ 6 tests pasando
│   └── test_models.py              ✅ 4 tests pasando
├── docs/
│   └── CMG_Chile_Predictor_Manual.docx
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
| 2021-01-01 → 2023-06-30 | Train (21.690 h efectivas, desde 2021-01-08 por el lag 168h) |
| 2023-07-01 → 2023-12-31 | Validación (4.416 h) |
| 2024-01-01 00:00 UTC → 2024-07-15 03:00 UTC | Test (4.707 h, tocar solo al final) |

Los límites del split son en **UTC** (`df.loc[...]` sobre índice UTC). El test
termina en julio 2024 porque es el último dato de CMG en la BD (~6,5 meses,
no el año completo).

### Reglas anti-fuga y de evaluación (fix/metodologia, 2026-10-15)
- Toda feature debe ser conocida antes de la hora objetivo T. Lags/rolling
  del CMG con shift(1); columnas `gen_*` desplazadas 1h (`_shift_generation`).
- La ventana LSTM termina en T-1 (nunca incluye la fila objetivo).
- Todos los modelos son one-step-ahead (horizon_h=1) con el CMG real hasta T-1.
  SARIMA: parámetros fijos, estado actualizado hora a hora con `results.extend()`.
- Todos los modelos se evalúan sobre las mismas filas (`restrict_to_common_rows`
  en evaluate.py) y se comparan contra la persistencia.
- Predicciones con `model_version = PREDICTIONS_VERSION` ("2.0", en config.py).
  Al cambiar metodología, subir la versión: el upsert es DO NOTHING y no
  sobrescribe filas existentes.

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
- Líneas base: persistencia pred(T)=CMG(T-1) y estacional pred(T)=CMG(T-24)
- SARIMA: estacionalidad s=24, selección (p,d,q) por AIC → (2,0,2)×(1,1,1,24),
  estimado sobre los últimos 6 meses del train, evaluado paso a paso
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
| `marginal_costs` | 48.523 | 2019-01 → 2024-07 | ✅ cargada (pipeline verificado, idempotente) |
| `generation_by_tech` | 35.060 | 2021-01 → 2024-12 | ✅ cargada (pipeline verificado, idempotente) |
| `demand` | 0 | — | ⬜ pendiente Selenium |
| `reservoir_levels` | 85 | 2019-01 → 2026-01 | ✅ cargada (pipeline verificado, idempotente) |
| `weather` | 2.192 | 2019-01 → 2024-12 | ✅ cargada (pipeline verificado, idempotente) |
| `predictions` | 13.951 (v1.0) + 23.536 (v2.0) | 5 modelos | ✅ v2.0 = 4.707 filas × 5 modelos (+1 hora DST de SARIMA sin actual); v1.0 es histórico obsoleto |

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

# Entrenar modelos (los scripts no usan flags)
python models/baselines.py
python models/sarima.py
python models/xgboost_model.py
python models/lstm_model.py

# Evaluar
python models/evaluate.py

# Dashboard
streamlit run dashboard/app.py
```

---

## Resultados de modelos

Evaluación v2.0 sobre el test (2024-01-01 00:00 UTC → 2024-07-15 03:00 UTC,
4.707 horas, mismas filas para todos), barra Quillota 220kV, horizon_h=1.

| Modelo         | MAE   | RMSE  | MAPE   | R²    | Mejora vs persistencia |
|----------------|-------|-------|--------|-------|------------------------|
| SARIMA         | 9.09  | 15.35 | 22.69% | 0.895 | +22.8%                 |
| LSTM           | 9.95  | 17.50 | 29.70% | 0.864 | +15.4%                 |
| XGBoost        | 10.38 | 16.99 | 28.16% | 0.872 | +11.8%                 |
| Persistencia   | 11.77 | 23.45 | 32.36% | 0.756 | 0.0%                   |
| Estacional_24h | 19.49 | 34.62 | 37.25% | 0.467 | −65.6%                 |

**Mejor modelo:** SARIMA en todas las métricas.

**Interpretación:**
- Los resultados v1.0 (SARIMA R²=-0.27, XGBoost MAE=10.18) eran inválidos:
  XGBoost usaba la generación de la hora T (fuga) y SARIMA hacía un forecast
  de ~4.700 pasos desfasado 6 meses. El "cambio de régimen solar" NO explicaba
  el R² negativo. Detalle en PROBLEMAS_Y_SOLUCIONES.md (2026-10-15).
- A 1 hora, la persistencia ya da R²=0.76; reportar siempre la mejora sobre ella.
- El MAPE elevado se debe a horas con CMG ≈ 0 (mediodía solar).

---

## Próximos pasos en orden

1. ✅ `cen_reservoirs.py` — explorar formulario CEN y descargar datos
2. ✅ `weather.py` — consumir Open-Meteo API para temperatura/precipitaciones
3. ✅ `etl/transform.py` — limpieza centralizada
4. ✅ `etl/load.py` — función de carga genérica
5. ✅ `etl/pipeline.py` — orquestador completo
6. ✅ `db/feature_store.sql` — ejecutado
7. ✅ `features/build_features.py` — lags, rolling, encoding cíclico
8. ✅ Notebooks EDA (01_eda.ipynb completo)
9. ✅ SARIMA — paso a paso, MAE=9.09, R2=0.895 (mejor modelo)
10. ✅ XGBoost — sin fuga, MAE=10.38, R2=0.872
11. ✅ LSTM — MAE=9.95, R2=0.864
12. ✅ Evaluación comparativa con reportes (`models/evaluate.py`)
13. ✅ Dashboard Streamlit — 3 páginas interactivas con Plotly
14. ✅ README completo con arquitectura, resultados e instrucciones
15. ✅ Tests — 16 tests (test_etl, test_features, test_models)
15b. ✅ fix/metodologia — fuga en generación, SARIMA paso a paso, líneas base,
    filas comunes, README con limitaciones
16. ⬜ `cen_demand.py` con Selenium (fase posterior)
17. ⬜ Despliegue en la nube con Docker
18. ⬜ Temporal Fusion Transformer

---

## Estado final del proyecto

**Fecha:** 2026-10-15 (rama `fix/metodologia`, sin merge a main)

Revisión metodológica completada. 16/16 tests pasando. Tres modelos + dos
líneas base evaluados sobre las mismas 4.707 horas de test con datos reales
del CEN. Predicciones vigentes: model_version 2.0.