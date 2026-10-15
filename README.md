# CMG Chile Predictor

### Sistema predictivo del costo marginal eléctrico de Chile

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-18-336791?logo=postgresql&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-FF6600?logo=xgboost&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)

---

## Descripción

El **Costo Marginal de la Energía (CMG)** es el precio al que el sistema eléctrico valora la última unidad de energía despachada en cada hora. En Chile, el Coordinador Eléctrico Nacional (CEN) lo calcula y publica cada hora para cada barra del Sistema Eléctrico Nacional. Es la señal económica central del mercado: determina contratos de largo plazo, ingresos de generadores, costos de grandes consumidores industriales y mineros, y el valor de las coberturas financieras. La barra **Quillota 220kV** es una de las referencias de mayor liquidez del Sistema Interconectado Central.

Este proyecto implementa un pipeline completo de predicción del CMG horario: descarga y procesa datos abiertos del CEN y Open-Meteo, los almacena en PostgreSQL, construye 29 features con ingeniería temporal y energética, y compara tres modelos de distinta complejidad — SARIMA, XGBoost y una red LSTM con ventana de 168 horas — contra dos líneas base ingenuas (persistencia y estacional 24h). Todos predicen la hora siguiente (T+1) y se evalúan sobre exactamente las mismas 4.707 horas de un test cronológico estricto (enero–julio 2024). Los resultados se exponen en un dashboard interactivo Streamlit con Plotly.

Los datos provienen exclusivamente del **Coordinador Eléctrico Nacional** (coordinador.cl), organismo público regulado que publica datos de operación del sistema eléctrico bajo licencia de datos abiertos. La temperatura diaria se obtiene de la API gratuita Open-Meteo.

---

## Resultados

Evaluación sobre el **conjunto de test independiente: 2024-01-01 00:00 UTC → 2024-07-15 03:00 UTC** (31-dic-2023 21:00 → 14-jul-2024 23:00 hora de Chile; **4.707 horas, ≈ 6,5 meses**, hasta el último dato de CMG disponible). Ningún dato de este periodo se usó para entrenar ni para seleccionar hiperparámetros. Todos los modelos predicen **una hora hacia adelante** y se evalúan sobre **exactamente las mismas filas**.

| Modelo | MAE (USD/MWh) | RMSE (USD/MWh) | MAPE | R² | Mejora vs persistencia |
|--------|:-------------:|:--------------:|:----:|:--:|:----------------------:|
| **SARIMA** | **9.09** | **15.35** | **22.69%** | **0.895** | **+22.8%** |
| LSTM | 9.95 | 17.50 | 29.70% | 0.864 | +15.4% |
| XGBoost | 10.38 | 16.99 | 28.16% | 0.872 | +11.8% |
| *Persistencia* (T-1) | 11.77 | 23.45 | 32.36% | 0.756 | 0.0% |
| *Estacional 24h* (T-24) | 19.49 | 34.62 | 37.25% | 0.467 | −65.6% |

*Mejora vs persistencia = reducción porcentual del MAE respecto de predecir `CMG(T) = CMG(T-1)`.*

**Cómo leer estos números.** Con horizonte de 1 hora, el CMG es muy autocorrelacionado: la persistencia ya logra R² = 0,76. Por eso la métrica que importa es cuánto mejora cada modelo sobre esa línea base, no el R² absoluto.

- **SARIMA es el mejor modelo** (−22,8 % de MAE vs persistencia). Se estima una sola vez sobre train y luego, hora por hora, predice T y actualiza su estado con el CMG real observado (sin re-estimar parámetros). La estructura AR(2)·MA(2) más el componente estacional diario captura muy bien la dinámica de corto plazo.
- **LSTM** (+15,4 %) y **XGBoost** (+11,8 %) superan a la persistencia, pero no a SARIMA. El LSTM tiene menor MAE que XGBoost, pero mayor RMSE (falla más en los picos). En XGBoost, las features más importantes son `lag_cmg_1h` y `lag_cmg_24h`; la generación por tecnología, que para evitar fuga solo está disponible hasta T-1, aporta poco con horizonte de 1 hora.
- **Estacional 24h** (`CMG(T) = CMG(T-24)`) es claramente peor que la persistencia: a 1 hora, lo que pasó hace una hora informa más que lo que pasó ayer a la misma hora.

> El MAPE es alto en todos los modelos porque hay horas con CMG ≈ 0 USD/MWh (mediodía solar), donde el error relativo se dispara aunque el error absoluto sea pequeño. El MAE es la métrica más relevante operacionalmente.

### Correcciones metodológicas (v2.0)

Una versión anterior reportaba SARIMA con R² = −0,27 y XGBoost/LSTM con MAE ≈ 10,2. Esos números venían de tres problemas, ya corregidos:

1. **Fuga de información en generación:** XGBoost usaba la generación real de la misma hora T que predecía, un dato que no se conoce al momento de predecir. Ahora todas las columnas `gen_*` están desplazadas 1 hora.
2. **SARIMA evaluado con otro protocolo:** se hacía un único `forecast()` de ~4.700 pasos que además partía en julio de 2023 (fin del train) y se re-etiquetaba como 2024. Ahora es one-step-ahead con actualización hora por hora, igual que los otros modelos.
3. **Distinto conjunto de filas por modelo:** el LSTM perdía las primeras 168 horas del test. Ahora todos se evalúan sobre las mismas 4.707 horas, y se agregaron líneas base para dimensionar la mejora real.

---

## Stack tecnológico

| Categoría | Tecnología |
|-----------|-----------|
| Lenguaje | Python 3.11 |
| Base de datos | PostgreSQL 18 |
| ORM / driver | SQLAlchemy 2.x + psycopg2-binary |
| Manipulación de datos | pandas, numpy |
| Imputación de gaps | interpolación lineal (≤ 3h), LightGBM (> 3h) |
| Feriados nacionales | `holidays` (Chile) |
| Modelo estadístico | statsmodels — SARIMAX |
| Modelo de árboles | XGBoost 2.x |
| Modelo deep learning | TensorFlow / Keras 2.x |
| Visualización | Plotly |
| Dashboard | Streamlit |
| Clima externo | Open-Meteo API (gratuita, sin autenticación) |

---

## Arquitectura del proyecto

```
┌─────────────────────────────────────────────────────────────────┐
│                         FUENTES DE DATOS                        │
│   CEN (TSV anuales)          Open-Meteo API (REST)              │
│   · CMG horario 2019-2024    · Temperatura diaria 2019-2024     │
│   · Generación por tech.     · Precipitaciones diarias          │
│   · Niveles de embalses                                         │
└────────────┬────────────────────────────┬───────────────────────┘
             │                            │
             ▼                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ETL  (etl/)                                 │
│   Scrapers → transform.py (limpieza) → load.py (upsert)        │
│   Orquestado por pipeline.py  ·  Idempotente (ON CONFLICT)     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PostgreSQL  (cen_data)                        │
│   marginal_costs · generation_by_tech · reservoir_levels        │
│   weather · predictions · [feature_store — vista materializada] │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              Feature Engineering  (features/)                   │
│   Lags 1h–168h · Rolling mean/std 24h & 168h                   │
│   Generación desplazada 1h (solo info hasta T-1)               │
│   Encoding cíclico hora/dow/mes · Feriados · Join multitabla   │
│   Output: 30.981 filas × 31 columnas = target + is_imputed     │
│           + 29 features de modelo (2021–2024)                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
            ┌────────────────┼────────────────┬────────────────┐
            ▼                ▼                ▼                ▼
     ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
     │  Baselines   │ │   SARIMA     │ │   XGBoost    │ │     LSTM     │
     │ persistencia │ │ MAE =  9.09  │ │ MAE = 10.38  │ │ MAE =  9.95  │
     │ estacional24 │ │  R² = 0.895  │ │  R² = 0.872  │ │  R² = 0.864  │
     └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
            └────────────────┼────────────────┴────────────────┘
                             │  predictions (PostgreSQL)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│               Dashboard Streamlit  (dashboard/)                 │
│   Página 1: Histórico CMG + heatmap hora×mes                   │
│   Página 2: Comparación modelos + error por hora               │
│   Página 3: Análisis de drivers (solar, gas)                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Estructura del repositorio

```
CMG_CHILE_PREDICTOR/
├── data/
│   ├── raw/                        # TSV originales descargados del CEN
│   │   ├── cmg_quillota_2019.tsv   # CMG horario barra Quillota
│   │   ├── cmg_quillota_2020.tsv
│   │   ├── cmg_quillota_2021.tsv
│   │   ├── cmg_quillota_2022.tsv
│   │   ├── cmg_quillota_2023.tsv
│   │   ├── cmg_quillota_2024.tsv
│   │   └── gen_real_2021-2024.tsv  # Generación real por tecnología
│   └── processed/
├── db/
│   ├── schema.sql                  # Definición de tablas y constraints
│   └── feature_store.sql           # Vista materializada feature_store
├── etl/
│   ├── scrapers/
│   │   ├── cen_marginal.py         # Parser CMG (TSV CEN, formato WIDE)
│   │   ├── cen_generation.py       # Parser generación por tecnología
│   │   ├── cen_reservoirs.py       # Parser niveles de embalses (CSV)
│   │   └── weather.py              # Descarga temperatura Open-Meteo
│   ├── transform.py                # Limpieza: outliers, NaN, zona horaria
│   ├── load.py                     # Upsert genérico ON CONFLICT DO NOTHING
│   └── pipeline.py                 # Orquestador backfill / incremental
├── features/
│   └── build_features.py           # Feature matrix: lags, rolling, cíclico
├── models/
│   ├── baselines.py                # Persistencia (T-1) y estacional (T-24)
│   ├── sarima.py                   # SARIMA(2,0,2)×(1,1,1,24), grid AIC, paso a paso
│   ├── xgboost_model.py            # XGBoost con early stopping sobre val
│   ├── lstm_model.py               # LSTM 2 capas, ventana 168h, MAE loss
│   ├── evaluate.py                 # Métricas sobre filas comunes + mejora vs persistencia
│   └── saved/
│       └── reports/
│           └── metrics_comparison.csv
├── notebooks/
│   └── 01_eda.ipynb                # EDA completo: distribución, estacionalidad
├── dashboard/
│   └── app.py                      # Dashboard Streamlit — 3 páginas
├── config.py                       # Variables de entorno centralizadas
├── requirements.txt
├── .env.example
├── PROBLEMAS_Y_SOLUCIONES.md       # Registro de bugs y soluciones
└── CLAUDE.md                       # Decisiones técnicas internas
```

---

## Instalación y uso

### Prerrequisitos

- Python 3.11
- PostgreSQL 18 en `localhost:5432`
- Base de datos `cen_data` creada previamente

```sql
-- Crear la base de datos (ejecutar una sola vez como superusuario)
CREATE DATABASE cen_data;
```

### 1. Clonar el repositorio

```bash
git clone https://github.com/fernandoureta/cmg-chile-predictor.git
cd cmg-chile-predictor
```

### 2. Crear y activar entorno virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno

```bash
cp .env.example .env
```

Editar `.env`:

```ini
DB_HOST=localhost
DB_PORT=5432
DB_NAME=cen_data
DB_USER=postgres
DB_PASSWORD=tu_password
```

### 5. Inicializar PostgreSQL

```bash
psql -U postgres -h localhost -d cen_data -f db/schema.sql
psql -U postgres -h localhost -d cen_data -f db/feature_store.sql
```

### 6. Descargar datos del CEN

Los archivos TSV se descargan manualmente desde el portal del CEN (ver sección [Fuentes de datos](#fuentes-de-datos)) y se colocan en `data/raw/`. Luego ejecutar el backfill:

```bash
python etl/pipeline.py --mode backfill
```

Esto carga en orden: CMG → Generación → Embalses → Clima (Open-Meteo se descarga automáticamente vía API).

### 7. Construir la feature matrix

```bash
python features/build_features.py
```

### 8. Entrenar los modelos

```bash
python models/baselines.py       # segundos
python models/sarima.py          # ~9 min (grid AIC + predicción paso a paso)
python models/xgboost_model.py   # ~1 min
python models/lstm_model.py      # ~30-60 min en CPU (GPU recomendada)
```

Cada modelo guarda sus predicciones en la tabla `predictions` de PostgreSQL con `model_version = "2.0"` (constante `PREDICTIONS_VERSION` en `config.py`).

### 9. Evaluación comparativa

```bash
python models/evaluate.py
```

Restringe todos los modelos a las mismas filas de test y genera `models/saved/reports/metrics_comparison.csv` (incluye la columna `mejora_vs_persistencia_pct`) y tres gráficos PNG.

### 10. Lanzar el dashboard

```bash
streamlit run dashboard/app.py
```

Abre el navegador en `http://localhost:8501`.

---

## Fuentes de datos

| Fuente | Dataset | Resolución | Período | Acceso |
|--------|---------|------------|---------|--------|
| [Coordinador Eléctrico Nacional](https://www.coordinador.cl) | Costo marginal horario (CMG) | Horaria | 2019–2024 | Descarga manual desde portal CEN |
| [Coordinador Eléctrico Nacional](https://www.coordinador.cl) | Generación real por tecnología | Horaria | 2021–2024 | Descarga manual desde portal CEN |
| [Coordinador Eléctrico Nacional](https://www.coordinador.cl) | Niveles de embalses | Mensual | 2019–2026 | Descarga manual desde portal CEN |
| [Open-Meteo](https://open-meteo.com) | Temperatura máxima y precipitación | Diaria | 2019–2024 | API REST gratuita (sin clave) |

**Nota sobre demanda horaria:** Los datos de demanda eléctrica horaria del CEN requieren 48 descargas manuales individuales para cubrir 4 años (formulario de descarga mes a mes). Los datasets alternativos de energiaabierta.cl tienen resolución diaria, insuficiente para el modelo horario. La implementación de un scraper automatizado con Selenium está prevista como trabajo futuro.

---

## Decisiones de arquitectura

### Split cronológico estricto — sin data leakage

El período de entrenamiento está limitado por la fuente más restrictiva (generación disponible desde 2021):

| Período (límites en UTC) | Rol | Filas |
|---------|-----|-------|
| 2021-01-08 → 2023-06-30 23:00 | Train | 21.690 h |
| 2023-07-01 00:00 → 2023-12-31 23:00 | Validación | 4.416 h |
| 2024-01-01 00:00 → 2024-07-15 03:00 | Test — solo al final | 4.707 h |

El train empieza el 8 de enero de 2021 porque el lag de 168 h necesita una semana de historia. El test termina en julio de 2024 porque es el último dato de CMG disponible en la base.

El conjunto de test nunca se toca durante desarrollo ni selección de hiperparámetros. Todas las features usan solo información disponible antes de la hora objetivo: lags y rolling con `shift(1)`, y generación por tecnología desplazada 1 hora.

### Zona horaria — normalización total a UTC

Los archivos del CEN tienen exactamente 24 filas por día y omiten la hora repetida del cambio horario de abril (clocks chilenos retroceden una hora). La normalización correcta para todos los scrapers:

```python
series.dt.tz_localize(
    "America/Santiago",
    ambiguous=True,             # hora ambigua = DST activo (UTC-3)
    nonexistent="shift_forward",
).dt.tz_convert("UTC")
```

Toda la base de datos almacena timestamps en UTC. La conversión a zona Santiago se aplica solo en la capa de presentación (dashboard, análisis por hora).

### Imputación de gaps

| Tipo de gap | Método |
|-------------|--------|
| ≤ 3 horas consecutivas | Interpolación lineal |
| > 3 horas consecutivas | LightGBM con features temporales |
| Valores negativos | Reemplazar con NaN antes de imputar |
| Spikes > Q99.9 × 3 | Reemplazar con NaN antes de imputar |

### Feature engineering

La feature matrix tiene 31 columnas: el target (`cmg_usd_mwh`), el flag `is_imputed` (no se usa como feature) y **29 features de modelo**, organizadas en seis grupos:

| Grupo | Variables |
|-------|-----------|
| **Lags del CMG** | lag 1h, 2h, 3h, 6h, 12h, 24h, 168h (1 semana) |
| **Rolling stats** | mean y std con ventana 24h y 168h (sobre valores con shift=1) |
| **Generación** | solar, eólica, hidro embalse, hidro pasada, gas, carbón, diésel, total (MW) — valor de T-1 |
| **Embalses** | energía embalsada (GWh, mensual, forward-fill) |
| **Clima** | temperatura máxima diaria (°C) |
| **Calendario** | hora sin/cos, día semana sin/cos, mes sin/cos, is_weekend, is_holiday |

El encoding cíclico (sin/cos) preserva la continuidad circular de hora 23→0, lunes→domingo, diciembre→enero, evitando el problema de saltos arbitrarios en encoding ordinal.

---

## Limitaciones

- **Horizonte de 1 hora.** Todos los modelos predicen solo T+1 usando el CMG real hasta T-1. Es un problema más fácil que el pronóstico day-ahead (T+1 a T+24), que es el relevante para muchas decisiones de mercado. El objetivo original de 24 horas está pendiente.
- **Sin demanda horaria.** La demanda es uno de los principales drivers del CMG y no está en el feature set (ver [Fuentes de datos](#fuentes-de-datos)).
- **Test de ~6,5 meses.** El test cubre enero–julio 2024 (4.707 horas), no un año completo: no incluye la segunda mitad del año ni un ciclo hidrológico completo, y las métricas podrían cambiar en otra estación.
- **Una sola barra.** Solo se modela Quillota 220kV.
- **Granularidad de variables exógenas.** La temperatura es el máximo diario y los embalses son mensuales; se asignan a todas las horas del día/mes, por lo que en la práctica solo serían conocidos completamente al cierre de ese periodo.
- **SARIMA entrenado sobre 6 meses.** Por costo computacional, sus parámetros se estiman con los últimos 6 meses del train y no se re-estiman durante el test.

---

## Trabajo futuro

- **`cen_demand.py` con Selenium** — automatizar la descarga de demanda horaria (48 formularios en el portal CEN) para incorporar esta variable al feature set, actualmente omitida.
- **Temporal Fusion Transformer (TFT)** — modelo de atención diseñado específicamente para series temporales multivariadas con estacionalidades complejas; candidato natural para superar al LSTM en este dataset.
- **Despliegue en la nube** — containerizar con Docker y desplegar en Google Cloud Run o AWS App Runner con scheduler APScheduler para actualización diaria automática.
- **Alertas de precio alto** — notificaciones vía email o Telegram cuando la predicción supere el percentil 90 del CMG histórico, útil para grandes consumidores industriales.
- **Horizonte multi-paso** — extender de horizon_h=1 a predicciones T+1 → T+24 (estrategia directa o multi-output), comparando de nuevo contra persistencia y estacional 24h.

---

## Licencia

MIT — ver [LICENSE](LICENSE) para detalles.

Los datos del CEN son de acceso público bajo la política de datos abiertos del Coordinador Eléctrico Nacional de Chile.
