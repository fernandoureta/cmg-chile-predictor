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

Este proyecto implementa un pipeline completo de predicción del CMG horario: descarga y procesa datos abiertos del CEN y Open-Meteo, los almacena en PostgreSQL, construye una feature matrix de 29 variables con ingeniería temporal y energética, y entrena tres modelos de distinta complejidad — SARIMA como línea base estadística, XGBoost con gradient boosting y una red LSTM con ventana de 168 horas — evaluándolos sobre un conjunto de test cronológico estricto (todo el año 2024). Los resultados se exponen en un dashboard interactivo Streamlit con Plotly.

Los datos provienen exclusivamente del **Coordinador Eléctrico Nacional** (coordinador.cl), organismo público regulado que publica datos de operación del sistema eléctrico bajo licencia de datos abiertos. La temperatura diaria se obtiene de la API gratuita Open-Meteo.

---

## Resultados

Evaluación sobre **conjunto de test independiente: 2024-01-01 → 2024-12-31** (8.760 horas, nunca vistas durante entrenamiento ni validación).

| Modelo  | MAE (USD/MWh) | RMSE (USD/MWh) | MAPE    | R²    |
|---------|:-------------:|:--------------:|:-------:|:-----:|
| SARIMA  | 42.62         | 53.52          | 92.13%  | -0.27 |
| **XGBoost** | **10.18** | **15.72**  | **27.27%** | **0.89** |
| LSTM    | 10.18         | 17.15          | 27.43%  | 0.87  |

**XGBoost gana en todas las métricas.**

El resultado más relevante es que **XGBoost empata en MAE con la LSTM** (10.18 USD/MWh) y la supera en RMSE y R². Esto no es sorprendente dado el volumen de datos disponible (~30.000 observaciones de entrenamiento): con datasets de este tamaño, un modelo de árboles bien calibrado con feature engineering de calidad es tan competitivo como una red recurrente, y más interpretable. El feature más importante en XGBoost es `gen_solar_mw`, confirmando que el cambio estructural del mercado (expansión solar masiva desde 2023) es la variable dominante en 2024.

El R² negativo de SARIMA (−0.27) no indica un error de implementación: es el resultado esperado de un modelo univariado sin variables exógenas ante un cambio de régimen. SARIMA aprende la estacionalidad de 2021–2023, pero no puede capturar el efecto solar de 2024 porque no tiene acceso a `gen_solar_mw`. Sirve como línea base que cuantifica el valor del feature engineering.

> El MAPE elevado en los tres modelos se debe a horas con CMG ≈ 0 USD/MWh (mediodía solar en verano), donde el error relativo se dispara aunque el error absoluto sea pequeño. En contexto operacional, el MAE absoluto es la métrica más relevante.

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
│   Encoding cíclico hora/dow/mes · Feriados · Join multitabla   │
│   Output: 30.981 filas × 31 columnas (2021–2024)               │
└────────────────────────────┬────────────────────────────────────┘
                             │
             ┌───────────────┼───────────────┐
             ▼               ▼               ▼
     ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
     │   SARIMA     │ │   XGBoost    │ │     LSTM     │
     │ MAE = 42.62  │ │ MAE = 10.18  │ │ MAE = 10.18  │
     │  R² = -0.27  │ │  R² = 0.89   │ │  R² = 0.87   │
     └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
            └────────────────┼────────────────┘
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
│   ├── sarima.py                   # SARIMA(2,0,2)×(1,1,1,24), grid AIC
│   ├── xgboost_model.py            # XGBoost con early stopping sobre val
│   ├── lstm_model.py               # LSTM 2 capas, ventana 168h, MAE loss
│   ├── evaluate.py                 # Métricas comparativas + graficos + CSV
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
git clone https://github.com/fernandoureta/CMG_chile_predictor.git
cd CMG_chile_predictor
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
python models/sarima.py          # ~5 min
python models/xgboost_model.py   # ~2 min
python models/lstm_model.py      # ~20 min (GPU recomendada)
```

Cada modelo guarda sus predicciones en la tabla `predictions` de PostgreSQL.

### 9. Evaluación comparativa

```bash
python models/evaluate.py
```

Genera `models/saved/reports/metrics_comparison.csv` y tres gráficos PNG.

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

| Período | Rol | Filas |
|---------|-----|-------|
| 2021-01-01 → 2023-06-30 | Train | ~21.000 h |
| 2023-07-01 → 2023-12-31 | Validación | ~4.400 h |
| 2024-01-01 → 2024-12-31 | Test — solo al final | ~8.760 h |

El conjunto de test nunca se toca durante desarrollo ni selección de hiperparámetros. Las features de rolling y lags usan `shift(1)` para garantizar que no haya fuga de información del futuro.

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

29 variables organizadas en cinco grupos:

| Grupo | Variables |
|-------|-----------|
| **Lags del CMG** | lag 1h, 2h, 3h, 6h, 12h, 24h, 168h (1 semana) |
| **Rolling stats** | mean y std con ventana 24h y 168h (sobre valores con shift=1) |
| **Generación** | solar, eólica, hidro embalse, hidro pasada, gas, carbón, diésel, total (MW) |
| **Clima** | temperatura máxima diaria (°C) |
| **Calendario** | hora sin/cos, día semana sin/cos, mes sin/cos, is_weekend, is_holiday |

El encoding cíclico (sin/cos) preserva la continuidad circular de hora 23→0, lunes→domingo, diciembre→enero, evitando el problema de saltos arbitrarios en encoding ordinal.

---

## Trabajo futuro

- **`cen_demand.py` con Selenium** — automatizar la descarga de demanda horaria (48 formularios en el portal CEN) para incorporar esta variable al feature set, actualmente omitida.
- **Temporal Fusion Transformer (TFT)** — modelo de atención diseñado específicamente para series temporales multivariadas con estacionalidades complejas; candidato natural para superar al LSTM en este dataset.
- **Despliegue en la nube** — containerizar con Docker y desplegar en Google Cloud Run o AWS App Runner con scheduler APScheduler para actualización diaria automática.
- **Alertas de precio alto** — notificaciones vía email o Telegram cuando la predicción supere el percentil 90 del CMG histórico, útil para grandes consumidores industriales.
- **Horizonte multi-paso** — extender el modelo de horizon_h=1 a predicciones T+1 → T+24 simultáneas, reemplazando el enfoque recursivo actual.

---

## Licencia

MIT — ver [LICENSE](LICENSE) para detalles.

Los datos del CEN son de acceso público bajo la política de datos abiertos del Coordinador Eléctrico Nacional de Chile.
