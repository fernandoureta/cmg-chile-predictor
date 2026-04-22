# CMG Chile Predictor

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-18-336791?logo=postgresql&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-MAE%3D10.18-brightgreen)
![License](https://img.shields.io/badge/licencia-MIT-green)

Sistema de predicción del **Costo Marginal de la Energía Eléctrica (CMG)** en Chile, barra Quillota 220kV, con resolución horaria. Compara tres enfoques de modelado — SARIMA, XGBoost y LSTM — sobre datos reales del Coordinador Eléctrico Nacional (CEN).

Proyecto de portafolio — Ingeniería en Informática.

---

## ¿Qué es el CMG y por qué predecirlo?

El **Costo Marginal de la Energía** es el precio al que el sistema eléctrico valora la última unidad de energía producida. En Chile, el CEN lo calcula cada hora para cada barra del Sistema Eléctrico Nacional (SEN) y lo publica en su portal. Es la señal económica fundamental del mercado eléctrico chileno:

- **Generadores** lo usan para optimizar despacho y contratos.
- **Grandes consumidores** (minería, industria) lo monitorizan para reducir costos.
- **Traders y comercializadoras** lo necesitan para valorar portafolios y coberturas.

La barra **Quillota 220kV** es una de las referencias centrales del Sistema Interconectado Central (SIC), históricamente la de mayor liquidez en contratos de largo plazo.

**El desafío:** desde 2023, la masiva entrada de energía solar en Chile genera picos de precio cero al mediodía y spikes extremos en horas de alta demanda. Este cambio de régimen hace que modelos puramente estadísticos (SARIMA) queden obsoletos y requiere modelos capaces de incorporar variables exógenas como la generación por tecnología.

---

## Resultados

Evaluación sobre conjunto de **test independiente: 2024-01-01 → 2024-12-31** (8.760 horas, nunca vistas durante el entrenamiento).

| Modelo  | MAE (USD/MWh) | RMSE (USD/MWh) | MAPE    | R²    |
|---------|:-------------:|:--------------:|:-------:|:-----:|
| SARIMA  | 42.62         | 53.52          | 92.13%  | -0.27 |
| XGBoost | **10.18**     | **15.72**      | **27.27%** | **0.89** |
| LSTM    | 10.18         | 17.15          | 27.43%  | 0.87  |

**Mejor modelo: XGBoost** — gana en todas las métricas.

> El MAPE elevado en los tres modelos se debe a horas con CMG ≈ 0 USD/MWh (mediodía solar), que inflan el error relativo. La métrica más relevante en contexto operacional es el MAE absoluto.

---

## Stack tecnológico

| Capa | Tecnología |
|------|-----------|
| Lenguaje | Python 3.11 |
| Base de datos | PostgreSQL 18 |
| ORM / driver | SQLAlchemy 2.x + psycopg2-binary |
| ETL | pandas, numpy, requests |
| Feature engineering | pandas, numpy, holidays |
| Modelo estadístico | statsmodels (SARIMAX) |
| Modelo de árboles | XGBoost |
| Modelo deep learning | TensorFlow / Keras |
| Visualización | matplotlib, Plotly |
| Dashboard | Streamlit |
| Clima externo | Open-Meteo API (gratuita, sin clave) |

---

## Estructura del repositorio

```
CMG_CHILE_PREDICTOR/
├── data/
│   ├── raw/                        # TSV originales descargados del CEN
│   │   ├── cmg_quillota_2019.tsv
│   │   ├── cmg_quillota_2020.tsv
│   │   ├── cmg_quillota_2021.tsv
│   │   ├── cmg_quillota_2022.tsv
│   │   ├── cmg_quillota_2023.tsv
│   │   ├── cmg_quillota_2024.tsv
│   │   └── gen_real_2021-2024.tsv
│   └── processed/
├── db/
│   ├── schema.sql                  # Definicion de tablas y vistas
│   └── feature_store.sql           # Vista materializada feature_store
├── etl/
│   ├── scrapers/
│   │   ├── cen_marginal.py         # Parser CMG (TSV anual CEN)
│   │   ├── cen_generation.py       # Parser generacion por tecnologia
│   │   ├── cen_reservoirs.py       # Parser niveles de embalses
│   │   └── weather.py              # Descarga Open-Meteo (temperatura)
│   ├── transform.py                # Limpieza centralizada (outliers, NaN)
│   ├── load.py                     # Upsert generico a PostgreSQL
│   └── pipeline.py                 # Orquestador backfill / incremental
├── features/
│   └── build_features.py           # Lags, rolling stats, encoding ciclico
├── models/
│   ├── sarima.py                   # SARIMA(2,0,2)x(1,1,1,24)
│   ├── xgboost_model.py            # XGBoost con early stopping
│   ├── lstm_model.py               # LSTM 2 capas, ventana 168h
│   ├── evaluate.py                 # Comparacion y reportes
│   └── saved/                      # Modelos serializados y graficos
│       └── reports/
│           └── metrics_comparison.csv
├── notebooks/
│   └── 01_eda.ipynb                # Analisis exploratorio completo
├── dashboard/
│   └── app.py                      # Dashboard Streamlit (en desarrollo)
├── config.py                       # Carga variables de entorno
├── requirements.txt
├── .env.example
└── CLAUDE.md                       # Decisiones tecnicas internas
```

---

## Instalación

### Requisitos previos

- Python 3.11
- PostgreSQL 18 corriendo en `localhost:5432`
- Base de datos `cen_data` creada con usuario `postgres`

### Pasos

**1. Clonar el repositorio**

```bash
git clone https://github.com/fernandoureta/CMG_chile_predictor.git
cd CMG_chile_predictor
```

**2. Crear y activar entorno virtual**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python -m venv venv
source venv/bin/activate
```

**3. Instalar dependencias**

```bash
pip install -r requirements.txt
```

**4. Configurar variables de entorno**

```bash
cp .env.example .env
```

Editar `.env` con los datos de conexión a PostgreSQL:

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=cen_data
DB_USER=postgres
DB_PASSWORD=tu_password
```

**5. Inicializar el schema de base de datos**

```bash
psql -U postgres -h localhost -d cen_data -f db/schema.sql
psql -U postgres -h localhost -d cen_data -f db/feature_store.sql
```

---

## Uso

### ETL — Carga de datos históricos

Los archivos TSV del CEN deben descargarse manualmente desde [coordinador.cl](https://www.coordinador.cl) y colocarse en `data/raw/`. El pipeline completo carga las cuatro fuentes de datos:

```bash
python etl/pipeline.py --mode backfill
```

O scraper por scraper:

```bash
python etl/scrapers/cen_marginal.py      # CMG horario 2019-2024
python etl/scrapers/cen_generation.py   # Generacion por tecnologia 2021-2024
python etl/scrapers/cen_reservoirs.py   # Niveles de embalses (mensual)
python etl/scrapers/weather.py          # Temperatura diaria (Open-Meteo)
```

### Feature engineering

```bash
python features/build_features.py
```

Construye la feature matrix de 31 columnas desde PostgreSQL: lags del CMG (1h a 168h), rolling mean/std (24h y 168h), encoding cíclico de hora/día/mes, generación por tecnología y temperatura. Resultado: ~30.981 filas × 31 columnas (2021-2024).

### Entrenamiento de modelos

```bash
python models/sarima.py          # SARIMA — ~5 min
python models/xgboost_model.py   # XGBoost — ~2 min
python models/lstm_model.py      # LSTM — ~20 min (GPU recomendada)
```

Cada modelo guarda sus predicciones en la tabla `predictions` de PostgreSQL e imprime métricas al finalizar.

### Evaluación comparativa

```bash
python models/evaluate.py
```

Genera en `models/saved/reports/`:
- `metrics_comparison.csv` — tabla de métricas por modelo
- `predictions_vs_real.png` — predicciones vs valores reales (30 días)
- `error_distribution.png` — histograma de errores por modelo
- `error_by_hour.png` — MAE por hora del día (efecto solar)

### Dashboard

```bash
streamlit run dashboard/app.py
```

*(En desarrollo)*

---

## Fuentes de datos

| Fuente | Dataset | Resolución | Período |
|--------|---------|------------|---------|
| [Coordinador Eléctrico Nacional](https://www.coordinador.cl) | Costo marginal horario | Horaria | 2019–2024 |
| [Coordinador Eléctrico Nacional](https://www.coordinador.cl) | Generación real por tecnología | Horaria | 2021–2024 |
| [Coordinador Eléctrico Nacional](https://www.coordinador.cl) | Niveles de embalses | Mensual | 2019–2026 |
| [Open-Meteo](https://open-meteo.com) | Temperatura máxima diaria | Diaria | 2019–2024 |

Los datos del CEN son públicos y de libre descarga desde su portal. Open-Meteo es una API gratuita que no requiere clave de autenticación.

---

## Decisiones de arquitectura

### Split cronológico

El período de train está limitado por la fuente más restrictiva (generación disponible desde 2021):

| Período | Rol |
|---------|-----|
| 2021-01-01 → 2023-06-30 | Train (2,5 años) |
| 2023-07-01 → 2023-12-31 | Validación (6 meses) |
| 2024-01-01 → 2024-12-31 | Test — solo al final |

### Zona horaria

Los archivos del CEN tienen exactamente 24 filas por día y omiten la hora repetida del cambio horario de abril. La normalización correcta para todos los scrapers es:

```python
series.dt.tz_localize(
    "America/Santiago",
    ambiguous=True,            # hora ambigua = DST activo (UTC-3)
    nonexistent="shift_forward",
).dt.tz_convert("UTC")
```

Toda la base de datos almacena timestamps en UTC. La conversión a zona Santiago se hace solo en capa de presentación.

### Features del modelo

29 features agrupados en cinco categorías:

| Categoría | Features |
|-----------|---------|
| Lags del CMG | lag 1h, 2h, 3h, 6h, 12h, 24h, 168h |
| Rolling stats | mean/std 24h y 168h (con shift=1 para evitar leakage) |
| Generación | solar, eólica, hidro embalse, hidro pasada, gas, carbón, diésel, total |
| Clima | temperatura máxima diaria |
| Calendario | hora sin/cos, día semana sin/cos, mes sin/cos, is_weekend, is_holiday |

### Modelos

| Modelo | Configuración clave |
|--------|-------------------|
| SARIMA | (2,0,2)×(1,1,1,24), selección por AIC, datos últimos 6 meses de train |
| XGBoost | n_estimators=1000, lr=0.05, max_depth=6, early_stopping=50 sobre val MAE |
| LSTM | ventana=168h, LSTM(128)→Dropout(0.2)→LSTM(64)→Dense(1), Adam lr=0.001 |

---

## Licencia

MIT — ver [LICENSE](LICENSE).
