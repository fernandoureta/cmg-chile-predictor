-- Schema completo: tablas, indices, vista materializada
-- Proyecto: CMG Chile Predictor
-- Base de datos: cen_data | Motor: PostgreSQL 16
-- Zona horaria: todos los campos TIMESTAMPTZ almacenados en UTC

-- =============================================================================
-- EXTENSION
-- =============================================================================
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;


-- =============================================================================
-- TABLA: marginal_costs
-- Costo marginal real horario por barra del SEN (USD/MWh)
-- Fuente: Coordinador Electrico Nacional
-- =============================================================================
CREATE TABLE IF NOT EXISTS marginal_costs (
    id             BIGSERIAL    PRIMARY KEY,
    datetime       TIMESTAMPTZ  NOT NULL,
    barra          TEXT         NOT NULL,
    cmg_usd_mwh    NUMERIC(10, 4) NOT NULL,
    is_imputed     BOOLEAN      NOT NULL DEFAULT FALSE,
    created_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_marginal_costs_datetime_barra UNIQUE (datetime, barra)
);

CREATE INDEX IF NOT EXISTS idx_marginal_costs_datetime
    ON marginal_costs (datetime DESC);

CREATE INDEX IF NOT EXISTS idx_marginal_costs_barra_datetime
    ON marginal_costs (barra, datetime DESC);


-- =============================================================================
-- TABLA: generation_by_tech
-- Generacion horaria por tecnologia del SEN (MW)
-- Fuente: Coordinador Electrico Nacional
-- =============================================================================
CREATE TABLE IF NOT EXISTS generation_by_tech (
    id                        BIGSERIAL    PRIMARY KEY,
    datetime                  TIMESTAMPTZ  NOT NULL,
    gen_solar_mw              NUMERIC(10, 2),
    gen_wind_mw               NUMERIC(10, 2),
    gen_hydro_reservoir_mw    NUMERIC(10, 2),
    gen_hydro_runofriver_mw   NUMERIC(10, 2),
    gen_gas_mw                NUMERIC(10, 2),
    gen_coal_mw               NUMERIC(10, 2),
    gen_diesel_mw             NUMERIC(10, 2),
    gen_total_mw              NUMERIC(10, 2),
    created_at                TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_generation_by_tech_datetime UNIQUE (datetime)
);

CREATE INDEX IF NOT EXISTS idx_generation_by_tech_datetime
    ON generation_by_tech (datetime DESC);


-- =============================================================================
-- TABLA: demand
-- Demanda real horaria del SEN (MW)
-- Fuente: Coordinador Electrico Nacional
-- =============================================================================
CREATE TABLE IF NOT EXISTS demand (
    id          BIGSERIAL    PRIMARY KEY,
    datetime    TIMESTAMPTZ  NOT NULL,
    demand_mw   NUMERIC(10, 2) NOT NULL,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_demand_datetime UNIQUE (datetime)
);

CREATE INDEX IF NOT EXISTS idx_demand_datetime
    ON demand (datetime DESC);


-- =============================================================================
-- TABLA: reservoir_levels
-- Cota energetica diaria de embalses del SEN (GWh)
-- Fuente: Coordinador Electrico Nacional
-- =============================================================================
CREATE TABLE IF NOT EXISTS reservoir_levels (
    id          BIGSERIAL  PRIMARY KEY,
    date        DATE       NOT NULL,
    energy_gwh  NUMERIC(10, 2) NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_reservoir_levels_date UNIQUE (date)
);

CREATE INDEX IF NOT EXISTS idx_reservoir_levels_date
    ON reservoir_levels (date DESC);


-- =============================================================================
-- TABLA: weather
-- Temperatura maxima diaria y precipitaciones por region
-- Fuente: Open-Meteo API
-- =============================================================================
CREATE TABLE IF NOT EXISTS weather (
    id           BIGSERIAL  PRIMARY KEY,
    date         DATE       NOT NULL,
    region       TEXT       NOT NULL,
    temp_max_c   NUMERIC(5, 2),
    precip_mm    NUMERIC(7, 2),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_weather_date_region UNIQUE (date, region)
);

CREATE INDEX IF NOT EXISTS idx_weather_date
    ON weather (date DESC);

CREATE INDEX IF NOT EXISTS idx_weather_date_region
    ON weather (date DESC, region);


-- =============================================================================
-- TABLA: predictions
-- Predicciones de cada modelo con horizonte T+1 a T+24
-- =============================================================================
CREATE TABLE IF NOT EXISTS predictions (
    id               BIGSERIAL    PRIMARY KEY,
    datetime         TIMESTAMPTZ  NOT NULL,
    barra            TEXT         NOT NULL,
    model_name       TEXT         NOT NULL,
    model_version    TEXT         NOT NULL,
    predicted_cmg    NUMERIC(10, 4) NOT NULL,
    actual_cmg       NUMERIC(10, 4),
    horizon_h        SMALLINT     NOT NULL CHECK (horizon_h BETWEEN 1 AND 24),
    created_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_predictions_datetime_barra_model_horizon
        UNIQUE (datetime, barra, model_name, model_version, horizon_h)
);

CREATE INDEX IF NOT EXISTS idx_predictions_datetime
    ON predictions (datetime DESC);

CREATE INDEX IF NOT EXISTS idx_predictions_model_barra
    ON predictions (model_name, barra, datetime DESC);


-- =============================================================================
-- Total de tablas creadas: 6
-- marginal_costs, generation_by_tech, demand,
-- reservoir_levels, weather, predictions
-- =============================================================================
