-- Vista materializada para feature engineering
-- Proyecto: CMG Chile Predictor
-- Une todas las tablas base filtrada para barra = 'Quillota 220 kV'
-- Granularidad: una fila por hora
-- Zona horaria: datetime en UTC (convertir a America/Santiago en Python si se necesita)

-- =============================================================================
-- DROP (si se necesita recrear con nueva estructura)
-- =============================================================================
-- DROP MATERIALIZED VIEW IF EXISTS feature_store;


-- =============================================================================
-- VISTA MATERIALIZADA: feature_store
-- =============================================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS feature_store AS

SELECT
    -- ── Identificacion temporal ──────────────────────────────────────────────
    mc.datetime                                                  AS datetime,
    mc.datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Santiago'
                                                                 AS datetime_santiago,
    DATE(mc.datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Santiago')
                                                                 AS date_local,

    -- ── Variable objetivo ────────────────────────────────────────────────────
    mc.cmg_usd_mwh,
    mc.is_imputed                                                AS cmg_is_imputed,

    -- ── Generacion por tecnologia (MW) ───────────────────────────────────────
    gt.gen_solar_mw,
    gt.gen_wind_mw,
    gt.gen_hydro_reservoir_mw,
    gt.gen_hydro_runofriver_mw,
    gt.gen_gas_mw,
    gt.gen_coal_mw,
    gt.gen_diesel_mw,
    gt.gen_total_mw,

    -- ── Demanda (MW) ─────────────────────────────────────────────────────────
    d.demand_mw,

    -- ── Embalses (GWh) ───────────────────────────────────────────────────────
    rl.energy_gwh                                                AS reservoir_gwh,

    -- ── Variables meteorologicas ─────────────────────────────────────────────
    w.temp_max_c,
    w.precip_mm

FROM marginal_costs mc

LEFT JOIN generation_by_tech gt
    ON gt.datetime = mc.datetime

LEFT JOIN demand d
    ON d.datetime = mc.datetime

LEFT JOIN reservoir_levels rl
    ON rl.date = DATE(mc.datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Santiago')

LEFT JOIN weather w
    ON w.date = DATE(mc.datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Santiago')
   AND w.region = 'Valparaiso'

WHERE mc.barra = 'Quillota 220 kV'

ORDER BY mc.datetime ASC;


-- =============================================================================
-- INDICE sobre la vista materializada (datetime es la clave de consulta)
-- =============================================================================
CREATE UNIQUE INDEX IF NOT EXISTS idx_feature_store_datetime
    ON feature_store (datetime);


-- =============================================================================
-- REFRESCO DE LA VISTA
-- Ejecutar despues de cada carga ETL para mantener la vista actualizada.
-- CONCURRENTLY requiere el indice unico creado arriba y no bloquea lecturas.
-- =============================================================================
-- REFRESH MATERIALIZED VIEW CONCURRENTLY feature_store;
