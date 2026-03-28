-- =============================================================================
-- Lean Pipeline Tables (Qualifying-only inference)
-- Run in Supabase SQL editor.
-- Creates: qualifying_raw, results_raw, features_by_race
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- -----------------------------------------------------------------------------
-- qualifying_raw: append-only-ish raw qualifying rows per race
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS qualifying_raw (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    ingested_at TIMESTAMPTZ DEFAULT NOW(),

    race_key TEXT NOT NULL,
    race_year INTEGER,
    event TEXT,
    circuit TEXT,
    session TEXT DEFAULT 'Q',
    source TEXT,

    driver TEXT NOT NULL,
    team TEXT,
    qualifying_position INTEGER,
    qualifying_lap_time_s DOUBLE PRECISION,

    payload JSONB,
    payload_hash TEXT,

    UNIQUE (race_key, driver)
);

CREATE INDEX IF NOT EXISTS idx_qualifying_raw_race_key ON qualifying_raw(race_key);
CREATE INDEX IF NOT EXISTS idx_qualifying_raw_year ON qualifying_raw(race_year);

-- -----------------------------------------------------------------------------
-- results_raw: append-only-ish raw race result rows per race
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS results_raw (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    ingested_at TIMESTAMPTZ DEFAULT NOW(),

    race_key TEXT NOT NULL,
    race_year INTEGER,
    event TEXT,
    circuit TEXT,
    session TEXT DEFAULT 'R',
    source TEXT,

    driver TEXT NOT NULL,
    team TEXT,
    finishing_position INTEGER,
    points DOUBLE PRECISION,
    status TEXT,

    payload JSONB,
    payload_hash TEXT,

    UNIQUE (race_key, driver)
);

CREATE INDEX IF NOT EXISTS idx_results_raw_race_key ON results_raw(race_key);
CREATE INDEX IF NOT EXISTS idx_results_raw_year ON results_raw(race_year);

-- -----------------------------------------------------------------------------
-- features_by_race: precomputed features used at inference (reproducible)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS features_by_race (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    computed_at TIMESTAMPTZ DEFAULT NOW(),

    race_key TEXT NOT NULL,
    race_year INTEGER,
    event TEXT,
    circuit TEXT,

    feature_version TEXT DEFAULT 'v1',

    driver TEXT NOT NULL,
    team TEXT,
    qualifying_position INTEGER,

    -- Engineered numeric features
    team_perf_score DOUBLE PRECISION,
    elo_rating DOUBLE PRECISION,
    recent_form_avg DOUBLE PRECISION,
    circuit_history_avg DOUBLE PRECISION,
    driver_experience_score DOUBLE PRECISION,

    -- Encoded categories (optional but convenient)
    driver_enc INTEGER,
    team_enc INTEGER,
    circuit_enc INTEGER,

    extras JSONB,

    UNIQUE (race_key, feature_version, driver)
);

CREATE INDEX IF NOT EXISTS idx_features_by_race_race_key ON features_by_race(race_key);
CREATE INDEX IF NOT EXISTS idx_features_by_race_year ON features_by_race(race_year);
