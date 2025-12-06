-- =============================================================================
-- F1 Prediction System - Supabase Database Schema
-- Run this in Supabase SQL Editor to create the tables
-- =============================================================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- RACES TABLE (Historical race data)
-- =============================================================================
CREATE TABLE IF NOT EXISTS races (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    race_key VARCHAR(255),
    race_year INTEGER NOT NULL,
    event VARCHAR(255) NOT NULL,
    circuit VARCHAR(255),
    event_date DATE,
    
    -- Driver info
    driver VARCHAR(10) NOT NULL,
    team VARCHAR(100),
    
    -- Qualifying
    qualifying_position INTEGER,
    qualifying_lap_time_s DECIMAL(10, 3),
    
    -- Race result
    finishing_position INTEGER,
    points DECIMAL(5, 2),
    
    -- Calculated features (can be updated)
    elo_rating DECIMAL(10, 2) DEFAULT 1500,
    recent_form_avg DECIMAL(5, 2),
    circuit_history_avg DECIMAL(5, 2),
    team_perf_score DECIMAL(5, 4),
    driver_experience_score DECIMAL(5, 4),
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Indexes for common queries
    UNIQUE(race_key, driver)
);

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_races_year ON races(race_year);
CREATE INDEX IF NOT EXISTS idx_races_driver ON races(driver);
CREATE INDEX IF NOT EXISTS idx_races_event_date ON races(event_date);
CREATE INDEX IF NOT EXISTS idx_races_circuit ON races(circuit);

-- =============================================================================
-- DRIVERS TABLE (Driver metadata)
-- =============================================================================
CREATE TABLE IF NOT EXISTS drivers (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    code VARCHAR(10) UNIQUE NOT NULL,
    full_name VARCHAR(255),
    nationality VARCHAR(100),
    date_of_birth DATE,
    
    -- Current stats (updated periodically)
    current_team VARCHAR(100),
    current_elo DECIMAL(10, 2) DEFAULT 1500,
    total_races INTEGER DEFAULT 0,
    total_wins INTEGER DEFAULT 0,
    total_podiums INTEGER DEFAULT 0,
    
    -- Metadata
    active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- TEAMS TABLE (Constructor/Team metadata)
-- =============================================================================
CREATE TABLE IF NOT EXISTS teams (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    full_name VARCHAR(255),
    country VARCHAR(100),
    
    -- Current stats
    current_season_points DECIMAL(10, 2) DEFAULT 0,
    team_perf_score DECIMAL(5, 4) DEFAULT 0.5,
    
    -- Metadata
    active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- PREDICTIONS TABLE (Prediction logging for accuracy tracking)
-- =============================================================================
CREATE TABLE IF NOT EXISTS predictions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    
    -- Race info
    race VARCHAR(255) NOT NULL,
    race_year INTEGER,
    circuit VARCHAR(255),
    
    -- Prediction
    predicted VARCHAR(10) NOT NULL,
    confidence DECIMAL(5, 2),
    model_version VARCHAR(20) DEFAULT 'v3',
    
    -- Actual result (filled in after race)
    actual VARCHAR(10),
    correct BOOLEAN,
    
    -- Full prediction data (JSON)
    full_predictions JSONB,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for accuracy queries
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_correct ON predictions(correct);

-- =============================================================================
-- QUALIFYING_CACHE TABLE (Cached qualifying telemetry data)
-- =============================================================================
CREATE TABLE IF NOT EXISTS qualifying_cache (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    race_key VARCHAR(255) NOT NULL UNIQUE,
    race_year INTEGER NOT NULL,
    
    -- Qualifying data (JSON array of driver telemetry)
    qualifying_data JSONB NOT NULL,
    
    -- TTL and metadata
    cached_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ DEFAULT (NOW() + INTERVAL '365 days'),
    
    CONSTRAINT valid_expiry CHECK (expires_at > cached_at)
);

-- Index for fast lookups and TTL cleanup
CREATE INDEX IF NOT EXISTS idx_qualifying_cache_race_key ON qualifying_cache(race_key);
CREATE INDEX IF NOT EXISTS idx_qualifying_cache_year ON qualifying_cache(race_year);
CREATE INDEX IF NOT EXISTS idx_qualifying_cache_expires_at ON qualifying_cache(expires_at);

-- =============================================================================
-- CIRCUITS TABLE (Circuit metadata)
-- =============================================================================
CREATE TABLE IF NOT EXISTS circuits (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    country VARCHAR(100),
    city VARCHAR(100),
    
    -- Track characteristics
    track_length_km DECIMAL(5, 3),
    total_laps INTEGER,
    track_type VARCHAR(50), -- 'street', 'permanent', 'hybrid'
    
    -- Metadata
    active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- MODEL_REGISTRY TABLE (ML model tracking)
-- =============================================================================
CREATE TABLE IF NOT EXISTS model_registry (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    version VARCHAR(20) NOT NULL,
    
    -- Model info
    model_type VARCHAR(50) DEFAULT 'xgboost',
    training_data_version VARCHAR(50),
    
    -- Metrics
    accuracy DECIMAL(5, 4),
    f1_score DECIMAL(5, 4),
    precision_score DECIMAL(5, 4),
    recall_score DECIMAL(5, 4),
    
    -- Features used (JSON array)
    features JSONB,
    
    -- Status
    status VARCHAR(20) DEFAULT 'active', -- 'active', 'deprecated', 'testing'
    
    -- Metadata
    registered_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(model_name, version)
);

-- =============================================================================
-- ROW LEVEL SECURITY (Optional - for multi-tenant apps)
-- =============================================================================
-- Enable RLS on tables
-- ALTER TABLE races ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;

-- Allow public read access (for API)
-- CREATE POLICY "Public read access" ON races FOR SELECT USING (true);
-- CREATE POLICY "Public read access" ON predictions FOR SELECT USING (true);

-- =============================================================================
-- FUNCTIONS
-- =============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for updated_at
DROP TRIGGER IF EXISTS races_updated_at ON races;
CREATE TRIGGER races_updated_at
    BEFORE UPDATE ON races
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS drivers_updated_at ON drivers;
CREATE TRIGGER drivers_updated_at
    BEFORE UPDATE ON drivers
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS teams_updated_at ON teams;
CREATE TRIGGER teams_updated_at
    BEFORE UPDATE ON teams
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- =============================================================================
-- VIEWS (Optional - for common queries)
-- =============================================================================

-- Recent predictions with accuracy
CREATE OR REPLACE VIEW recent_predictions AS
SELECT 
    race,
    race_year,
    predicted,
    actual,
    correct,
    confidence,
    model_version,
    timestamp
FROM predictions
ORDER BY timestamp DESC
LIMIT 100;

-- Driver statistics
CREATE OR REPLACE VIEW driver_statistics AS
SELECT 
    driver,
    COUNT(*) as total_races,
    COUNT(CASE WHEN finishing_position = 1 THEN 1 END) as wins,
    COUNT(CASE WHEN finishing_position <= 3 THEN 1 END) as podiums,
    ROUND(AVG(finishing_position)::numeric, 2) as avg_finish,
    ROUND(AVG(qualifying_position)::numeric, 2) as avg_qualifying,
    MAX(race_year) as last_season
FROM races
WHERE finishing_position IS NOT NULL
GROUP BY driver
ORDER BY wins DESC;

-- =============================================================================
-- SAMPLE DATA INSERT (for testing)
-- =============================================================================
-- Uncomment to insert sample data

/*
INSERT INTO drivers (code, full_name, nationality, current_team) VALUES
('VER', 'Max Verstappen', 'Dutch', 'Red Bull'),
('HAM', 'Lewis Hamilton', 'British', 'Ferrari'),
('NOR', 'Lando Norris', 'British', 'McLaren'),
('LEC', 'Charles Leclerc', 'Monegasque', 'Ferrari'),
('SAI', 'Carlos Sainz', 'Spanish', 'Williams'),
('RUS', 'George Russell', 'British', 'Mercedes'),
('PIA', 'Oscar Piastri', 'Australian', 'McLaren'),
('ALO', 'Fernando Alonso', 'Spanish', 'Aston Martin');

INSERT INTO teams (name, full_name, country) VALUES
('Red Bull', 'Oracle Red Bull Racing', 'Austria'),
('Ferrari', 'Scuderia Ferrari', 'Italy'),
('McLaren', 'McLaren F1 Team', 'United Kingdom'),
('Mercedes', 'Mercedes-AMG Petronas F1 Team', 'Germany'),
('Aston Martin', 'Aston Martin Aramco F1 Team', 'United Kingdom'),
('Williams', 'Williams Racing', 'United Kingdom'),
('Alpine', 'BWT Alpine F1 Team', 'France'),
('Haas', 'MoneyGram Haas F1 Team', 'United States'),
('Racing Bulls', 'Visa Cash App RB F1 Team', 'Italy'),
('Kick Sauber', 'Stake F1 Team Kick Sauber', 'Switzerland');
*/

-- =============================================================================
-- ROW LEVEL SECURITY (RLS) POLICIES
-- =============================================================================

-- Enable RLS on qualifying_cache
ALTER TABLE qualifying_cache ENABLE ROW LEVEL SECURITY;

-- Allow anyone to read cached qualifying data (public endpoint)
CREATE POLICY "Allow public read access to qualifying_cache" ON qualifying_cache
    FOR SELECT USING (true);

-- Only allow authenticated users (service role) to write
CREATE POLICY "Allow insert/update/delete for service role" ON qualifying_cache
    FOR ALL USING (true) WITH CHECK (true);

-- =============================================================================
-- GRANT PERMISSIONS (for Supabase)
-- =============================================================================
-- These are typically handled automatically by Supabase

-- GRANT ALL ON ALL TABLES IN SCHEMA public TO authenticated;
-- GRANT ALL ON ALL TABLES IN SCHEMA public TO anon;
-- GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO authenticated;
-- GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO anon;
