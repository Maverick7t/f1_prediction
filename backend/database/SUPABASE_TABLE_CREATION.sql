-- ============================================================================
-- F1 QUALIFYING CACHE - TABLE CREATION SCRIPT
-- ============================================================================
-- 
-- HOW TO USE THIS SCRIPT:
-- 1. Go to https://app.supabase.com
-- 2. Select your F1 prediction project
-- 3. Click "SQL Editor" in the left sidebar
-- 4. Click "+ New Query"
-- 5. Copy and paste this ENTIRE script below the line (starting from CREATE TABLE)
-- 6. Click the blue "Run" button or press Ctrl+Enter
-- 7. You should see "Query executed successfully" at the bottom
--
-- If you get any errors, check the error message carefully and refer to 
-- CACHE_SETUP_GUIDE.md Troubleshooting section.
--
-- ============================================================================

-- Create the qualifying_cache table
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

-- Create indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_qualifying_cache_race_key ON qualifying_cache(race_key);
CREATE INDEX IF NOT EXISTS idx_qualifying_cache_year ON qualifying_cache(race_year);
CREATE INDEX IF NOT EXISTS idx_qualifying_cache_expires_at ON qualifying_cache(expires_at);

-- Enable Row Level Security
ALTER TABLE qualifying_cache ENABLE ROW LEVEL SECURITY;

-- Allow public read access (for API endpoints)
CREATE POLICY "Allow public read access to qualifying_cache" ON qualifying_cache
    FOR SELECT USING (true);

-- Allow authenticated users (service role) to write
CREATE POLICY "Allow insert/update/delete for service role" ON qualifying_cache
    FOR ALL USING (true) WITH CHECK (true);

-- ============================================================================
-- VERIFICATION QUERY
-- ============================================================================
--
-- After running the above, execute this query to verify the table was created:
-- 
-- SELECT * FROM qualifying_cache;
--
-- Expected result: "Query returned 0 rows" (table exists but is empty)
-- 
-- You should also see the table appear in the left sidebar under "Tables"
--
-- ============================================================================
