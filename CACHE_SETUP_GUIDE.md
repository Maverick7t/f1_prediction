# F1 Prediction Cache Setup Guide

## Overview
This guide walks you through setting up the Supabase cache for F1 qualifying telemetry data to eliminate worker timeouts on Render.

**Problem Solved**: Render's 30-second worker heartbeat was causing 502 timeouts on telemetry endpoints (which take 25-30 seconds to load from FastF1 API)

**Solution**: Cache latest qualifying data in Supabase PostgreSQL for fast retrieval

---

## Step 1: Create the Cache Table in Supabase ✅ PENDING

### What You Need
- Supabase account credentials (you already have these)
- Access to Supabase SQL Editor
- The table definition from `backend/schema.sql`

### Steps

1. **Open Supabase Dashboard**
   - Go to https://app.supabase.com
   - Select your project

2. **Open SQL Editor**
   - Click "SQL Editor" in the left sidebar
   - Click "+ New Query"

3. **Copy and Run the Table Creation Script**

   Copy this entire block from your local `backend/schema.sql` (lines 127-147):

   ```sql
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

   -- RLS Policies
   ALTER TABLE qualifying_cache ENABLE ROW LEVEL SECURITY;

   CREATE POLICY "Allow public read access to qualifying_cache" ON qualifying_cache
       FOR SELECT USING (true);

   CREATE POLICY "Allow insert/update/delete for service role" ON qualifying_cache
       FOR ALL USING (true) WITH CHECK (true);
   ```

4. **Execute the Query**
   - Paste into the SQL editor
   - Click the blue "Run" button (or Ctrl+Enter)
   - You should see: "Query executed successfully"

5. **Verify Table Creation**
   - Run this verification query in SQL Editor:
   ```sql
   SELECT * FROM qualifying_cache;
   ```
   - Should return: "Query returned 0 rows"

---

## Step 2: Populate the Cache with Data

### On Your Local Machine

1. **Run the Cache Population Script**
   ```powershell
   cd c:\Users\moina\Downloads\f1_hackathon
   python backend/scripts/cache_qualifying.py 2025
   ```

2. **Expected Output**
   ```
   Loading latest 2025 qualifying sessions...
   Processing Qatar Grand Prix (2025-11-30)
   - [Data loading and processing...]
   Successfully saved to Supabase: 2025_Qatar_Grand_Prix
   
   Processing Las Vegas Grand Prix (2025-11-23)
   - [Data loading and processing...]
   Successfully saved to Supabase: 2025_Las_Vegas_Grand_Prix
   
   Cache population completed successfully!
   ```

3. **Check for Errors**
   - If you see `APIError` or database errors, verify:
     - Supabase credentials in `.env` file are correct
     - Table was created successfully (verify in Supabase dashboard)
     - Network connection is stable

### What the Script Does
- Loads latest 2025 qualifying sessions from FastF1 API
- Extracts top 6 drivers' telemetry for each session
- Saves to Supabase `qualifying_cache` table
- Automatically handles UTC to local time conversion
- Each entry stored as JSONB for flexibility

---

## Step 3: Verify Data in Supabase

### Check Cached Data in Dashboard

1. **Go to Supabase Dashboard**
   - Navigate to "Table Editor"
   - Select `qualifying_cache` from the left sidebar

2. **Inspect the Data**
   - Should see 1-2 rows (one per recent qualifying)
   - `race_key`: "2025_Qatar_Grand_Prix" (format)
   - `race_year`: 2025
   - `cached_at`: Recent timestamp
   - `expires_at`: One year from now
   - `qualifying_data`: Large JSONB object with driver telemetry

### Run SQL Verification Query

In SQL Editor, run:
```sql
SELECT 
    race_key,
    race_year,
    cached_at,
    expires_at,
    jsonb_array_length(qualifying_data) as driver_count
FROM qualifying_cache
ORDER BY cached_at DESC;
```

**Expected Result:**
```
race_key                          | race_year | cached_at              | expires_at            | driver_count
2025_Qatar_Grand_Prix            | 2025      | 2025-11-30 XX:XX:XX   | 2026-11-30 XX:XX:XX   | 6
```

---

## Step 4: Test the API Endpoints

### Test Locally First

1. **Start Local Backend**
   ```powershell
   cd c:\Users\moina\Downloads\f1_hackathon
   python backend/api.py
   # Should start on http://localhost:5000
   ```

2. **Test Each Endpoint**

   **Endpoint 1: Qualifying Circuit Telemetry**
   ```bash
   curl http://localhost:5000/api/qualifying-circuit-telemetry
   ```
   - Should return cached data instantly (<200ms)
   - Includes all telemetry for all drivers

   **Endpoint 2: Latest Qualifying Session**
   ```bash
   curl http://localhost:5000/api/latest-qualifying-session
   ```
   - Should return top 6 drivers from latest qualifying
   - Returns instantly from cache

   **Endpoint 3: Specific Driver Telemetry**
   ```bash
   curl -X POST http://localhost:5000/api/driver-telemetry \
     -H "Content-Type: application/json" \
     -d '{"driver": "VER"}'
   ```
   - Returns telemetry for specific driver
   - Retrieved from cached data

3. **Expected Response Pattern**
   ```json
   {
     "status": "success",
     "source": "supabase_cache",
     "data": {
       "race_key": "2025_Qatar_Grand_Prix",
       "drivers": [...]
     }
   }
   ```

---

## Step 5: Deploy to Render

### Automatic Deployment

1. **Push Changes to GitHub**
   - Already done! Latest commit: `b73d000`
   - Contains updated `schema.sql` with table definition

2. **Render Auto-Deploys**
   - Render watches your GitHub repository
   - Automatically redeploys when you push changes
   - Check deployment status in Render dashboard

3. **Monitor Logs**
   - Go to Render dashboard → Your service
   - Click "Logs" to see deployment progress
   - Look for any startup errors

### First Production Test

After Render deploys:

1. **Test Production Endpoints**
   ```bash
   # Replace with your actual Render URL
   curl https://your-app-name.onrender.com/api/qualifying-circuit-telemetry
   ```

2. **Expected Behavior**
   - Response time: <200ms (from cache)
   - No 502 Bad Gateway errors
   - Data returns from Supabase cache

3. **Verify in Render Logs**
   - Should see: "Cache hit: 2025_Qatar_Grand_Prix"
   - No FastF1 API loading messages during requests
   - No worker timeout messages

---

## Troubleshooting

### Problem: "Table 'public.qualifying_cache' not found"

**Solution:**
- The table definition wasn't executed in Supabase
- Go back to Step 1 and run the SQL in Supabase SQL Editor
- Verify execution was successful

### Problem: Cache Script Fails with APIError

**Solutions:**
1. Check Supabase credentials in `.env`:
   ```
   SUPABASE_URL=https://xcxyljkomclvmvjmyjxf.supabase.co
   SUPABASE_KEY=your-api-key-here
   ```

2. Verify table exists:
   ```sql
   SELECT table_name FROM information_schema.tables WHERE table_name='qualifying_cache';
   ```

3. Check RLS policies aren't blocking writes:
   ```sql
   SELECT * FROM pg_policies WHERE tablename = 'qualifying_cache';
   ```

### Problem: Endpoints Return "Cache not found for race"

**Solutions:**
1. Run cache population script again
2. Check data in Supabase dashboard
3. Verify script ran without errors
4. Check API logs for the race_key being requested

### Problem: Render Still Shows 502 Errors

**Solutions:**
1. Wait 5 minutes for full deployment
2. Check Render logs for startup errors
3. Verify Supabase credentials are in Render environment variables
4. Check if `DEVELOPMENT_MODE` is set to False in production

---

## How It Works

### Cache Flow

```
User Request to /api/qualifying-circuit-telemetry
    ↓
API checks DEVELOPMENT_MODE environment variable
    ↓
PRODUCTION MODE:
    ├─ Query: "Get latest cached qualifying from Supabase"
    ├─ Supabase returns data instantly (<50ms)
    └─ Response: Cached data
    
DEVELOPMENT MODE:
    ├─ Try Supabase cache first
    ├─ If not found, load fresh from FastF1 API (25-30s)
    └─ Save to Supabase for next request
```

### Cache Population Flow

```
Run: python backend/scripts/cache_qualifying.py 2025
    ↓
Script finds latest completed qualifying session for 2025
    ↓
FastF1 API loads full qualifying telemetry (25-30 seconds)
    ↓
Extract top 6 drivers' telemetry data
    ↓
Save to Supabase (instant)
    ↓
Script exits successfully
```

### On Render Deployment

```
1. API starts on port 5000
2. DEVELOPMENT_MODE = False (production)
3. All telemetry requests use Supabase cache
4. No 25-30 second API calls during worker heartbeat
5. Response times: <200ms
6. No worker timeouts = No 502 errors
```

---

## Environment Variables

### Local Development (`.env` file)

```env
# Supabase
SUPABASE_URL=https://xcxyljkomclvmvjmyjxf.supabase.co
SUPABASE_KEY=your-anon-key
SUPABASE_SERVICE_KEY=your-service-key

# Mode
DEVELOPMENT_MODE=True  # Allows fresh FastF1 loads locally
```

### Render Production

Set these in Render dashboard → Environment:

```
SUPABASE_URL=https://xcxyljkomclvmvjmyjxf.supabase.co
SUPABASE_KEY=your-anon-key
DEVELOPMENT_MODE=False  # Cache-only mode
```

---

## Cache Lifecycle

### TTL (Time To Live)

- **Default**: 365 days from `cached_at`
- **Stored in**: `expires_at` column
- **Automatic cleanup**: Run cleanup task daily (optional)

### Cleanup Query (Optional)

To remove expired entries:
```sql
DELETE FROM qualifying_cache 
WHERE expires_at < NOW();
```

### Manual Cache Update

To force refresh the cache:
```powershell
# This will reload latest qualifying and overwrite existing cache
python backend/scripts/cache_qualifying.py 2025
```

---

## Summary of What Was Changed

### Code Changes (Completed)

1. ✅ **backend/api.py**: Modified 3 endpoints for cache-first strategy
2. ✅ **backend/scripts/cache_qualifying.py**: Updated to Supabase-only
3. ✅ **backend/schema.sql**: Added `qualifying_cache` table definition
4. ✅ **GitHub**: All changes committed and pushed

### Database Setup (Your Action Required)

1. ⏳ **Create table in Supabase**: Run SQL from Step 1
2. ⏳ **Populate with data**: Run script from Step 2
3. ⏳ **Verify data**: Check Supabase dashboard
4. ⏳ **Test endpoints**: Verify API responses
5. ⏳ **Monitor production**: Check Render logs

---

## Next Steps

1. **Right Now**: Execute Step 1 (create table in Supabase)
2. **Then**: Execute Step 2 (run cache script locally)
3. **Then**: Verify in Supabase dashboard (Step 3)
4. **Then**: Test endpoints locally (Step 4)
5. **Finally**: Monitor Render deployment (Step 5)

Once all steps are complete, your F1 prediction API will serve telemetry from cache, eliminating worker timeouts and delivering <200ms response times!
