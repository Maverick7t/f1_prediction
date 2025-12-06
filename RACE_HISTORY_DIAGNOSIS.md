# Race History Feature - Complete Diagnosis Report

## 1. ERRORS IDENTIFIED

### Error #1: CORS Policy Violation
```
Access to fetch at 'https://f1-prediction-neid.onrender.com/api/race-history' 
from origin 'https://fonewinner.vercel.app' has been blocked by CORS policy: 
No 'Access-Control-Allow-Origin' header is present on the requested resource.
```

**Status**: Server-side CORS configured, but endpoint might be returning 502 before headers are sent

### Error #2: 502 Bad Gateway
```
GET https://f1-prediction-neid.onrender.com/api/race-history net::ERR_FAILED 502 (Bad Gateway)
```

**Root Cause**: The endpoint is timing out (30-second Render limit) or crashing

### Error #3: Fetch Failed
```
Error fetching race history: TypeError: Failed to fetch
```

**Symptom**: Network error - combination of CORS + 502

---

## 2. WHAT'S MISSING (Data Pipeline)

### Missing #1: Recent Race Data Storage
**Location**: Supabase database

**What's Missing**:
- ❌ Recent 2025 race results not stored in database
- ❌ No `races` table populated with 2025 races
- ❌ No `qualifying_cache` data being used to build race history

**Expected**:
```sql
SELECT * FROM races WHERE race_year = 2025 ORDER BY event_date DESC LIMIT 5;
-- Should return: 5 recent races with driver, team, qualifying_position, finishing_position, points
```

**Current State**: 
- Table exists but may be empty or only has historical data (2018-2024)
- Scheduler caches qualifying data but doesn't populate race results

---

### Missing #2: Race Results Endpoint
**Location**: `backend/api.py`

**What's Missing**:
- ❌ No endpoint to fetch completed race results from FastF1
- ❌ `get_race_winner_from_fastf1()` function exists but may not be working
- ❌ No method to store race results to database

**Current Implementation** (line 1398-1540):
```python
@app.route("/api/race-history", methods=["GET"])
def race_history():
    # Try to get actual winner from FastF1 race results first
    actual_winner = get_race_winner_from_fastf1(race_year, race_name)
    
    # If not found in FastF1, try training data
    if actual_winner is None:
        actual_winner = "TBA"  # DEFAULT IF NOT FOUND ← PROBLEM!
```

**Problem**: Falls back to "TBA" instead of actually loading race results

---

### Missing #3: Model Prediction Logic Integration
**Location**: `backend/api.py` line 1467+

**What's Missing**:
- ⚠️ Predictions run on qualifying data
- ⚠️ But predictions may not use correct features
- ❌ Model might not have all drivers in 2025 season in training data
- ❌ No validation that qualifying_data matches model's expected input format

**Current Logic**:
```python
# Create DataFrame from qualifying data
qual_df = pd.DataFrame(qualifying_data)
race_key = f"{race_year}_{race_name.replace(' ', '_').replace('/', '_')}"

# Run model prediction
predictions = infer_from_qualifying(qual_df, race_key, race_year, race_name, race_name)
predicted_winner = predictions["winner_prediction"]["driver"]
predicted_confidence = int(predictions["winner_prediction"]["percentage"])
```

**Questions**:
- Does `infer_from_qualifying()` exist? ← CHECK NEEDED
- Does qualifying_data structure match model's expected format? ← CHECK NEEDED
- Are 2025 driver codes consistent with model training? ← CHECK NEEDED

---

### Missing #4: Database Population for Race Results
**Location**: Entire pipeline missing

**What's Missing**:
- ❌ No code to fetch race results from FastF1
- ❌ No code to store race results to `races` table
- ❌ No scheduled job to update race results (scheduler only caches qualifying)
- ❌ No endpoint to query race results from database

**Expected Flow**:
```
1. Qualifying happens → Scheduler caches qualifying ✓
2. Race completes → NO HANDLER
3. Results should be fetched from FastF1 → MISSING
4. Results stored to `races` table → MISSING
5. API serves from database → MISSING
```

---

## 3. DATABASE STATE (VERIFIED)

### Table: `races`
**Status**: ❌ **COMPLETELY EMPTY** (0 rows)

```
Total races: 0
2025 races: 0
Historical data: 0 (2018-2024 training data exists but not in this table)
```

**Problem**: 
- No race results for ANY year
- No finish positions
- No actual race winners
- API falls back to "TBA" for all races

---

### Table: `qualifying_cache`
**Status**: ✅ **HAS DATA** (1 entry)

```
Total entries: 1
Entry: 2025_23_Qatar_Grand_Prix
Cached: 2025-12-06
TTL: 365 days
```

**Note**: This is the data we loaded locally, but NOT being used by race-history endpoint

---

### Table: `drivers`
**Status**: ❌ **COMPLETELY EMPTY** (0 rows)

```
Total drivers: 0
Teams: 0
Active drivers: 0
```

**Problem**: 
- No driver metadata
- No driver codes
- No current team info
- API can't lookup drivers

---

### Table: `predictions`
**Status**: ⚠️ **MINIMAL DATA** (3 entries)

```
Total predictions: 3
Recent predictions: 3 (all-time)
```

**Problem**:
- Very few predictions logged
- API errors may prevent logging
- No recent race history data to log

---

## 4. BROKEN COMPONENTS

### Broken #1: `get_recent_qualifying_from_fastf1()`
**File**: `backend/api.py` line 256

**Status**: Code exists but may have issues:
- ✓ Fetches event schedule
- ✓ Filters for past events
- ❓ May fail if FastF1 API changes format
- ❓ May fail if telemetry endpoint times out (25-30 seconds)

**Test**: Run locally:
```bash
python -c "from api import get_recent_qualifying_from_fastf1; print(get_recent_qualifying_from_fastf1(2025, n=1))"
```

---

### Broken #2: `get_race_winner_from_fastf1()`
**File**: `backend/api.py` (needs location check)

**Status**: May not exist or may be incomplete

**What it should do**:
```python
def get_race_winner_from_fastf1(year, race_name):
    """Load race results from FastF1 and return winner driver code"""
    # 1. Load race session (not qualifying)
    # 2. Get results table
    # 3. Filter for finishing_position == 1
    # 4. Return driver code
```

**Current Issue**: Falls back to "TBA" when not found

---

### Broken #3: `infer_from_qualifying()`
**File**: `backend/api.py` (location unknown)

**Status**: May not exist or may have wrong signature

**What it should do**:
```python
def infer_from_qualifying(qual_df, race_key, race_year, race_name, circuit):
    """Run model prediction on qualifying data"""
    # Return: {"winner_prediction": {"driver": "VER", "percentage": 85}}
```

**Current Issue**: May throw exceptions if called with wrong data format

---

### Broken #4: Missing `infer_from_qualifying` Function
**Status**: ❓ CRITICAL - Need to locate

**Where to search**:
```bash
grep -r "def infer_from_qualifying" backend/
# If empty: FUNCTION DOESN'T EXIST
```

---

### Broken #5: Frontend CORS Handling
**File**: `frontend/src/api.js` line 22

**Current**:
```javascript
const response = await fetch(`${API_BASE_URL}/api/race-history`);
if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
}
```

**Issue**: No special handling for CORS errors or 502s
**Missing**: Retry logic, timeout handling, fallback UI state

---

## 5. DATA FLOW ISSUES

### Current Race History Flow (BROKEN)
```
Frontend Request
  ↓
/api/race-history endpoint (line 1398)
  ↓
get_recent_qualifying_from_fastf1(2025, n=5) ← CAN TIMEOUT (25-30 seconds)
  ↓
[Worker timeout after 30 seconds]
  ↓
502 Bad Gateway
  ↓
CORS headers not sent (response already sent)
  ↓
Frontend gets CORS error + 502 error
```

### Expected Race History Flow (WHAT WE NEED)
```
Frontend Request
  ↓
/api/race-history endpoint
  ↓
Check Supabase for recent races (qualifying_cache + races tables)
  ↓
Get cached qualifying (instant)
  ↓
Get cached race results (instant)
  ↓
Get cached predictions (instant)
  ↓
Return <200ms
```

---

## 6. MISSING DATA SOURCES

### Missing: Cached Race Results
**Should be in**: `races` table or separate `race_results_cache` table

**What we need**:
```json
{
  "race": "Qatar Grand Prix",
  "year": 2025,
  "date": "2025-12-06",
  "finishing_position": 1,
  "winner": "VER",
  "drivers": [
    {"driver": "VER", "position": 1, "points": 25},
    {"driver": "NOR", "position": 2, "points": 18},
    ...
  ]
}
```

---

### Missing: Historical Prediction Accuracy
**Should be in**: `predictions` table

**What we need**:
```sql
SELECT 
  race_name,
  predicted_winner,
  actual_winner,
  predicted_correct,
  confidence
FROM predictions
WHERE race_year = 2025
ORDER BY created_at DESC
LIMIT 5;
```

---

## 7. SCHEDULER GAP

### What Scheduler Does (✓)
- Runs every 6 hours
- Checks for new qualifying sessions
- Loads qualifying telemetry from FastF1
- Caches to `qualifying_cache` table

### What Scheduler Should Do (❌ MISSING)
- Check for COMPLETED races (not just qualifying)
- Load race results from FastF1
- Store to `races` table
- Doesn't happen - no code for this!

---

## 8. SUMMARY OF MISSING PIECES

| Component | Location | Status | Impact |
|-----------|----------|--------|--------|
| CORS Headers | api.py:50-70 | ✓ Configured | Issue: May not send on 502 |
| Endpoint /api/race-history | api.py:1398 | ✓ Exists | 502 timeout |
| get_recent_qualifying_from_fastf1() | api.py:256 | ✓ Exists | Works but CAN TIMEOUT |
| get_race_winner_from_fastf1() | api.py:? | ❓ Location unknown | Falls back to "TBA" |
| infer_from_qualifying() | api.py:? | ❌ MISSING | Predictions fail silently |
| Race results fetching | None | ❌ MISSING | No race results loaded |
| Race results storage | None | ❌ MISSING | No race results saved |
| races table population | None | ❌ MISSING | Table empty for 2025 |
| Scheduler race results job | None | ❌ MISSING | No scheduled update |
| Frontend error handling | api.js:22 | ⚠️ Weak | No retry/fallback |
| Database connection check | api.py:? | ❓ Unknown | May fail silently |

---

## 9. NEXT STEPS TO DIAGNOSE

### Step 1: Check What Functions Exist
```bash
cd c:\Users\moina\Downloads\f1_hackathon\backend
grep -n "def get_race_winner_from_fastf1" api.py
grep -n "def infer_from_qualifying" api.py
```

### Step 2: Check Database State
```bash
# Connect to Supabase and run:
SELECT race_year, COUNT(*) FROM races GROUP BY race_year;
SELECT race_year, COUNT(*) FROM drivers GROUP BY race_year;
SELECT COUNT(*) FROM qualifying_cache;
```

### Step 3: Check Render Logs
```
View: https://dashboard.render.com
Look for: race-history endpoint errors
Watch for: FastF1 timeout messages (>25 seconds)
```

### Step 4: Test Locally
```bash
python -c "
from backend.api import get_recent_qualifying_from_fastf1
result = get_recent_qualifying_from_fastf1(2025, n=1)
print(result)
"
```

---

## 10. ROOT CAUSE ANALYSIS

**Why is /api/race-history returning 502?**

1. **Timeout Issue**: `get_recent_qualifying_from_fastf1()` loads from FastF1 (25-30 seconds)
2. **Render Limit**: 30-second worker timeout
3. **No Cache**: Function doesn't check `qualifying_cache` table first
4. **Missing Functions**: `get_race_winner_from_fastf1()` or `infer_from_qualifying()` may not exist
5. **Database Missing**: `races` table empty or not populated with 2025 data

**Why does frontend show CORS error?**

- 502 is returned BEFORE CORS headers are added
- FastF1 API call times out
- Worker dies
- Flask never sends response with headers
- Browser sees no CORS headers + 502 error

---

## BLOCKERS TO FIXING

1. ❓ Need to verify if `infer_from_qualifying()` exists
2. ❓ Need to check `races` table data for 2025
3. ❓ Need to check if `get_race_winner_from_fastf1()` exists
4. ❓ Need to verify qualifying_cache table has correct data structure
5. ❓ Need to check Render logs for actual error messages

