"""
F1 Prediction API Server
Flask REST API to serve race predictions to the frontend

VERSION: 2.1.0 (Optimized - Startup preload removed)
DEPLOYED: 2025-12-05
FIXES: Worker timeout issue, memory optimization, code cleanup
"""

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np
import os
import logging
import requests
from urllib.parse import quote_plus
import time
from pathlib import Path
import hashlib
import fastf1
from datetime import datetime
import json

# Import configuration
from config import config, ensure_directories, print_config

from mlflow_manager import (
    register_model, 
    log_prediction, 
    get_model_registry,
    get_prediction_accuracy
)

# Import new feature store for efficient feature retrieval
from feature_store import get_feature_store, FeatureStore

# Import Supabase prediction logger for accuracy tracking
from database_v2 import get_prediction_logger, PredictionLogger, get_qualifying_cache

# Import file-based caching for expensive queries
from file_cache import get_file_cache, CACHE_KEYS, CACHE_TTL

# Import background scheduler for auto-caching qualifying data
from scheduler import start_scheduler, stop_scheduler

# Initialize Flask app
app = Flask(__name__)

# Configure CORS with explicit settings for Vercel frontend
CORS(app, resources={
    r"/api/*": {
        "origins": ["https://fonewinner.vercel.app", "http://localhost:5173", "http://localhost:3000", "*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": False
    }
})

# Add headers to prevent caching issues and ensure CORS
@app.after_request
def add_header(response):
    """Add headers to prevent caching of API responses and ensure CORS"""
    # Always add CORS headers
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    
    if '/api/' in str(request.path):
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response

# Error handlers to ensure CORS headers are sent even on errors
@app.errorhandler(500)
def handle_500(e):
    response = jsonify({"success": False, "error": "Internal server error"})
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response, 500

@app.errorhandler(404)
def handle_404(e):
    response = jsonify({"success": False, "error": "Not found"})
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response, 404

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {e}")
    response = jsonify({"success": False, "error": str(e)})
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response, 500

# Configure logging based on config
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Print configuration on startup (helpful for debugging)
print_config()

# Ensure all required directories exist
ensure_directories()

# =============================================================================
# CONFIGURATION FROM ENVIRONMENT
# =============================================================================
# All paths now come from config (which reads from .env)
HIST_CSV = str(config.DATA_PATH)
META_FILE = str(config.META_FILE)
MODEL_WIN_FILE = str(config.MODEL_WIN_FILE)
MODEL_POD_FILE = str(config.MODEL_POD_FILE)

# Cache directories
CACHE_DIR = config.CACHE_DIR
IMG_CACHE = config.IMG_CACHE_DIR

# External APIs
ERGAST_BASE = config.ERGAST_BASE_URL

# Confidence thresholds from config
CONFIDENCE_THRESHOLDS = config.CONFIDENCE_THRESHOLDS
CONFIDENCE_COLORS = config.CONFIDENCE_COLORS

# F1 Driver Code Mapping (2025 season + fallbacks)
# Maps driver numbers to 3-letter codes used by FastF1
DRIVER_CODE_MAP = {
    '1': 'VER', '4': 'NOR', '16': 'LEC', '55': 'SAI',
    '63': 'RUS', '81': 'PIA', '30': 'BEA', '14': 'ALO',
    '6': 'TSU', '10': 'GAG', '27': 'HUL', '18': 'SIR',
    '31': 'OCO', '87': 'STR', '43': 'BOT', '23': 'ALB',
    '12': 'HAM', '5': 'VET', '22': 'LAT', '44': 'HAM',  # alt codes
    # 2024 drivers (fallback)
    '3': 'RIC', '77': 'BOT', '20': 'MAG', '11': 'PER'
}

# Configure FastF1 cache
fastf1.Cache.enable_cache(str(config.FASTF1_CACHE_DIR))
logger.info(f"FastF1 cache enabled at: {config.FASTF1_CACHE_DIR}")
try:
    meta = joblib.load(META_FILE)
    encoders = meta["encoders"]
    feature_cols = meta["feature_cols"]
    model_win = joblib.load(MODEL_WIN_FILE)["model"]
    model_pod = joblib.load(MODEL_POD_FILE)["model"]
    
    # NEW: Use feature store for efficient feature retrieval
    feature_store = get_feature_store(config)
    logger.info(f"✓ Feature store initialized: {feature_store.health_check()}")
    
    # NEW: Initialize Supabase prediction logger
    prediction_logger = get_prediction_logger(config)
    logger.info(f"✓ Prediction logger initialized (mode: {prediction_logger.mode})")
    
    # Load historical data (still needed for some legacy endpoints)
    # Prefer parquet if available (5.5x smaller, faster reads)
    from pathlib import Path
    parquet_path = Path(HIST_CSV).with_suffix('.parquet')
    if parquet_path.exists():
        hist_data = pd.read_parquet(parquet_path)
        logger.info(f"✓ Loaded historical data from parquet ({parquet_path.stat().st_size / 1024:.1f} KB)")
    else:
        hist_data = pd.read_csv(HIST_CSV)
        logger.info(f"⚠ Using CSV for historical data (consider running build_feature_snapshots.py)")
    logger.info("Models and data loaded successfully")
    
    # Register models in MLflow (production models)
    try:
        register_model(
            model_name="xgb_winner",
            model_version="v3",
            model_path=MODEL_WIN_FILE,
            metrics={
                "accuracy": 0.82,
                "f1_score": 0.79,
                "precision": 0.85,
                "recall": 0.75
            },
            features=feature_cols,
            training_data_version="2018-2024"
        )
        
        register_model(
            model_name="xgb_podium",
            model_version="v2",
            model_path=MODEL_POD_FILE,
            metrics={
                "accuracy": 0.76,
                "f1_score": 0.73,
                "precision": 0.79,
                "recall": 0.68
            },
            features=feature_cols,
            training_data_version="2018-2024"
        )
        logger.info("✓ Models registered in MLflow")
    except Exception as mlflow_err:
        logger.warning(f"MLflow registration warning: {mlflow_err}")
        
except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise

# ---------- helper: simple file cache ----------
def cache_path_for_key(prefix: str, key: str, ext: str):
    """Generate cache path for a given key"""
    h = hashlib.sha1(key.encode()).hexdigest()
    return IMG_CACHE / f"{prefix}_{h}.{ext}"

# ---------- helper: fetch image from Wikipedia (page image) ----------
def fetch_wikipedia_image(entity_name: str, fallback_size=600):
    """
    Returns local path to cached image (JPEG/PNG) or None.
    Uses MediaWiki 'pageimages' & 'original' where available.
    """
    key = entity_name.strip()
    p = cache_path_for_key("wiki", key, "jpg")
    if p.exists():
        return str(p)

    try:
        # 1) search page
        s = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "format": "json",
                "titles": entity_name,
                "prop": "pageimages",
                "piprop": "original",
            },
            timeout=8
        )
        s.raise_for_status()
        pages = s.json().get("query", {}).get("pages", {})
        for pageid, page in pages.items():
            orig = page.get("original", {})
            img_url = orig.get("source")
            if img_url:
                r = requests.get(img_url, timeout=10, stream=True)
                r.raise_for_status()
                with open(p, "wb") as fh:
                    for chunk in r.iter_content(1024*8):
                        fh.write(chunk)
                return str(p)
    except Exception as e:
        logger.debug(f"Wikipedia image fetch failed for {entity_name}: {e}")

    return None

# ---------- FastF1 helpers ----------
def get_recent_qualifying_from_fastf1(year, n=5):
    """
    Fetch last n completed qualifying sessions from FastF1 for a given year
    Returns list of dicts with qualifying data matching training data format
    """
    try:
        logger.info(f"Fetching {n} recent qualifying sessions from FastF1 for year {year}...")
        
        # Get event schedule for the year
        schedule = fastf1.get_event_schedule(year)
        
        # Convert event date to datetime
        schedule['EventDate'] = pd.to_datetime(schedule['EventDate'])
        
        # Filter for events before today
        today = pd.to_datetime(datetime.now().date())
        past_schedule = schedule[schedule['EventDate'] <= today]
        
        # Sort by date descending to get most recent first
        past_schedule = past_schedule.sort_values('EventDate', ascending=False)
        
        # Take top n
        past_schedule = past_schedule.head(n)
        
        results = []
        
        for idx, event in past_schedule.iterrows():
            event_name = event.get('EventName') or event.get('Event') or 'Unknown'
            event_date = event['EventDate']
            
            try:
                logger.info(f"Loading qualifying for {event_name} ({year})...")
                
                # Load qualifying session
                q_session = fastf1.get_session(year, event_name, 'Q')
                q_session.load(telemetry=False, laps=True)
                
                if q_session.results is None or q_session.results.empty:
                    logger.warning(f"No qualifying results for {event_name}")
                    continue
                
                # Extract qualifying data
                qual_df = q_session.results.copy()
                
                # Build qualifying list
                qualifying_list = []
                for idx, driver_result in qual_df.iterrows():
                    driver_number = driver_result.get('DriverNumber')
                    driver_name = f"{driver_result.get('FirstName', '')} {driver_result.get('LastName', '')}".strip()
                    team = driver_result.get('TeamName', 'Unknown')
                    
                    # Use DriverNumber as position (results are already sorted by qualifying order)
                    position = qual_df.index.tolist().index(idx) + 1 if idx in qual_df.index else None
                    
                    # Try GridPosition first, then use index position
                    grid_pos = driver_result.get('GridPosition')
                    if pd.notna(grid_pos) and grid_pos > 0:
                        position = int(grid_pos)
                    else:
                        position = qual_df.index.tolist().index(idx) + 1 if idx in qual_df.index else position
                    
                    # Get fastest qualifying lap time (Q3 > Q2 > Q1)
                    q3_time = driver_result.get('Q3')
                    q2_time = driver_result.get('Q2')
                    q1_time = driver_result.get('Q1')
                    
                    qual_time = None
                    if pd.notna(q3_time):
                        qual_time = q3_time.total_seconds() if hasattr(q3_time, 'total_seconds') else float(q3_time)
                    elif pd.notna(q2_time):
                        qual_time = q2_time.total_seconds() if hasattr(q2_time, 'total_seconds') else float(q2_time)
                    elif pd.notna(q1_time):
                        qual_time = q1_time.total_seconds() if hasattr(q1_time, 'total_seconds') else float(q1_time)
                    
                    # Map driver number to 3-letter code
                    driver_num_str = str(int(driver_number)).zfill(2) if pd.notna(driver_number) else 'UNK'
                    driver_code = DRIVER_CODE_MAP.get(driver_num_str.lstrip('0') or '0', driver_num_str)
                    
                    # If mapping not found, try to extract from driver name
                    if driver_code == driver_num_str and driver_name:
                        # Use first 3 letters of last name
                        last_name = driver_name.split()[-1] if ' ' in driver_name else driver_name
                        driver_code = last_name[:3].upper()
                    
                    qualifying_list.append({
                        'driver': driver_code,
                        'team': team,
                        'qualifying_position': position if position and position > 0 else len(qualifying_list) + 1,
                        'qualifying_lap_time_s': qual_time
                    })
                
                # Sort by qualifying position
                qualifying_list = sorted(qualifying_list, key=lambda x: x['qualifying_position'])
                
                if qualifying_list:
                    results.append({
                        'race_name': event_name,
                        'year': year,
                        'date': event_date.strftime('%Y-%m-%d'),
                        'qualifying_data': qualifying_list
                    })
                    logger.info(f"Got qualifying for {event_name}: {len(qualifying_list)} drivers")
                    for q in qualifying_list[:3]:
                        logger.info(f"    P{q['qualifying_position']}: {q['driver']} ({q['team']})")
                else:
                    logger.warning(f"No valid qualifying data for {event_name}")
                    
            except Exception as e:
                logger.warning(f"Could not load qualifying for {event_name}: {e}")
                continue
        
        # Reverse to get chronological order
        results.reverse()
        
        logger.info(f"Successfully loaded {len(results)} recent qualifying sessions")
        return results
        
    except Exception as e:
        logger.error(f"FastF1 qualifying fetch error: {e}")
        return []

def get_race_winner_from_fastf1(year, race_name):
    """
    Try to fetch the actual race winner from FastF1 if race has been completed
    Returns driver code (e.g., 'VER') or None if race not completed
    """
    
    try:
        logger.info(f"Fetching race result for {race_name} ({year})...")
        
        # Load race session
        race_session = fastf1.get_session(year, race_name, 'R')
        race_session.load(telemetry=False, laps=True)
        
        if race_session.results is None or race_session.results.empty:
            logger.warning(f"No race results found for {race_name} (race may not have completed)")
            return None
        
        # Get the first finisher (race winner)
        results_df = race_session.results
        
        # Filter out DNF (Did Not Finish) - look for those with points or position
        finished = results_df[results_df['Points'] > 0] if 'Points' in results_df.columns else results_df
        
        if finished.empty:
            # If no one has points, just take position 1
            finished = results_df
        
        # Get first place finisher
        winner = finished.iloc[0]
        driver_number = winner.get('DriverNumber')
        driver_name = f"{winner.get('FirstName', '')} {winner.get('LastName', '')}".strip()
        
        # Map driver number to code
        driver_num_str = str(int(driver_number)).zfill(2) if pd.notna(driver_number) else 'UNK'
        driver_code = DRIVER_CODE_MAP.get(driver_num_str.lstrip('0') or '0', driver_num_str)
        
        # If mapping not found, use first 3 letters of last name
        if driver_code == driver_num_str and driver_name:
            last_name = driver_name.split()[-1] if ' ' in driver_name else driver_name
            driver_code = last_name[:3].upper()
        
        logger.info(f"  Race winner found: {driver_code} ({driver_name})")
        return driver_code
        
    except Exception as e:
        logger.debug(f"Could not fetch race result for {race_name}: {e}")
        return None

def get_next_race_from_fastf1(year=None):
    """
    Get the next upcoming race from FastF1 after today
    Returns dict with race info or None if no upcoming races
    """
    try:
        if year is None:
            year = 2025
        
        logger.info(f"Fetching next race from FastF1 for year {year}...")
        
        # Get event schedule
        schedule = fastf1.get_event_schedule(year)
        
        # Convert event date to datetime
        schedule['EventDate'] = pd.to_datetime(schedule['EventDate'])
        
        # Get today
        today = pd.to_datetime(datetime.now().date())
        
        # Filter for events AFTER today (future races)
        future_schedule = schedule[schedule['EventDate'] > today]
        
        if future_schedule.empty:
            logger.info(f"No upcoming races found for {year}")
            return None
        
        # Get the first (nearest) upcoming race
        next_event = future_schedule.sort_values('EventDate').iloc[0]
        
        event_name = next_event.get('EventName') or next_event.get('Event') or 'Unknown'
        event_date = next_event['EventDate']
        
        logger.info(f"Next race: {event_name} on {event_date.strftime('%Y-%m-%d')}")
        
        return {
            "success": True,
            "event_name": event_name,
            "date": event_date.strftime('%Y-%m-%d'),
            "circuit": event_name,
            "year": year,
            "time": None
        }
        
    except Exception as e:
        logger.error(f"FastF1 next race fetch error: {e}")
        return None

# ---------- Ergast helpers ----------
def ergast_next_race():
    """Fetch next race from Ergast API"""
    try:
        r = requests.get(f"{ERGAST_BASE}/current/next.json", timeout=6)
        r.raise_for_status()
        races = r.json().get("MRData", {}).get("RaceTable", {}).get("Races", [])
        if not races:
            return None
        race = races[0]
        # map to useful fields
        return {
            "race_key": f"{race.get('season')}_{race.get('round')}_{race.get('raceName')}".replace(" ", "_"),
            "race_year": int(race.get("season")),
            "event": race.get("raceName"),
            "circuit": race.get("Circuit", {}).get("circuitName"),
            "country": race.get("Circuit", {}).get("Location", {}).get("country"),
            "date": race.get("date"),
            "time": race.get("time", None)
        }
    except Exception as e:
        logger.warning(f"Ergast next race fetch failed: {e}")
        return None

def ergast_standings(season="current"):
    """Fetch driver and constructor standings from Ergast"""
    try:
        r = requests.get(f"{ERGAST_BASE}/{season}/driverStandings.json", timeout=6)
        r.raise_for_status()
        driv = r.json().get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
        drivers = []
        if driv:
            st = driv[0].get("DriverStandings", [])
            for d in st:
                driver = d.get("Driver", {})
                constructors = d.get("Constructors", [])
                drivers.append({
                    "position": int(d.get("position")),
                    "points": float(d.get("points")),
                    "wins": int(d.get("wins", 0)),
                    "driver_name": f"{driver.get('givenName','')} {driver.get('familyName','')}".strip(),
                    "driver_code": driver.get("code") or driver.get("driverId"),
                    "constructor": constructors[0].get("name") if constructors else None
                })
        # constructors
        r2 = requests.get(f"{ERGAST_BASE}/{season}/constructorStandings.json", timeout=6)
        r2.raise_for_status()
        cons = []
        cs = r2.json().get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
        if cs:
            for c in cs[0].get("ConstructorStandings", []):
                cons.append({
                    "position": int(c.get("position")),
                    "constructor": c.get("Constructor", {}).get("name"),
                    "points": float(c.get("points")),
                    "wins": int(c.get("wins", 0))
                })
        return {"drivers": drivers, "constructors": cons}
    except Exception as e:
        logger.warning(f"Ergast standings fetch failed: {e}")
        return {"drivers": [], "constructors": []}

# -------------------------
# Qualifying fetch helpers
# -------------------------
def get_qual_from_history(race_key=None, race_year=None, circuit=None, event=None):
    """
    Try to pull qualifying rows from hist_data already loaded.
    Returns DataFrame or empty DataFrame.
    Matching strategy:
      1) exact race_key if provided
      2) race_year + circuit match
      3) race_year + event substring match
    """
    df = hist_data.copy()
    if race_key:
        q = df[(df.get("race_key") == race_key) & df["qualifying_position"].notna()]
        if not q.empty:
            return q.sort_values("qualifying_position")
    if race_year and circuit:
        q = df[(df["race_year"] == int(race_year)) & (df.get("circuit", "").astype(str).str.lower() == str(circuit).lower()) & df["qualifying_position"].notna()]
        if not q.empty:
            return q.sort_values("qualifying_position")
    if race_year and event:
        q = df[(df["race_year"] == int(race_year)) & (df.get("event", "").astype(str).str.lower().str.contains(str(event).lower())) & df["qualifying_position"].notna()]
        if not q.empty:
            return q.sort_values("qualifying_position")
    # nothing found
    return pd.DataFrame(columns=df.columns)

def fetch_qualifying_from_ergast(race_year: int, circuit: str = None, event: str = None):
    """
    Query Ergast API to get qualifying for a race in a year.
    Strategy:
      1) Get season schedule: /api/f1/{year}.json and match by circuit/event to find round
      2) If round found, call /api/f1/{year}/{round}/qualifying.json
    Returns list of dicts: [{'driver':'NOR','team':'McLaren','qualifying_position':1, 'qualifying_lap_time_s': 86.123}, ...]
    """
    base = "http://ergast.com/api/f1"
    try:
        # 1) fetch season calendar
        sched_url = f"{base}/{int(race_year)}.json"
        r = requests.get(sched_url, timeout=8)
        r.raise_for_status()
        cal = r.json().get("MRData", {}).get("RaceTable", {}).get("Races", [])
        # try to find round by circuit or event
        found_round = None
        for race in cal:
            race_name = race.get("raceName", "")
            circuit_name = race.get("Circuit", {}).get("circuitName", "")
            if circuit and circuit.lower() in circuit_name.lower():
                found_round = race.get("round")
                break
            if event and event.lower() in race_name.lower():
                found_round = race.get("round")
                break
        if not found_round and len(cal) == 1:
            found_round = cal[0].get("round")

        if not found_round:
            # fallback: return empty list
            return []

        # 2) fetch qualifying for that round
        qurl = f"{base}/{race_year}/{found_round}/qualifying.json"
        rq = requests.get(qurl, timeout=8)
        rq.raise_for_status()
        qdata = rq.json().get("MRData", {}).get("RaceTable", {}).get("Races", [])
        if not qdata:
            return []

        qualifiers = []
        for qual in qdata[0].get("QualifyingResults", []):
            driver = qual.get("Driver", {})
            constructor = qual.get("Constructor", {})
            pos = qual.get("position")
            # Ergast returns multiple Q sessions with times; take fastest Q1/Q2/Q3 if present
            # QualifyingResults has 'Q1', 'Q2', 'Q3' keys sometimes — choose fastest available time string
            time_s = None
            # pick any 'Q' field that's present
            for k in ["Q3", "Q2", "Q1"]:
                t = qual.get(k)
                if t:
                    # Ergast format is mm:ss.sss or sss.sss; convert to seconds if mm:ss present
                    try:
                        if ":" in t:
                            mm, ss = t.split(":")
                            time_s = float(mm) * 60 + float(ss)
                        else:
                            time_s = float(t)
                    except Exception:
                        time_s = None
                    break
            qualifiers.append({
                "driver": driver.get("code") or driver.get("driverId") or f"{driver.get('familyName')}",
                "team": constructor.get("name"),
                "qualifying_position": int(pos) if pos is not None else None,
                "qualifying_lap_time_s": time_s
            })
        # sort by position
        qualifiers = sorted([q for q in qualifiers if q.get("qualifying_position") is not None], key=lambda x: x["qualifying_position"])
        return qualifiers
    except Exception as e:
        logger.warning(f"Ergast fetch failed: {e}")
        return []


def get_confidence_level(percentage):
    """
    Map percentage to confidence level label and color
    Returns: {"level": "VERY HIGH"|"HIGH"|"MODERATE"|"LOW", "color": "#hex_color", "thresholds": {...}}
    """
    if percentage >= CONFIDENCE_THRESHOLDS["very_high"]:
        level = "VERY HIGH"
    elif percentage >= CONFIDENCE_THRESHOLDS["high"]:
        level = "HIGH"
    elif percentage >= CONFIDENCE_THRESHOLDS["moderate"]:
        level = "MODERATE"
    else:
        level = "LOW"
    
    return {
        "level": level,
        "color": CONFIDENCE_COLORS.get(level.lower().replace(" ", "_"), "#ef4444"),
        "thresholds": CONFIDENCE_THRESHOLDS
    }


def infer_from_qualifying(qual_df, race_key, race_year, event, circuit):
    """
    Generate predictions from qualifying data using pre-computed features.
    
    OPTIMIZED: Uses FeatureStore for instant feature lookup instead of
    computing from 7 years of CSV data on every request.
    
    Required model features:
    - qualifying_position (from qualifying data)
    - TeamPerfScore (from team snapshot)
    - EloRating (from driver snapshot)
    - RecentFormAvg (from driver snapshot)
    - CircuitHistoryAvg (from circuit snapshot)
    - DriverExperienceScore (from driver snapshot)
    - driver_enc, team_enc, circuit_enc (encoded categoricals)
    """
    import time
    start = time.time()
    
    q = qual_df.copy()
    q["race_key"] = race_key
    q["race_year"] = race_year
    q["event"] = event
    q["circuit"] = circuit
    
    # Get pre-computed features from FeatureStore (cached, instant)
    driver_codes = q["driver"].tolist()
    teams = q["team"].tolist() if "team" in q.columns else [None] * len(driver_codes)
    
    # Fetch features for all drivers efficiently
    feature_rows = []
    for i, driver in enumerate(driver_codes):
        team = teams[i] if i < len(teams) else None
        
        # Get pre-computed driver features (RecentFormAvg, EloRating, TotalRaces, etc.)
        driver_features = feature_store.get_driver_features(driver)
        
        # Get circuit-specific history (cached per driver+circuit pair)
        circuit_history = feature_store.get_circuit_history(driver, circuit)
        
        # Get team performance score for this year
        team_perf = feature_store.get_team_perf_score(team, race_year) if team else 0.5
        
        feature_rows.append({
            "driver": driver,
            "RecentFormAvg": driver_features.get("RecentFormAvg", 10.0),
            "EloRating": driver_features.get("EloRating", 1500.0),
            "DriverExperienceScore": driver_features.get("DriverExperienceScore", 0.0),
            "CircuitHistoryAvg": circuit_history,
            "TeamPerfScore": team_perf,
        })
    
    features_df = pd.DataFrame(feature_rows)
    
    # Merge qualifying data with pre-computed features
    race_rows = q.merge(features_df, on="driver", how="left")
    
    # Fill missing features with defaults (for rookies/new drivers)
    race_rows["RecentFormAvg"] = race_rows["RecentFormAvg"].fillna(10.0)
    race_rows["EloRating"] = race_rows["EloRating"].fillna(1500.0)
    race_rows["DriverExperienceScore"] = race_rows["DriverExperienceScore"].fillna(0.0)
    race_rows["CircuitHistoryAvg"] = race_rows["CircuitHistoryAvg"].fillna(10.0)
    race_rows["TeamPerfScore"] = race_rows["TeamPerfScore"].fillna(0.5)
    
    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in race_rows.columns:
            race_rows[col] = 0

    # Encode categorical features
    for c in ["driver", "team", "circuit"]:
        le = encoders[c]
        classes = list(le.classes_)
        mapped = []
        for v in race_rows[c].astype(str):
            if v in classes:
                mapped.append(classes.index(v))
            else:
                mapped.append(len(classes))
        race_rows[f"{c}_enc"] = mapped

    X = race_rows[feature_cols].fillna(0)
    
    logger.debug(f"Feature preparation took {time.time() - start:.3f}s")
    logger.debug(f"Features prepared: {list(X.columns)}")

    # Predict
    race_rows["p_win"] = model_win.predict_proba(X)[:,1]
    race_rows["p_pod"] = model_pod.predict_proba(X)[:,1]

    # Hybrid podium ranking
    max_grid = int(race_rows["qualifying_position"].max(skipna=True)) if race_rows["qualifying_position"].notna().any() else 20
    qual = race_rows["qualifying_position"].fillna(max_grid)
    qual_score = 1 - ((qual - 1) / max(1, max_grid - 1))
    race_rows["hybrid_score"] = 0.6 * race_rows["p_pod"] + 0.4 * qual_score

    # Build outputs
    winner = race_rows.sort_values("p_win", ascending=False).iloc[0]
    hybrid_board = race_rows.sort_values("hybrid_score", ascending=False)

    # Calculate winner confidence
    winner_pct = int(float(winner["p_win"]) * 100)
    winner_confidence = get_confidence_level(winner_pct)
    
    # Log prediction in MLflow (local tracking)
    try:
        log_prediction(
            race_name=event,
            predicted_winner=winner["driver"],
            confidence=winner_pct,
            model_version="v3"
        )
    except Exception as e:
        logger.debug(f"MLflow prediction logging skipped: {e}")
    
    # Log prediction to Supabase (production tracking)
    try:
        prediction_logger.log_prediction(
            race_name=event,
            predicted_winner=winner["driver"],
            confidence=winner_pct,
            model_version="v3"
        )
    except Exception as e:
        logger.debug(f"Supabase prediction logging skipped: {e}")

    return {
        "winner_prediction": {
            "driver": winner["driver"], 
            "team": winner.get("team", None), 
            "p_win": float(winner["p_win"]),
            "percentage": winner_pct,
            "confidence": winner_confidence["level"],
            "confidence_color": winner_confidence["color"]
        },
        "top3_prediction": hybrid_board.head(3)[["driver","team","hybrid_score","p_pod"]].apply(
            lambda row: {
                "driver": row["driver"],
                "team": row["team"],
                "hybrid_score": float(row["hybrid_score"]),
                "percentage": int(float(row["hybrid_score"]) * 100),
                "confidence": get_confidence_level(int(float(row["hybrid_score"]) * 100))["level"],
                "confidence_color": get_confidence_level(int(float(row["hybrid_score"]) * 100))["color"]
            }, axis=1
        ).tolist(),
        "full_predictions": race_rows.sort_values("p_win", ascending=False)[["driver","team","p_win","p_pod","hybrid_score"]].apply(
            lambda row: {
                "driver": row["driver"],
                "team": row["team"],
                "p_win": float(row["p_win"]),
                "percentage": int(float(row["p_win"]) * 100),
                "confidence": get_confidence_level(int(float(row["p_win"]) * 100))["level"],
                "confidence_color": get_confidence_level(int(float(row["p_win"]) * 100))["color"]
            }, axis=1
        ).tolist(),
        "confidence_thresholds": CONFIDENCE_THRESHOLDS
    }


@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        "status": "online",
        "message": "F1 Prediction API is running",
        "version": "1.0.0"
    })


@app.route('/api/health')
def health_check():
    """Detailed health check with feature store status"""
    fs_health = feature_store.health_check()
    return jsonify({
        "status": "healthy",
        "version": "2.0.0",
        "architecture": "optimized",
        "feature_store": {
            "initialized": True,
            "drivers_cached": fs_health["drivers_in_memory"],
            "parquet_available": fs_health["parquet_exists"],
            "redis_connected": fs_health["redis_connected"],
            "lru_cache": fs_health["lru_cache_info"]
        },
        "prediction_logger": {
            "mode": prediction_logger.mode,
            "supabase_connected": prediction_logger.mode == "supabase"
        },
        "data": {
            "historical_format": "parquet" if fs_health["parquet_exists"] else "csv",
            "compression": "5.5x" if fs_health["parquet_exists"] else "none"
        }
    })


@app.route('/api/prediction-accuracy', methods=['GET'])
def prediction_accuracy():
    """
    Get prediction accuracy statistics from Supabase/CSV logs.
    Returns overall accuracy, recent accuracy, and trend.
    """
    try:
        stats = prediction_logger.get_accuracy_stats()
        return jsonify({
            "success": True,
            "data": stats
        })
    except Exception as e:
        logger.error(f"Error getting accuracy stats: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/prediction-history', methods=['GET'])
def prediction_history():
    """
    Get recent prediction history from Supabase/CSV logs.
    Query params: limit (default 100)
    """
    try:
        limit = request.args.get('limit', 100, type=int)
        history = prediction_logger.get_prediction_history(limit=limit)
        
        return jsonify({
            "success": True,
            "count": len(history),
            "data": history.to_dict(orient='records') if not history.empty else []
        })
    except Exception as e:
        logger.error(f"Error getting prediction history: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/update-actual-winner', methods=['POST'])
def update_actual_winner():
    """
    Update the actual winner for a race after it completes.
    Expects JSON: {"race_name": "São Paulo Grand Prix", "actual_winner": "NOR"}
    """
    try:
        data = request.json or {}
        race_name = data.get('race_name')
        actual_winner = data.get('actual_winner')
        
        if not race_name or not actual_winner:
            return jsonify({
                "success": False,
                "error": "Both race_name and actual_winner are required"
            }), 400
        
        success = prediction_logger.update_actual_winner(race_name, actual_winner)
        
        return jsonify({
            "success": success,
            "message": f"Updated actual winner for {race_name} to {actual_winner}" if success else "Update failed"
        })
    except Exception as e:
        logger.error(f"Error updating actual winner: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    Expects JSON: {
        "race_key": "2025__Sao_Paulo_Grand_Prix",
        "race_year": 2025,
        "event": "São Paulo Grand Prix",
        "circuit": "Interlagos",
        "qualifying": [  # Optional - will auto-fetch if not provided
            {"driver": "NOR", "team": "McLaren", "qualifying_position": 1},
            ...
        ]
    }
    """
    try:
        data = request.json or {}
        
        # Validate minimum required fields
        required_fields = ["race_key", "race_year", "event", "circuit"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # If qualifying not provided, attempt to fetch via internal functions
        qualifying = data.get("qualifying")
        if not qualifying:
            # prefer race_key else try (race_year + circuit/event)
            rk = data.get("race_key")
            ry = data.get("race_year")
            circuit = data.get("circuit")
            event = data.get("event")
            try:
                # call internal function rather than HTTP call
                hist_q = get_qual_from_history(race_key=rk, race_year=ry, circuit=circuit, event=event)
                if not hist_q.empty:
                    qual_df = hist_q.sort_values("qualifying_position")[["driver","team","qualifying_position","qualifying_lap_time_s"]]
                    qualifying = qual_df.to_dict("records")
                else:
                    # try Ergast live fetch
                    if ry is None:
                        return jsonify({"success": False, "error": "race_year required when qualifying not provided"}), 400
                    qualifying = fetch_qualifying_from_ergast(int(ry), circuit=circuit, event=event)
                    if not qualifying:
                        return jsonify({"success": False, "error": "No qualifying data available"}), 404
            except Exception as e:
                logger.error(f"Could not fetch qualifying: {e}")
                return jsonify({"success": False, "error": str(e)}), 500

        # Convert qualifying data to DataFrame
        qual_df = pd.DataFrame(qualifying)
        
        # Generate predictions
        predictions = infer_from_qualifying(
            qual_df,
            data["race_key"],
            data["race_year"],
            data["event"],
            data["circuit"]
        )
        
        return jsonify({
            "success": True,
            "data": predictions
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/qualifying', methods=['GET'])
def get_qualifying():
    """
    Query params:
      - race_key (optional)
      - race_year (optional)
      - circuit (optional)
      - event (optional)
    Behavior:
      1) returns qualifying rows from history if found
      2) otherwise attempts to fetch from Ergast and returns that
    """
    try:
        race_key = request.args.get("race_key")
        race_year = request.args.get("race_year")
        circuit = request.args.get("circuit")
        event = request.args.get("event")

        # 1) try history
        hist_q = get_qual_from_history(race_key=race_key, race_year=race_year, circuit=circuit, event=event)
        if not hist_q.empty:
            # normalize to minimal JSON structure
            rows = hist_q.sort_values("qualifying_position").loc[:, ["driver", "team", "qualifying_position", "qualifying_lap_time_s"]].to_dict("records")
            return jsonify({"success": True, "source": "history", "qualifying": rows})

        # 2) attempt live fetch from Ergast
        if race_year is None:
            return jsonify({"success": False, "error": "race_year required if not present in history"}), 400

        qual_list = fetch_qualifying_from_ergast(int(race_year), circuit=circuit, event=event)
        if qual_list:
            return jsonify({"success": True, "source": "ergast", "qualifying": qual_list})
        else:
            return jsonify({"success": False, "error": "No qualifying data found"}), 404

    except Exception as e:
        logger.error(f"Qualifying lookup error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


def get_races_with_predictions_and_history():
    """
    Get last 5 completed races with predictions and history.
    OPTIMIZED: Uses cache first, skips expensive FastF1 calls when possible
    Returns faster even if not all data is complete - targets <15s response
    """
    try:
        logger.info("Building race history with predictions...")
        
        year = 2025
        schedule = fastf1.get_event_schedule(year)
        schedule['EventDate'] = pd.to_datetime(schedule['EventDate'])
        
        today = pd.to_datetime(datetime.now().date())
        
        # Get last 5 completed races
        past_races = schedule[schedule['EventDate'] <= today].sort_values('EventDate', ascending=False)
        last_5_races = past_races.head(5)
        
        races = []
        processed_count = 0
        
        # Process each of the last 5 races - but timeout early if taking too long
        for idx, event in last_5_races.iterrows():
            try:
                if processed_count >= 2:  # Only process 2 races to keep response time under 15s
                    logger.info(f"Stopping at {processed_count} races to keep response fast")
                    break
                
                race_name = event.get('EventName') or event.get('Event') or 'Unknown'
                race_year = int(event.get('Year') or event.get('year') or 2025)
                race_round = int(event.get('RoundNumber') or event.get('round') or 1)
                race_date = event.get('EventDate')
                race_key = f"{race_year}_{race_round}_{race_name.replace(' ', '_')}"
                
                logger.info(f"Processing race: {race_name} ({race_date.strftime('%Y-%m-%d')})")
                
                # 1. Try to get qualifying data from cache FIRST (fast)
                qual_data = None
                
                try:
                    cache = get_qualifying_cache(config)
                    cached_qual = cache.get_cached_qualifying(race_key)
                    
                    if cached_qual:
                        if isinstance(cached_qual, str):
                            import json
                            qual_data = json.loads(cached_qual)
                        else:
                            qual_data = cached_qual
                        
                        # Normalize field names
                        if isinstance(qual_data, list):
                            for entry in qual_data:
                                if 'code' in entry and 'driver' not in entry:
                                    entry['driver'] = entry.pop('code')
                                if 'position' in entry and 'qualifying_position' not in entry:
                                    entry['qualifying_position'] = entry.pop('position')
                        
                        logger.info(f"  ✓ Got qualifying from cache: {len(qual_data) if isinstance(qual_data, list) else 1} drivers")
                        
                except Exception as e:
                    logger.debug(f"Cache lookup failed: {e}")
                    qual_data = None
                
                # If no cached data, skip FastF1 fetch (too slow on Render, causes timeout)
                if qual_data is None:
                    logger.info(f"  Skipping FastF1 fetch for {race_name} (would cause timeout)")
                    continue
                
                # 2. Run model prediction
                prediction_data = {
                    "predicted_winner": "N/A",
                    "predicted_confidence": 0,
                    "top3": []
                }
                
                try:
                    qual_df = pd.DataFrame(qual_data) if isinstance(qual_data, list) else pd.DataFrame([qual_data])
                    predictions = infer_from_qualifying(qual_df, race_key, race_year, race_name, race_name)
                    
                    prediction_data = {
                        "predicted_winner": predictions["winner_prediction"]["driver"],
                        "predicted_confidence": predictions["winner_prediction"]["percentage"],
                        "top3": [d["driver"] for d in predictions["top3_prediction"][:3]]
                    }
                    
                    logger.info(f"  ✓ Prediction: {prediction_data['predicted_winner']} ({prediction_data['predicted_confidence']}%)")
                    
                except Exception as e:
                    logger.error(f"  Prediction error for {race_name}: {e}")
                
                # 3. Try to get actual winner from training data (fast, cached)
                actual_winner = "TBA"
                is_correct = False
                
                try:
                    # Use training data to find actual winner (no FastF1 call!)
                    hist_match = hist_data[
                        (hist_data['race_year'] == race_year) & 
                        (hist_data['event'].str.contains(race_name.split()[0], case=False, na=False))
                    ]
                    
                    if not hist_match.empty:
                        winners = hist_match[hist_match['finishing_position'] == 1]
                        if not winners.empty:
                            actual_winner = winners.iloc[0]['driver']
                            is_correct = (actual_winner.upper() == str(prediction_data['predicted_winner']).upper())
                            logger.info(f"  ✓ Actual winner from history: {actual_winner} | Correct: {is_correct}")
                    else:
                        logger.info(f"  No results found for {race_name} in training data")
                        
                except Exception as e:
                    logger.debug(f"  Could not fetch results: {e}")
                
                races.append({
                    "round": race_round,
                    "race": race_name,
                    "circuit": race_name,
                    "date": race_date.strftime('%Y-%m-%d'),
                    "predicted_winner": prediction_data["predicted_winner"],
                    "predicted_top3": prediction_data["top3"],
                    "predicted_confidence": prediction_data["predicted_confidence"],
                    "actual_winner": actual_winner,
                    "is_correct": is_correct,
                    "qualifying_count": len(qual_data) if isinstance(qual_data, list) else 1
                })
                
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing race: {e}")
                continue
        
        logger.info(f"✓ Race history complete: {len(races)} races processed")
        return races
        
    except Exception as e:
        logger.error(f"Error building race history: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []


def get_next_race_prediction():
    """
    Get prediction for the next upcoming race.
    Fetches qualifying, runs prediction, returns complete prediction data.
    
    Returns dict with:
    - race_name, race_year, circuit, date
    - predicted_winner, predicted_confidence, top3
    - qualifying_data
    - status (ready/pending)
    """
    try:
        logger.info("Getting next race prediction...")
        
        year = 2025
        schedule = fastf1.get_event_schedule(year)
        schedule['EventDate'] = pd.to_datetime(schedule['EventDate'])
        
        today = pd.to_datetime(datetime.now().date())
        
        # Find next race
        future_races = schedule[schedule['EventDate'] > today].sort_values('EventDate')
        
        if future_races.empty:
            logger.warning("No upcoming races found")
            return None
        
        next_event = future_races.iloc[0]
        race_name = next_event.get('EventName') or next_event.get('Event') or 'Unknown'
        race_year = int(next_event.get('Year') or next_event.get('year') or 2025)
        race_round = int(next_event.get('RoundNumber') or next_event.get('round') or 1)
        race_date = next_event.get('EventDate')
        race_key = f"{race_year}_{race_round}_{race_name.replace(' ', '_')}"
        
        logger.info(f"Next race: {race_name} ({race_date.strftime('%Y-%m-%d')})")
        
        # Get qualifying data (from cache or latest available)
        qual_data = None
        
        try:
            cache = get_qualifying_cache(config)
            cached_qual = cache.get_cached_qualifying(race_key)
            
            if cached_qual:
                if isinstance(cached_qual, str):
                    import json
                    qual_data = json.loads(cached_qual)
                else:
                    qual_data = cached_qual
                
                # Normalize field names
                if isinstance(qual_data, list):
                    for entry in qual_data:
                        if 'code' in entry and 'driver' not in entry:
                            entry['driver'] = entry.pop('code')
                        if 'position' in entry and 'qualifying_position' not in entry:
                            entry['qualifying_position'] = entry.pop('position')
            else:
                # Use latest cached qualifying from any race
                latest = cache.get_latest_cached_qualifying()
                if latest:
                    if isinstance(latest, str):
                        import json
                        qual_data = json.loads(latest)
                    else:
                        qual_data = latest
                    
                    # Normalize field names
                    if isinstance(qual_data, list):
                        for entry in qual_data:
                            if 'code' in entry and 'driver' not in entry:
                                entry['driver'] = entry.pop('code')
                            if 'position' in entry and 'qualifying_position' not in entry:
                                entry['qualifying_position'] = entry.pop('position')
                    
                    logger.info(f"  Using latest cached qualifying (different race)")
                else:
                    logger.warning(f"  No qualifying data cached yet")
                    qual_data = None
                    
        except Exception as e:
            logger.debug(f"Cache lookup failed: {e}")
            qual_data = None
        
        # If we have qualifying, run prediction
        prediction_data = {
            "predicted_winner": "TBA",
            "predicted_confidence": 0,
            "top3": [],
            "status": "pending"
        }
        
        if qual_data:
            try:
                qual_df = pd.DataFrame(qual_data) if isinstance(qual_data, list) else pd.DataFrame([qual_data])
                predictions = infer_from_qualifying(qual_df, race_key, race_year, race_name, race_name)
                
                prediction_data = {
                    "predicted_winner": predictions["winner_prediction"]["driver"],
                    "predicted_confidence": predictions["winner_prediction"]["percentage"],
                    "top3": [d["driver"] for d in predictions["top3_prediction"][:3]],
                    "full_predictions": predictions.get("full_predictions", []),
                    "status": "ready"
                }
                
                logger.info(f"  ✓ Prediction: {prediction_data['predicted_winner']} ({prediction_data['predicted_confidence']}%)")
                
            except Exception as e:
                logger.error(f"  Prediction error: {e}")
        
        return {
            "round": race_round,
            "race": race_name,
            "circuit": race_name,
            "date": race_date.strftime('%Y-%m-%d'),
            "qualifying_count": len(qual_data) if isinstance(qual_data, list) else (1 if qual_data else 0),
            "predicted_winner": prediction_data["predicted_winner"],
            "predicted_confidence": prediction_data["predicted_confidence"],
            "predicted_top3": prediction_data["top3"],
            "status": prediction_data["status"]
        }
        
    except Exception as e:
        logger.error(f"Error getting next race prediction: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


@app.route('/api/predict/sao-paulo', methods=['GET'])
def predict_sao_paulo():
    """
    Current Race Tab Endpoint
    Returns:
    1. Last 5 races with predictions, actuals, and accuracy
    2. Next race prediction (pending or ready)
    Follows the race history + prediction logic
    
    Response includes CORS headers for Vercel frontend access
    """
    try:
        logger.info("=== Current Race Tab Request ===")
        
        # 1. Get race history with predictions for last 5 races
        race_history = get_races_with_predictions_and_history()
        
        # 2. Get next race prediction
        next_race = get_next_race_prediction()
        
        # Calculate accuracy stats
        correct_predictions = 0
        accuracy_pct = 0
        
        if race_history:
            correct_predictions = sum(1 for r in race_history if r.get("is_correct", False))
            accuracy_pct = int((correct_predictions / len(race_history)) * 100) if race_history else 0
        
        logger.info(f"Accuracy: {accuracy_pct}% ({correct_predictions}/{len(race_history) if race_history else 0})")
        
        response = jsonify({
            "success": True,
            "race_history": race_history or [],
            "next_race": next_race,
            "accuracy": {
                "percentage": accuracy_pct,
                "correct_predictions": correct_predictions,
                "total_races": len(race_history) if race_history else 0
            }
        })
        
        # Ensure CORS headers are set
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        
        return response
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        response = jsonify({
            "success": False,
            "error": str(e)[:200],  # Limit error message length
            "type": type(e).__name__
        })
        
        # Ensure CORS headers even on error
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        
        return response, 500


# ---------- F1 Points System (2025) ----------
F1_POINTS_2025 = {
    1: 25,    # 1st place
    2: 18,    # 2nd place
    3: 15,    # 3rd place (podium)
    4: 12,
    5: 10,
    6: 8,
    7: 6,
    8: 4,
    9: 2,
    10: 1,
    # 11+: 0 points
}

def points_for_position(position):
    """Get F1 points for a finishing position"""
    try:
        pos = int(position)
        return F1_POINTS_2025.get(pos, 0)
    except:
        return 0

def get_2025_driver_standings_from_fastf1():
    """
    Fetch 2025 driver standings from Ergast API (actual current points)
    Returns list of drivers with current points and team info
    """
    try:
        logger.info("Fetching 2025 driver standings from Ergast...")
        
        # Use FastF1's Ergast wrapper to get latest standings
        from fastf1.ergast import Ergast
        ergast = Ergast()
        
        # Get latest standings for 2025 (None means latest round)
        standings_data = ergast.get_driver_standings(season=2025, round=None)
        
        if standings_data.content is None or len(standings_data.content) == 0:
            logger.warning("No standings data from Ergast")
            return []
        
        df = standings_data.content[0]
        
        # Convert Ergast standings to our format
        driver_standings = []
        
        for idx, row in df.iterrows():
            try:
                # Get driver info
                driver_code = row.get('driverCode') or row.get('code')
                given_name = row.get('givenName', '')
                family_name = row.get('familyName', '')
                driver_name = f"{given_name} {family_name}".strip()
                
                # Get team/constructor from list
                constructor_names = row.get('constructorNames', [])
                team_name = constructor_names[0] if isinstance(constructor_names, list) and len(constructor_names) > 0 else ''
                
                # Get actual points
                actual_points = int(float(row.get('points', 0)))
                position = int(row.get('position', idx + 1))
                
                driver_standings.append({
                    "position": position,
                    "name": driver_name,
                    "team": team_name,
                    "points": actual_points,
                    "races": int(row.get('wins', 0)),  # Store wins in races field for now
                    "code": driver_code
                })
                
                logger.debug(f"  {position}. {driver_name} ({team_name}): {actual_points} pts")
                
            except Exception as e:
                logger.debug(f"Error processing driver row: {e}")
                continue
        
        logger.info(f"✓ 2025 standings from Ergast: {len(driver_standings)} drivers")
        return driver_standings
        
    except Exception as e:
        logger.error(f"Error fetching 2025 standings from Ergast: {e}")
        return []

def predict_driver_points_for_future_races(driver_code, current_points, num_races=5):
    """
    Predict additional points for a driver in future races
    Simple prediction: average recent performance * remaining races
    """
    try:
        # Get recent races for this driver from training data
        recent = hist_data[hist_data['driver'] == driver_code].tail(10)
        
        if recent.empty:
            # No history - estimate average points per race
            avg_points_per_race = 1.5  # Conservative estimate
            return int(avg_points_per_race * num_races)
        
        # Calculate average points per race (including DNFs with 0 points)
        avg_points_per_race = recent['points'].mean()
        
        # Predict for remaining races
        predicted_additional = int(avg_points_per_race * num_races)
        
        return predicted_additional
        
    except Exception as e:
        logger.debug(f"Could not predict points for {driver_code}: {e}")
        return 0

def get_2025_constructor_standings_from_ergast():
    """
    Fetch 2025 constructor standings from Ergast API (actual current points)
    Returns list of constructors with current points
    """
    try:
        logger.info("Fetching 2025 constructor standings from Ergast...")
        
        # Use FastF1's Ergast wrapper to get latest constructor standings
        from fastf1.ergast import Ergast
        ergast = Ergast()
        
        # Get latest standings for 2025 (None means latest round)
        standings_data = ergast.get_constructor_standings(season=2025, round=None)
        
        if standings_data.content is None or len(standings_data.content) == 0:
            logger.warning("No constructor standings data from Ergast")
            return []
        
        df = standings_data.content[0]
        
        # Convert Ergast standings to our format
        constructor_standings = []
        
        for idx, row in df.iterrows():
            try:
                # Get constructor info
                constructor_name = row.get('constructorName', '')
                constructor_id = row.get('constructorId', '')
                
                # Get actual points
                actual_points = int(float(row.get('points', 0)))
                position = int(row.get('position', idx + 1))
                wins = int(row.get('wins', 0))
                
                constructor_standings.append({
                    "position": position,
                    "name": constructor_name,
                    "points": actual_points,
                    "wins": wins,
                    "id": constructor_id
                })
                
                logger.debug(f"  {position}. {constructor_name}: {actual_points} pts")
                
            except Exception as e:
                logger.debug(f"Error processing constructor row: {e}")
                continue
        
        logger.info(f"✓ 2025 constructor standings from Ergast: {len(constructor_standings)} constructors")
        return constructor_standings
        
    except Exception as e:
        logger.error(f"Error fetching 2025 constructor standings from Ergast: {e}")
        return []

def predict_constructor_points_for_future_races(constructor_name, current_points, num_races=5):
    """
    Predict additional points for a constructor in future races
    Aggregates from all drivers of that constructor
    """
    try:
        # Get all drivers from this constructor from training data
        constructor_drivers = hist_data[hist_data['team'].str.contains(constructor_name, case=False, na=False)]
        
        if constructor_drivers.empty:
            # No history - estimate average points per race
            avg_points_per_race = 3.0  # Conservative estimate for constructor
            return int(avg_points_per_race * num_races)
        
        # Calculate average points per race for this constructor
        avg_points_per_race = constructor_drivers['points'].mean()
        
        # Predict for remaining races
        predicted_additional = int(avg_points_per_race * num_races)
        
        return predicted_additional
        
    except Exception as e:
        logger.debug(f"Could not predict points for {constructor_name}: {e}")
        return 0


def get_latest_race_circuit_data():
    """Get latest completed race circuit data with top 6 drivers and their performance"""
    try:
        # Circuit image mapping for F1 races - using simple placeholder since direct URLs have issues
        # In production, these would come from a circuit database or cache
        circuit_images = {
            'São Paulo': 'https://via.placeholder.com/1440x810/1a1a1a/00d4ff?text=São+Paulo+Circuit',
            'Abu Dhabi': 'https://via.placeholder.com/1440x810/1a1a1a/00d4ff?text=Abu+Dhabi+Circuit',
            'Las Vegas': 'https://via.placeholder.com/1440x810/1a1a1a/00d4ff?text=Las+Vegas+Circuit',
            'Qatar': 'https://via.placeholder.com/1440x810/1a1a1a/00d4ff?text=Qatar+Circuit',
            'Suzuka': 'https://via.placeholder.com/1440x810/1a1a1a/00d4ff?text=Suzuka+Circuit'
        }
        
        # Check rounds in reverse order (prioritize recent races)
        # 2025: São Paulo (21), Las Vegas (22)
        rounds_to_check = [22, 21, 23, 24] + list(range(20, 0, -1))
        
        session = None
        latest_round = None
        
        logger.info(f"Searching for latest completed race with results...")
        
        for round_num in rounds_to_check:
            try:
                s = fastf1.get_session(2025, round_num, 'R')
                s.load(telemetry=False, weather=False)
                
                # Check if results are available and not empty
                if hasattr(s, 'results') and s.results is not None and not s.results.empty and len(s.results) > 0:
                    session = s
                    latest_round = round_num
                    logger.info(f"✓ Found race with results: Round {round_num} - {s.event.get('OfficialEventName', 'Unknown')}")
                    break
            except Exception as e:
                logger.debug(f"Round {round_num} error: {str(e)[:100]}")
                continue
        
        if session is None:
            logger.warning("No completed races with results found in FastF1")
            return None
        
        results = session.results
        event_name = session.event.get('OfficialEventName', 'Unknown Race')
        race_date = session.date.isoformat() if session.date else ""
        
        # Extract circuit name for image lookup
        circuit_lookup = None
        for key in circuit_images.keys():
            if key.lower() in event_name.lower():
                circuit_lookup = key
                break
        
        circuit_image = circuit_images.get(circuit_lookup, circuit_images.get('Suzuka', ''))
        
        # Get top 6 finishers
        top_6 = results.head(6)[['Position', 'DriverNumber', 'Abbreviation', 'FullName', 'TeamName', 'Points']]
        
        # Get fastest lap times by driver
        try:
            laps = session.laps
            fastest_laps = laps.groupby('Driver')['LapTime'].min().sort_values()
        except:
            fastest_laps = pd.Series()
        
        drivers_data = []
        for idx, row in top_6.iterrows():
            driver_abbr = row['Abbreviation']
            fastest_lap = None
            
            if driver_abbr in fastest_laps.index:
                fastest_lap = str(fastest_laps[driver_abbr]).split('.')[-1][:3]  # Format as MM:SS.mmm
            
            drivers_data.append({
                "position": int(row['Position']),
                "abbreviation": driver_abbr,
                "fullName": row['FullName'],
                "teamName": row['TeamName'],
                "points": int(row['Points']),
                "fastestLap": fastest_lap,
                "driverNumber": int(row['DriverNumber'])
            })
        
        # Circuit details (basic - could be enhanced with FastF1 corner data)
        circuit_data = {
            "circuitName": event_name,
            "circuitImage": circuit_image,
            "roundNumber": latest_round,
            "raceDate": race_date,
            "trackLength": "Unknown",  # Would need external data source
            "totalLaps": len(session.laps.groupby('LapNumber')) if not session.laps.empty else 0,
            "raceDistance": "Unknown"
        }
        
        logger.info(f"✓ Latest race circuit data: {event_name} (Round {latest_round}) with {len(drivers_data)} drivers")
        
        return {
            "circuit": circuit_data,
            "drivers": drivers_data
        }
        
    except Exception as e:
        logger.error(f"Error getting latest race circuit data: {e}")
        return None


# ---------- New Ergast + Wikipedia endpoints ----------
@app.route("/api/race-history", methods=["GET", "OPTIONS"])
def race_history():
    """
    Get last 5 completed races with:
    1. Qualifying data from Supabase cache (instant)
    2. Model predictions based on qualifying
    3. Actual winners (from training data for historical races)
    """
    # Handle CORS preflight
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response, 200
    try:
        # Check memory cache first
        cache = get_file_cache()
        cache_key = CACHE_KEYS["RACE_HISTORY"]
        cached_result = cache.get(cache_key, ttl_seconds=CACHE_TTL["RACE_HISTORY"])
        
        if cached_result is not None:
            logger.info("✓ Returning cached race history from memory")
            return jsonify({
                "success": True,
                "data": cached_result,
                "source": "memory_cache"
            })
        
        logger.info("Fetching race history from Supabase cache...")
        
        races = []
        
        # Get latest qualifying cache from Supabase
        try:
            # Use the QualifyingCache instance from database_v2
            cached_qualifying = get_qualifying_cache()
            
            if cached_qualifying is None:
                logger.warning("No qualifying data in Supabase cache")
                return jsonify({
                    "success": True,
                    "data": [],
                    "source": "empty",
                    "message": "No qualifying data cached yet"
                })
            
            # cached_qualifying is the JSON/dict of qualifying data
            if isinstance(cached_qualifying, str):
                import json
                qualifying_data = json.loads(cached_qualifying)
            else:
                qualifying_data = cached_qualifying
            
            # Extract race info from the cache or use latest
            race_name = "Latest Qualifying"  # Default name
            race_year = 2025
            race_date = "2025-12-06"
            
            logger.info(f"Cached qualifying type: {type(qualifying_data)}, is list: {isinstance(qualifying_data, list)}")
            if isinstance(qualifying_data, list) and len(qualifying_data) > 0:
                logger.info(f"Using cached qualifying with {len(qualifying_data)} drivers")
                logger.info(f"First driver keys: {list(qualifying_data[0].keys())}")
                
                # Extract driver code from different possible field names
                for driver_entry in qualifying_data:
                    # Map field names
                    if 'code' in driver_entry and 'name' not in driver_entry:
                        # Rename to match model expectations
                        driver_entry['driver'] = driver_entry.pop('code')
                    if 'name' in driver_entry and 'team' not in driver_entry:
                        driver_entry['team'] = driver_entry.get('team', 'Unknown')
                
                # Try to get actual winner from FastF1 (but with timeout fallback)
                actual_winner = "TBA"
                try:
                    actual_winner = get_race_winner_from_fastf1(race_year, race_name)
                except Exception as e:
                    logger.warning(f"Could not fetch race winner from FastF1: {e}")
                
                # If not found in FastF1, try training data
                if actual_winner is None:
                    actual_winner = "TBA"
                
                # Run model prediction using qualifying data
                predicted_winner = "N/A"
                predicted_confidence = 0
                is_correct = False
                
                try:
                    # Create DataFrame from qualifying data - handle both formats
                    if isinstance(qualifying_data[0], dict) and 'telemetry' in qualifying_data[0]:
                        # Format: [{"driver": "VER", "telemetry": [...], ...}, ...]
                        qual_df = pd.DataFrame([
                            {
                                'driver': d.get('driver'),
                                'team': d.get('team', 'Unknown'),
                                'qualifying_position': d.get('qualifying_position', i+1)
                            }
                            for i, d in enumerate(qualifying_data)
                        ])
                    elif 'code' in qualifying_data[0]:
                        # New format from cached telemetry
                        qual_df = pd.DataFrame([
                            {
                                'driver': d.get('code'),
                                'team': d.get('team', 'Unknown'),
                                'qualifying_position': d.get('qualifying_position', i+1)
                            }
                            for i, d in enumerate(qualifying_data)
                        ])
                    else:
                        # Format: [{"driver": "VER", "team": "...", "qualifying_position": 1}, ...]
                        qual_df = pd.DataFrame(qualifying_data)
                    
                    logger.info(f"qual_df shape: {qual_df.shape}, columns: {list(qual_df.columns)}")
                    race_key = f"{race_year}_Latest"
                    
                    # Run model prediction
                    predictions = infer_from_qualifying(
                        qual_df,
                        race_key,
                        race_year,
                        race_name,
                        race_name
                    )
                    
                    predicted_winner = predictions.get("winner_prediction", {}).get("driver", "N/A")
                    predicted_confidence = int(predictions.get("winner_prediction", {}).get("percentage", 0))
                    
                    # Check if prediction matches actual winner
                    if actual_winner != "TBA" and predicted_winner != "N/A":
                        is_correct = (predicted_winner.lower() == str(actual_winner).lower())
                    
                    logger.info(f"✓ Prediction: {predicted_winner} ({predicted_confidence}%) | Actual: {actual_winner} | Match: {is_correct}")
                    
                except Exception as e:
                    logger.error(f"Prediction error: {e}")
                    import traceback
                    traceback.print_exc()
                
                races.append({
                    "race": race_name,
                    "circuit": race_name,
                    "predicted_winner": predicted_winner,
                    "actual_winner": actual_winner,
                    "correct": is_correct,
                    "confidence": predicted_confidence,
                    "date": race_date
                })
        
        except ImportError:
            logger.warning("database_v2 not available, using file cache fallback")
            pass
        except Exception as e:
            logger.error(f"Supabase cache error: {e}")
            import traceback
            traceback.print_exc()
        
        logger.info(f"Race history ready: {len(races)} race(s) with predictions")
        
        # Save to memory cache for next request (short TTL)
        if races:
            cache.set(CACHE_KEYS["RACE_HISTORY"], races)
            logger.info("✓ Cached race history in memory")
        
        return jsonify({
            "success": True,
            "data": races,
            "source": "supabase_cache"
        })
    
    except Exception as e:
        logger.error(f"Race history error: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback: return empty with error explanation
        return jsonify({
            "success": False,
            "error": str(e),
            "data": []
        }), 500


# ---------- New Ergast + Wikipedia endpoints ----------
@app.route("/api/next-race", methods=["GET"])
def api_next_race():
    """Get next race information from FastF1 with circuit image from Wikipedia"""
    # Get next race from FastF1
    nr = get_next_race_from_fastf1(year=2025)
    
    if not nr:
        # Fallback to Ergast if FastF1 doesn't have data
        nr = ergast_next_race() or {}
    
    if not nr:
        return jsonify({"success": False, "error": "No next race found"}), 404
    
    # Try to fetch a circuit image via Wikipedia using event name
    event_name = nr.get("event_name") or nr.get("event")
    circuit_img = None
    if event_name:
        circuit_img = fetch_wikipedia_image(event_name)
        if circuit_img:
            return jsonify({"success": True, "race": nr, "circuit_image_url": f"/images/{Path(circuit_img).name}"})
    
    return jsonify({"success": True, "race": nr, "circuit_image_url": None})

# Serve cached images under /images/<name>
@app.route("/images/<path:filename>")
def images(filename):
    """Serve cached images"""
    fp = IMG_CACHE / filename
    if fp.exists():
        return send_file(str(fp))
    return ("Not found", 404)

@app.route("/api/driver-image", methods=["GET"])
def driver_image():
    """Get driver image from Wikipedia cache"""
    # query param 'name' expected: "Max Verstappen" or driver code
    name = request.args.get("name")
    if not name:
        return jsonify({"success": False, "error":"missing : name"}), 400
    local = fetch_wikipedia_image(name)
    if local:
        return jsonify({"success": True, "image_url": f"/images/{Path(local).name}"})
    return jsonify({"success": False, "image_url": None}), 404

@app.route("/api/driver-standings", methods=["GET"])
def api_driver_standings():
    """
    Get 2025 driver standings with actual points from FastF1
    and predicted future points based on model
    """
    try:
        logger.info("Fetching driver standings endpoint...")
        
        # Get actual standings from FastF1
        actual_standings = get_2025_driver_standings_from_fastf1()
        
        if not actual_standings:
            logger.warning("No standings data from FastF1, using Ergast fallback")
            ergast_data = ergast_standings("2025")
            drivers = ergast_data.get("drivers", [])
            
            # Convert to expected format
            result_data = []
            for driver in drivers:
                result_data.append({
                    "position": driver.get("position"),
                    "driverName": driver.get("driver_name"),
                    "teamName": driver.get("constructor"),
                    "points": int(driver.get("points", 0)),
                    "predictedPoints": int(driver.get("points", 0)) + predict_driver_points_for_future_races(driver.get("driver_code"), driver.get("points", 0), 5),
                    "headshotUrl": None,
                    "teamColor": "#1e3a8a"  # Default color
                })
            
            return jsonify({
                "success": True,
                "source": "Ergast (fallback)",
                "data": result_data
            })
        
        # Calculate predicted points for each driver (next 5 races)
        result_data = []
        for driver in actual_standings:
            predicted_additional = predict_driver_points_for_future_races(
                driver.get("code"),
                driver.get("points", 0),
                num_races=5
            )
            predicted_total = driver.get("points", 0) + predicted_additional
            
            result_data.append({
                "position": driver.get("position"),
                "driverName": driver.get("name"),
                "teamName": driver.get("team"),
                "points": int(driver.get("points", 0)),
                "predictedPoints": int(predicted_total),
                "headshotUrl": None,
                "teamColor": "#1e3a8a"  # TODO: Add team color mapping
            })
        
        logger.info(f"✓ Driver standings complete: {len(result_data)} drivers")
        return jsonify({
            "success": True,
            "source": "FastF1",
            "data": result_data
        })
        
    except Exception as e:
        logger.error(f"Error in driver_standings endpoint: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/api/constructor-standings", methods=["GET"])
def api_constructor_standings():
    """
    Get 2025 constructor standings with actual points from Ergast
    and predicted future points based on aggregated driver predictions
    """
    try:
        logger.info("Fetching constructor standings endpoint...")
        
        # Get actual standings from Ergast
        actual_standings = get_2025_constructor_standings_from_ergast()
        
        if not actual_standings:
            logger.warning("No constructor standings data from Ergast")
            return jsonify({
                "success": False,
                "error": "No constructor standings available"
            }), 404
        
        # Calculate predicted points for each constructor
        result_data = []
        for constructor in actual_standings:
            constructor_name = constructor.get("name", "")
            predicted_additional = predict_constructor_points_for_future_races(
                constructor_name,
                constructor.get("points", 0),
                num_races=5
            )
            predicted_total = constructor.get("points", 0) + predicted_additional
            
            result_data.append({
                "position": constructor.get("position"),
                "constructorName": constructor_name,
                "points": int(constructor.get("points", 0)),
                "predictedPoints": int(predicted_total),
                "wins": int(constructor.get("wins", 0)),
                "teamColor": "#1e3a8a"  # TODO: Add team color mapping
            })
        
        logger.info(f"✓ Constructor standings complete: {len(result_data)} constructors")
        return jsonify({
            "success": True,
            "source": "Ergast",
            "data": result_data
        })
        
    except Exception as e:
        logger.error(f"Error in constructor_standings endpoint: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/api/standings", methods=["GET"])
def standings():
    """Get driver and constructor standings from Ergast"""
    season = request.args.get("season", "current")
    return jsonify({"success": True, "standings": ergast_standings(season)})


@app.route("/api/latest-race-circuit", methods=["GET"])
def latest_race_circuit():
    """Get latest completed race circuit data with top 6 drivers"""
    try:
        circuit_data = get_latest_race_circuit_data()
        
        if not circuit_data:
            return jsonify({
                "success": False,
                "error": "No race circuit data available"
            }), 404
        
        return jsonify({
            "success": True,
            "data": circuit_data
        })
        
    except Exception as e:
        logger.error(f"Error in latest_race_circuit endpoint: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# Cache for qualifying session to avoid reloading
_cached_qualifying_session = None
_cached_qualifying_timestamp = None

def get_cached_qualifying_session():
    """Load and cache the latest qualifying session with real FastF1 data"""
    global _cached_qualifying_session, _cached_qualifying_timestamp
    import time
    from datetime import datetime
    
    current_time = time.time()
    # Refresh cache every 30 minutes
    if _cached_qualifying_session is not None and (current_time - _cached_qualifying_timestamp) < 1800:
        logger.debug("Returning cached qualifying session")
        return _cached_qualifying_session
    
    try:
        logger.info("Loading latest qualifying session from FastF1...")
        
        # Prioritize 2025, then 2024
        years_to_try = [2025, 2024, 2023]
        
        for year in years_to_try:
            try:
                logger.info(f"Attempting to load qualifying sessions for year {year}...")
                schedule = fastf1.get_event_schedule(year)
                schedule['EventDate'] = pd.to_datetime(schedule['EventDate'])
                today = pd.to_datetime(datetime.now().date())
                
                # Get completed events (past)
                past_events = schedule[schedule['EventDate'] <= today]
                past_events = past_events.sort_values('EventDate', ascending=False)
                
                if len(past_events) == 0:
                    logger.debug(f"No completed events found for {year}")
                    continue
                
                logger.info(f"Found {len(past_events)} completed events for {year}")
                
                # Try to load most recent qualifying sessions
                for idx, event in past_events.iterrows():
                    event_name = event.get('EventName') or event.get('Event')
                    round_num = event.get('RoundNumber')
                    event_date = event.get('EventDate')
                    
                    try:
                        logger.info(f"Attempting to load {event_name} (Round {round_num}) {year} qualifying... ({event_date})")
                        
                        # Load qualifying session with telemetry
                        session = fastf1.get_session(year, round_num, 'Q')
                        session.load(telemetry=True, weather=False)
                        
                        # Check if we have results and telemetry
                        if session.results is not None and len(session.results) > 0:
                            # Verify we can get telemetry from at least one driver
                            try:
                                test_driver = session.results.iloc[0]['Abbreviation']
                                test_laps = session.laps.pick_drivers([test_driver])
                                if not test_laps.empty:
                                    logger.info(f"✓ Successfully loaded {event_name} {year} qualifying session with {len(session.results)} drivers and telemetry")
                                    _cached_qualifying_session = session
                                    _cached_qualifying_timestamp = current_time
                                    return session
                            except Exception as e:
                                logger.debug(f"Telemetry check failed for {event_name}: {str(e)[:50]}")
                                continue
                        else:
                            logger.debug(f"No results for {event_name}")
                            continue
                            
                    except Exception as e:
                        logger.debug(f"Could not load {event_name}: {str(e)[:100]}")
                        continue
                        
            except Exception as e:
                logger.debug(f"Error loading {year} schedule: {str(e)[:100]}")
                continue
        
        logger.error("Could not load any qualifying session from FastF1 for years 2025, 2024, 2023")
        return None
        
    except Exception as e:
        logger.error(f"Error loading qualifying session: {str(e)[:200]}")
        return None


@app.route('/api/latest-qualifying-session', methods=['GET'])
def latest_qualifying_session_data():
    """Returns top 6 drivers from latest qualifying session with metadata"""
    try:
        # Production: serve from Supabase cache only (prevent timeout)
        if config.FLASK_ENV == "production":
            try:
                logger.info("✓ Checking Supabase for cached qualifying data...")
                qualifying_cache = get_qualifying_cache(config)
                
                # Get the latest cached qualifying data
                cached_data = qualifying_cache.get_latest_cached_qualifying()
                
                if cached_data:
                    logger.info("✓ Found cached qualifying data in Supabase")
                    return jsonify({
                        "success": True,
                        "data": cached_data[:6] if isinstance(cached_data, list) else cached_data,
                        "source": "supabase_cache"
                    }), 200
            except Exception as e:
                logger.debug(f"Supabase lookup failed: {e}")
            
            # No cache found in production
            logger.warning("⚠️  No cached latest qualifying session in production")
            return jsonify({
                "success": False,
                "error": "Qualifying data not yet cached",
                "message": "Run: python scripts/cache_qualifying.py",
                "source": "cache_miss"
            }), 404
        
        # Development: allow fresh load (no timeout risk locally)
        logger.info("Development mode: loading fresh qualifying session")
        session = get_cached_qualifying_session()
        if session is None:
            return jsonify({
                "success": False,
                "error": "Could not load qualifying session from FastF1"
            }), 404
        
        # Get top 6 drivers by qualifying position
        top6 = session.results.head(6)
        
        drivers_data = []
        for idx, row in top6.iterrows():
            drivers_data.append({
                "code": row['Abbreviation'],
                "name": row['FullName'],
                "team": row['TeamName'],
                "grid_position": int(row['GridPosition']) if pd.notna(row['GridPosition']) else idx + 1,
                "quali_time": str(row.get('Q3', row.get('Q2', row.get('Q1', 'N/A')))),
            })
        
        return jsonify({
            "success": True,
            "session": f"Round {int(session.event['RoundNumber'])} - {session.event['EventName']}",
            "date": str(session.event['EventDate']).split(' ')[0] if pd.notna(session.event['EventDate']) else 'N/A',
            "drivers": drivers_data
        }), 200
    
    except Exception as e:
        logger.error(f"Error in latest_qualifying_session endpoint: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/driver-telemetry', methods=['POST'])
def driver_telemetry_data():
    """Returns telemetry data for a specific driver from latest qualifying"""
    try:
        data = request.get_json()
        driver_code = data.get('driver_code', '').upper()
        
        if not driver_code:
            return jsonify({
                "success": False,
                "error": "driver_code required"
            }), 400
        
        # Production: serve from Supabase cache only (prevent timeout)
        if config.FLASK_ENV == "production":
            try:
                logger.info(f"✓ Checking Supabase for cached telemetry for {driver_code}...")
                qualifying_cache = get_qualifying_cache(config)
                cached_data = qualifying_cache.get_latest_cached_qualifying()
                
                if cached_data and isinstance(cached_data, list):
                    # Find driver in cached data
                    driver_data = next((d for d in cached_data if d.get("code") == driver_code), None)
                    if driver_data:
                        logger.info(f"✓ Found cached telemetry for {driver_code}")
                        return jsonify({
                            "success": True,
                            "driver_data": driver_data,
                            "source": "supabase_cache"
                        }), 200
            except Exception as e:
                logger.debug(f"Supabase lookup failed: {e}")
            
            # No cache found in production
            logger.warning(f"⚠️  No cached qualifying telemetry for {driver_code} in production")
            return jsonify({
                "success": False,
                "error": "Qualifying data not yet cached",
                "message": "Run: python scripts/cache_qualifying.py",
                "source": "cache_miss"
            }), 404
        
        # Development: allow fresh load (no timeout risk locally)
        logger.info(f"Development mode: loading fresh qualifying for {driver_code}")
        session = get_cached_qualifying_session()
        if session is None:
            return jsonify({
                "success": False,
                "error": "Could not load qualifying session from FastF1"
            }), 404
        
        # Get driver's laps
        driver_laps = session.laps.pick_driver(driver_code)
        if driver_laps is None or len(driver_laps) == 0:
            return jsonify({
                "success": False,
                "error": f"No laps found for driver {driver_code}"
            }), 404
        
        # Get fastest lap
        fastest_lap = driver_laps.pick_fastest()
        if fastest_lap is None:
            return jsonify({
                "success": False,
                "error": f"No fastest lap for driver {driver_code}"
            }), 404
        
        # Get telemetry
        telemetry = fastest_lap.get_telemetry()
        
        if telemetry is None or len(telemetry) == 0:
            return jsonify({
                "success": False,
                "error": f"No telemetry data for driver {driver_code}"
            }), 404
        
        logger.info(f"✓ Retrieved telemetry for {driver_code}: {len(telemetry)} data points")
        
        # Extract X, Y, Speed
        telemetry_data = {
            "x": telemetry['X'].tolist(),
            "y": telemetry['Y'].tolist(),
            "speed": telemetry['Speed'].tolist(),
            "throttle": telemetry['Throttle'].tolist() if 'Throttle' in telemetry.columns else [],
            "brake": telemetry['Brake'].tolist() if 'Brake' in telemetry.columns else [],
            "gear": telemetry['Gear'].tolist() if 'Gear' in telemetry.columns else [],
        }
        
        # Get driver info from session results
        driver_results = session.results[session.results['Abbreviation'] == driver_code]
        if len(driver_results) == 0:
            return jsonify({
                "success": False,
                "error": f"Driver {driver_code} not found in results"
            }), 404
        
        driver_info = driver_results.iloc[0]
        
        return jsonify({
            "success": True,
            "driver_code": driver_code,
            "driver_name": driver_info['FullName'],
            "team": driver_info['TeamName'],
            "grid_position": int(driver_info['GridPosition']) if pd.notna(driver_info['GridPosition']) else 0,
            "telemetry": telemetry_data,
            "stats": {
                "max_speed": float(telemetry['Speed'].max()),
                "min_speed": float(telemetry['Speed'].min()),
                "avg_speed": float(telemetry['Speed'].mean()),
                "data_points": len(telemetry)
            }
        }), 200
    
    except Exception as e:
        logger.error(f"Error in driver_telemetry endpoint: {str(e)[:200]}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/qualifying-circuit-telemetry')
def qualifying_circuit_telemetry():
    """
    Returns comprehensive telemetry data for top 6 qualifying drivers (cache-first)
    Query params: year, event (optional, defaults to latest)
    
    CACHING STRATEGY:
    - Latest session: Always from Supabase cache (30-second timeout protection)
    - Specific session: From cache if available, else loads fresh (dev only)
    - Production: Cache-only, no fresh loads (prevents worker timeout)
    """
    try:
        year = request.args.get('year', type=int)
        event = request.args.get('event', type=str)
        
        logger.info(f"Loading qualifying telemetry for {year} {event}")
        
        # Check Supabase cache first (all cases)
        if not year and not event:
            # Latest session - check Supabase
            try:
                logger.info("✓ Checking Supabase for cached qualifying telemetry")
                qualifying_cache = get_qualifying_cache(config)
                cached_result = qualifying_cache.get_latest_cached_qualifying()
                
                if cached_result is not None:
                    logger.info("✓ Returning cached qualifying telemetry from Supabase")
                    return jsonify({
                        "success": True,
                        "drivers": cached_result if isinstance(cached_result, list) else [],
                        "source": "supabase_cache"
                    }), 200
            except Exception as e:
                logger.debug(f"Supabase cache check failed: {e}")
            
            # Cache miss on production: return helpful message instead of timeout
            if config.FLASK_ENV == "production":
                logger.warning("⚠️  No cached qualifying telemetry found in production")
                return jsonify({
                    "success": False,
                    "error": "Qualifying data not yet cached",
                    "message": "Qualifying telemetry cache is populated after each race weekend using: python scripts/cache_qualifying.py",
                    "source": "cache_miss"
                }), 404
        
        # Load session only in development (specific race queries allowed)
        if year and event:
            if config.FLASK_ENV == "production":
                # Production: only serve cached data to prevent timeout
                logger.info(f"Production mode: checking cache for {year} {event}")
                try:
                    qualifying_cache = get_qualifying_cache(config)
                    cached = qualifying_cache.get_latest_cached_qualifying()
                    if cached:
                        logger.info(f"✓ Found cached session for {event}")
                        return jsonify({
                            "success": True,
                            "drivers": cached if isinstance(cached, list) else [],
                            "source": "supabase_cache"
                        }), 200
                except Exception as e:
                    logger.warning(f"Supabase cache check failed: {e}")
                
                # No cache found in production
                logger.warning(f"⚠️  No cache found for {year} {event} in production")
                return jsonify({
                    "success": False,
                    "error": "Qualifying data not cached",
                    "message": f"Qualifying telemetry for {event} is not yet cached. Please run: python scripts/cache_qualifying.py {year} '{event}'",
                    "source": "cache_miss"
                }), 404
            else:
                # Development: allow fresh loads (no timeout risk locally)
                logger.info(f"Development mode: loading fresh data for {year} {event}")
                session = fastf1.get_session(year, event, 'Q')
        else:
            session = get_cached_qualifying_session()
        
        if session is None:
            return jsonify({
                "success": False,
                "error": "Could not load qualifying session"
            }), 404
        
        session.load(laps=True, telemetry=True, weather=False)
        
        # Get top 6 drivers
        results = session.results.sort_values('Position')
        top_6_drivers = results.head(6)['Abbreviation'].tolist()
        
        drivers_data = []
        
        for idx, driver_code in enumerate(top_6_drivers):
            try:
                # Get driver info
                driver_row = session.results[session.results['Abbreviation'] == driver_code].iloc[0]
                
                # Get fastest lap
                driver_laps = session.laps.pick_drivers([driver_code])
                if driver_laps.empty:
                    continue
                
                fastest_lap = driver_laps.pick_fastest()
                telemetry = fastest_lap.get_telemetry()
                
                if telemetry.empty:
                    continue
                
                # Extract comprehensive telemetry data
                lap_time = fastest_lap['LapTime'].total_seconds() if pd.notna(fastest_lap['LapTime']) else 0
                
                # Sector times
                s1 = fastest_lap.get('Sector1Time', np.nan)
                s2 = fastest_lap.get('Sector2Time', np.nan)
                s3 = fastest_lap.get('Sector3Time', np.nan)
                
                sector1_s = s1.total_seconds() if pd.notna(s1) else 0
                sector2_s = s2.total_seconds() if pd.notna(s2) else 0
                sector3_s = s3.total_seconds() if pd.notna(s3) else 0
                
                # Acceleration data
                max_accel = telemetry['Acceleration'].max() if 'Acceleration' in telemetry.columns else 0
                avg_accel = telemetry['Acceleration'].mean() if 'Acceleration' in telemetry.columns else 0
                
                # Braking analysis
                brake_events = 0
                if 'Brake' in telemetry.columns:
                    brake_events = (telemetry['Brake'] > 0).sum()
                
                # Circuit trace data - convert to native Python types (single copy)
                circuit_trace = {
                    "x": telemetry['X'].astype(float).tolist(),
                    "y": telemetry['Y'].astype(float).tolist(),
                    "speed": telemetry['Speed'].astype(float).tolist()
                }
                
                # Available telemetry channels
                channels = {
                    "position": ["X", "Y"],
                    "performance": ["Speed", "Throttle", "Brake"],
                    "power": ["RPM", "Gear", "DRS"],
                    "motion": ["Acceleration", "nGear", "nLatG"]
                }
                
                driver_data = {
                    "code": str(driver_code),
                    "name": str(driver_row['FullName']),
                    "team": str(driver_row['TeamName']),
                    "qualifying_position": int(driver_row['Position']),
                    "telemetry_stats": {
                        "lap_time_s": float(round(lap_time, 3)),
                        "top_speed_kmh": float(round(telemetry['Speed'].max(), 1)),
                        "avg_speed_kmh": float(round(telemetry['Speed'].mean(), 1)),
                        "min_speed_kmh": float(round(telemetry['Speed'].min(), 1)),
                        "max_acceleration_g": float(round(max_accel, 2)),
                        "avg_acceleration_g": float(round(avg_accel, 2)),
                        "sector1_s": float(round(sector1_s, 3)),
                        "sector2_s": float(round(sector2_s, 3)),
                        "sector3_s": float(round(sector3_s, 3)),
                        "total_data_points": int(len(telemetry)),
                        "brake_zones": int(brake_events),
                    },
                    "circuit_trace": circuit_trace,
                    "available_channels": [str(c) for c in telemetry.columns.tolist()]
                }
                
                drivers_data.append(driver_data)
                logger.info(f"  ✓ {driver_code}: {len(telemetry)} telemetry points")
                
            except Exception as e:
                logger.error(f"  ❌ Error processing {driver_code}: {str(e)[:100]}")
                continue
        
        if not drivers_data:
            return jsonify({
                "success": False,
                "error": "No telemetry data could be extracted for top 6 drivers"
            }), 404
        
        session_info = {
            "year": int(session.event.get('Year', session.event.get('year', 2024))),
            "event": session.event.get('EventName', 'Unknown'),
            "circuit": session.event.get('CircuitName', 'Unknown'),
            "round": int(session.event.get('RoundNumber', 0))
        }
        
        result = {
            "session_info": session_info,
            "drivers": drivers_data
        }
        
        # Save to cache if this is the latest session query
        if not year and not event:
            cache = get_file_cache()
            cache.set(CACHE_KEYS["QUALIFYING_TELEMETRY"], result)
            logger.info("✓ Cached qualifying telemetry")
        
        return jsonify({
            "success": True,
            "session_info": session_info,
            "drivers": drivers_data,
            "source": "fastf1_fresh"
        }), 200
    
    except Exception as e:
        logger.error(f"Error in qualifying_circuit_telemetry: {str(e)[:200]}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Try to return stale cache if available
        try:
            cache = get_file_cache()
            cache_key = CACHE_KEYS["QUALIFYING_TELEMETRY"]
            cache_file = cache._get_cache_file(cache_key)
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    stale_data = json.load(f)
                logger.warning("Returning stale qualifying telemetry cache due to error")
                return jsonify({
                    "success": True,
                    "session_info": stale_data["session_info"],
                    "drivers": stale_data["drivers"],
                    "source": "stale_cache",
                    "error_note": "Fresh data unavailable, using cached data"
                }), 200
        except Exception as cache_err:
            logger.error(f"Could not retrieve stale cache: {cache_err}")
        
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/driver-circuit-details/<driver_code>', methods=['GET'])
def driver_circuit_details(driver_code):
    """
    Returns detailed telemetry analysis for a specific driver
    Query params: year, event (optional)
    """
    try:
        year = request.args.get('year', type=int)
        event = request.args.get('event', type=str)
        driver_code = driver_code.upper()
        
        # Load session
        if year and event:
            session = fastf1.get_session(year, event, 'Q')
        else:
            session = get_cached_qualifying_session()
        
        if session is None:
            return jsonify({
                "success": False,
                "error": "Could not load qualifying session"
            }), 404
        
        session.load(laps=True, telemetry=True, weather=False)
        
        # Get driver data
        driver_row = session.results[session.results['Abbreviation'] == driver_code]
        if driver_row.empty:
            return jsonify({
                "success": False,
                "error": f"Driver {driver_code} not found"
            }), 404
        
        driver_row = driver_row.iloc[0]
        
        # Get laps and fastest lap
        driver_laps = session.laps.pick_drivers([driver_code])
        if driver_laps.empty:
            return jsonify({
                "success": False,
                "error": f"No laps for driver {driver_code}"
            }), 404
        
        fastest_lap = driver_laps.pick_fastest()
        telemetry = fastest_lap.get_telemetry()
        
        if telemetry.empty:
            return jsonify({
                "success": False,
                "error": f"No telemetry for driver {driver_code}"
            }), 404
        
        # Detailed analysis
        detailed_stats = {
            "driver": {
                "code": driver_code,
                "name": driver_row['FullName'],
                "team": driver_row['TeamName'],
                "qualifying_position": int(driver_row['Position']),
            },
            "lap_analysis": {
                "lap_time_s": fastest_lap['LapTime'].total_seconds() if pd.notna(fastest_lap['LapTime']) else 0,
                "sector1_s": fastest_lap['Sector1Time'].total_seconds() if pd.notna(fastest_lap['Sector1Time']) else 0,
                "sector2_s": fastest_lap['Sector2Time'].total_seconds() if pd.notna(fastest_lap['Sector2Time']) else 0,
                "sector3_s": fastest_lap['Sector3Time'].total_seconds() if pd.notna(fastest_lap['Sector3Time']) else 0,
            },
            "speed_analysis": {
                "top_speed_kmh": round(telemetry['Speed'].max(), 1),
                "avg_speed_kmh": round(telemetry['Speed'].mean(), 1),
                "min_speed_kmh": round(telemetry['Speed'].min(), 1),
                "speed_std_dev": round(telemetry['Speed'].std(), 1),
            },
            "acceleration_analysis": {
                "max_g": round(telemetry['Acceleration'].max(), 2) if 'Acceleration' in telemetry.columns else 0,
                "avg_g": round(telemetry['Acceleration'].mean(), 2) if 'Acceleration' in telemetry.columns else 0,
                "min_g": round(telemetry['Acceleration'].min(), 2) if 'Acceleration' in telemetry.columns else 0,
            },
            "braking_analysis": {
                "brake_zones": (telemetry['Brake'] > 0).sum() if 'Brake' in telemetry.columns else 0,
                "avg_brake_pressure": round(telemetry['Brake'].mean() * 100, 1) if 'Brake' in telemetry.columns else 0,
                "max_brake_pressure": round(telemetry['Brake'].max() * 100, 1) if 'Brake' in telemetry.columns else 0,
            },
            "throttle_analysis": {
                "avg_throttle": round(telemetry['Throttle'].mean() * 100, 1) if 'Throttle' in telemetry.columns else 0,
                "max_throttle": round(telemetry['Throttle'].max() * 100, 1) if 'Throttle' in telemetry.columns else 0,
                "full_throttle_time_pct": round((telemetry['Throttle'] == 1).sum() / len(telemetry) * 100, 1) if 'Throttle' in telemetry.columns else 0,
            },
            "circuit_trace": {
                "x": telemetry['X'].tolist(),
                "y": telemetry['Y'].tolist(),
                "speed": telemetry['Speed'].tolist()
            },
            "available_channels": [col for col in telemetry.columns if col not in ['X', 'Y']],
            "data_points": len(telemetry)
        }
        
        return jsonify({
            "success": True,
            "data": detailed_stats
        }), 200
    
    except Exception as e:
        logger.error(f"Error in driver_circuit_details: {str(e)[:200]}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ========== MLflow Model Registry Endpoints ==========

@app.route("/api/model-registry", methods=["GET"])
def model_registry():
    """
    Get complete model registry from MLflow
    Shows all registered models, versions, and metrics
    """
    try:
        registry = get_model_registry()
        return jsonify({
            "success": True,
            "data": registry
        }), 200
    except Exception as e:
        logger.error(f"Error fetching model registry: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/model-metrics", methods=["GET"])
def model_metrics():
    """
    Get prediction accuracy metrics and model performance
    Includes overall accuracy, recent accuracy, and trends
    """
    try:
        accuracy = get_prediction_accuracy()
        return jsonify({
            "success": True,
            "data": accuracy,
            "timestamp": datetime.now().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Error fetching model metrics: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/mlflow-status", methods=["GET"])
def mlflow_status():
    """
    Health check for MLflow and model status
    Returns current model versions and MLflow UI URL
    """
    try:
        registry = get_model_registry()
        return jsonify({
            "success": True,
            "status": "healthy",
            "mlflow_tracking_uri": config.MLFLOW_TRACKING_URI,
            "mlflow_ui_command": f"mlflow ui --backend-store-uri {config.MLFLOW_TRACKING_URI}",
            "models_registered": len(registry.get("models", [])),
            "total_predictions_logged": registry.get("prediction_stats", {}).get("total_predictions", 0),
            "current_accuracy": registry.get("prediction_stats", {}).get("overall_accuracy", 0),
            "timestamp": datetime.now().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Error in mlflow_status: {e}")
        return jsonify({
            "success": False,
            "status": "error",
            "error": str(e)
        }), 500


# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    # Start background scheduler for auto-caching qualifying data
    logger.info("Starting background scheduler for automatic qualifying cache...")
    start_scheduler()
    
    try:
        # Run with Flask development server (for local development only)
        # In production, use: gunicorn -w 4 -b 0.0.0.0:5000 api:app
        app.run(
            debug=config.DEBUG,
            host=config.HOST,
            port=config.PORT
        )
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        stop_scheduler()

