"""
F1 Prediction API Server
Flask REST API to serve race predictions to the frontend

VERSION: 2.1.1 (Dynamic year fix + Python 3.12 compatibility)
DEPLOYED: 2026-03-24
FIXES: Use current year instead of hardcoded 2025, Python 3.12.5 with pyarrow wheels
"""

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import logging
import requests
from urllib.parse import quote_plus, urlparse
import time
from pathlib import Path
import hashlib
import fastf1
from datetime import datetime, timedelta
import uuid
import json
import traceback
import threading

# Import configuration
from utils.config import config, ensure_directories, print_config

# Import new feature store for efficient feature retrieval
from services.feature_store import get_feature_store, FeatureStore

# Import Supabase prediction logger + telemetry caches
from database.database_v2 import (
    get_prediction_logger,
    PredictionLogger,
    get_qualifying_cache,
    get_race_telemetry_cache,
)

# Optional Supabase-backed cache for standings snapshots
from database.standings_cache import try_create_cache

# Import file-based caching for expensive queries
from utils.file_cache import get_file_cache, CACHE_KEYS, CACHE_TTL

# Initialize Flask app
app = Flask(__name__)

# Configure CORS with explicit settings for Vercel frontend
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "https://fonewinner.vercel.app",
            "http://localhost:5173",
            "http://localhost:3000"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": False
    }
})

# Add cache-control headers for API responses (CORS is handled by Flask-CORS above)
@app.after_request
def add_header(response):
    """Add cache-control headers for API responses"""
    if '/api/' in str(request.path):
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response

# Error handlers (CORS headers added automatically by Flask-CORS)
@app.errorhandler(500)
def handle_500(e):
    return jsonify({"success": False, "error": "Internal server error"}), 500

@app.errorhandler(404)
def handle_404(e):
    return jsonify({"success": False, "error": "Not found"}), 404

@app.errorhandler(Exception)
def handle_exception(e):
    logger.exception("Unhandled exception")
    return jsonify({"success": False, "error": "Internal server error"}), 500

# Configure logging based on config
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _internal_error(context: str, e: Exception):
    """Log exception details server-side; return a safe generic error to clients."""
    error_id = uuid.uuid4().hex[:10]
    logger.error(f"{context} [error_id={error_id}]: {e}")
    logger.error(traceback.format_exc())
    return jsonify({"success": False, "error": "Internal server error", "error_id": error_id}), 500


def _get_bearer_or_api_key() -> str:
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return (
        request.headers.get("X-API-Key")
        or request.headers.get("x-api-key")
        or ""
    ).strip()


def _require_write_key():
    expected = getattr(config, "API_WRITE_KEY", None)
    if not expected:
        if config.FLASK_ENV == "production":
            logger.error("API_WRITE_KEY is not set in production")
            return jsonify({"success": False, "error": "Server misconfigured"}), 500
        logger.warning("API_WRITE_KEY not set; allowing write in non-production")
        return None

    provided = _get_bearer_or_api_key()
    if not provided or provided != expected:
        return jsonify({"success": False, "error": "Unauthorized"}), 401

    return None


def _encoder_classes(enc_obj):
    """Return a list of known classes for a LabelEncoder-like object or compat list."""
    try:
        if enc_obj is None:
            return []
        if hasattr(enc_obj, "classes_"):
            return [str(c) for c in list(enc_obj.classes_)]
        if isinstance(enc_obj, (list, tuple)):
            return [str(c) for c in list(enc_obj)]
    except Exception:
        pass
    return []


def _normalize_driver_code(value: str, driver_classes: list[str]) -> str:
    if value is None:
        return ""
    v = str(value).strip()
    if not v:
        return v
    v_up = v.upper()
    if v_up in driver_classes:
        return v_up
    return v_up


def _normalize_team_name(value: str, team_classes: list[str]) -> str:
    """Normalize common team aliases to the training labels."""
    if value is None:
        return ""
    v = str(value).strip()
    if not v:
        return v
    if v in team_classes:
        return v

    v_low = v.lower()
    alias = {
        "red bull": "Red Bull Racing",
        "red bull racing": "Red Bull Racing",
        "racing bulls": "RB",
        "rb f1 team": "RB",
        "visa cash app rb": "RB",
        "visa cash app rb f1 team": "RB",
        "alpine f1 team": "Alpine",
        "haas": "Haas F1 Team",
        "haas f1": "Haas F1 Team",
        "kick sauber": "Kick Sauber",
        "sauber": "Sauber",
        # 2026+ constructor branding (map to closest known training label)
        "audi": "Sauber",
        "alphatauri": "AlphaTauri",
    }

    if v_low in alias and alias[v_low] in team_classes:
        return alias[v_low]

    # Best-effort contains match (e.g. "Red Bull" -> "Red Bull Racing")
    matches = [c for c in team_classes if v_low in c.lower()]
    if matches:
        matches.sort(key=lambda s: (len(s), s))
        return matches[0]

    return v


def _normalize_circuit_label(circuit: str, event: str, circuit_classes: list[str]) -> str:
    """Map short race names (e.g. 'Chinese Grand Prix') to training circuit labels.

    Training data uses official event titles like 'FORMULA 1 ... CHINESE GRAND PRIX 2024'.
    """
    if circuit is None and event is None:
        return ""

    # Prefer explicit circuit param; fall back to event.
    raw = str(circuit or event).strip()
    if not raw:
        return raw
    if raw in circuit_classes:
        return raw

    candidates = [raw]
    if event and str(event).strip() and str(event).strip() not in candidates:
        candidates.append(str(event).strip())

    def year_key(label: str) -> int:
        import re

        years = [int(m.group(0)) for m in re.finditer(r"\b(19|20)\d{2}\b", label)]
        return max(years) if years else -1

    best = None
    best_score = None
    for cand in candidates:
        clow = cand.lower()
        matches = [c for c in circuit_classes if clow in c.lower()]
        if not matches and "grand prix" in clow:
            # try just the '{X} grand prix' substring
            gp = clow[clow.find("grand prix") - 40 :].strip()
            matches = [c for c in circuit_classes if gp in c.lower()]
        if not matches:
            continue

        # Prefer the most recent year; tie-break on shortest label.
        matches.sort(key=lambda s: (-year_key(s), len(s), s))
        pick = matches[0]
        score = (-year_key(pick), len(pick))
        if best is None or score < best_score:
            best = pick
            best_score = score

    return best or raw

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

# Compat (Option 2) artifacts
META_COMPAT_FILE = str(getattr(config, "META_COMPAT_FILE", config.META_FILE))
MODEL_WIN_JSON_FILE = str(getattr(config, "MODEL_WIN_JSON_FILE", config.MODEL_WIN_FILE))
MODEL_POD_JSON_FILE = str(getattr(config, "MODEL_POD_JSON_FILE", config.MODEL_POD_FILE))

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
    '44': 'HAM', '12': 'BEA',  # Hamilton=44, Bearman=12
    # Legacy/fallback drivers
    '5': 'VET', '22': 'LAT',
    '3': 'RIC', '77': 'BOT', '20': 'MAG', '11': 'PER'
}

# Configure FastF1 cache
fastf1.Cache.enable_cache(str(config.FASTF1_CACHE_DIR))
logger.info(f"FastF1 cache enabled at: {config.FASTF1_CACHE_DIR}")

# =============================================================================
# LAZY-LOADED INFERENCE ASSETS (Models + Encoders + FeatureStore)
# =============================================================================
# Keep startup fast: don't load large model files or Parquet snapshots at import
# time. Only initialize these when an inference endpoint is actually called.


class InferenceAssetsUnavailableError(RuntimeError):
    """Raised when inference assets (models/encoders/feature store) cannot be loaded."""


meta = None
encoders = {}
feature_cols = []
model_win = None
model_pod = None
feature_store = None

_inference_assets_loaded = False
_inference_assets_loaded_at = None
_inference_assets_load_error = None
_inference_assets_lock = threading.Lock()


def _load_meta_compat(path: str):
    try:
        p = Path(path)
        if p.exists() and p.suffix.lower() == ".json":
            payload = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and "feature_cols" in payload and "encoders" in payload:
                return payload
    except Exception:
        pass
    return None


def _load_meta_any() -> dict:
    compat = _load_meta_compat(META_COMPAT_FILE)
    if compat is not None:
        logger.info("✓ Loaded metadata from compat JSON")
        return compat

    import joblib  # type: ignore

    meta_obj = joblib.load(META_FILE)
    if not isinstance(meta_obj, dict):
        raise TypeError(f"metadata.joblib is not a dict: {type(meta_obj)}")
    logger.info("✓ Loaded metadata from joblib")
    return meta_obj


def _load_xgb_classifier_any(native_path: str, joblib_path: str):
    """Prefer native XGBoost model file; fallback to joblib."""
    try:
        p = Path(native_path)
        if p.exists() and p.suffix.lower() in {".json", ".ubj"}:
            import xgboost as xgb

            clf = xgb.XGBClassifier()
            clf.load_model(str(p))
            logger.info(f"✓ Loaded XGBoost model from native file: {p.name}")
            return clf
    except Exception as e:
        logger.warning(f"⚠ Failed to load native XGBoost model ({native_path}): {e}")

    import joblib  # type: ignore

    obj = joblib.load(joblib_path)
    if isinstance(obj, dict) and "model" in obj:
        logger.info(f"✓ Loaded XGBoost model from joblib dict: {Path(joblib_path).name}")
        return obj["model"]
    logger.info(f"✓ Loaded XGBoost model from joblib: {Path(joblib_path).name}")
    return obj


def ensure_inference_assets_loaded():
    """Load inference-time dependencies (models + feature store) on-demand."""
    global meta
    global encoders
    global feature_cols
    global model_win
    global model_pod
    global feature_store
    global _inference_assets_loaded
    global _inference_assets_loaded_at
    global _inference_assets_load_error

    if _inference_assets_loaded:
        return

    with _inference_assets_lock:
        if _inference_assets_loaded:
            return

        meta_json = Path(META_COMPAT_FILE)
        meta_joblib = Path(META_FILE)
        if not (meta_json.exists() or meta_joblib.exists()):
            raise InferenceAssetsUnavailableError(
                "Model metadata not found (metadata_compat.json or metadata.joblib)"
            )
        if not (Path(MODEL_WIN_JSON_FILE).exists() or Path(MODEL_WIN_FILE).exists()):
            raise InferenceAssetsUnavailableError("Winner model file not found")
        if not (Path(MODEL_POD_JSON_FILE).exists() or Path(MODEL_POD_FILE).exists()):
            raise InferenceAssetsUnavailableError("Podium model file not found")

        try:
            meta_local = _load_meta_any()
            encoders_local = meta_local["encoders"]
            feature_cols_local = meta_local["feature_cols"]
            model_win_local = _load_xgb_classifier_any(MODEL_WIN_JSON_FILE, MODEL_WIN_FILE)
            model_pod_local = _load_xgb_classifier_any(MODEL_POD_JSON_FILE, MODEL_POD_FILE)
            feature_store_local = get_feature_store(config)

            meta = meta_local
            encoders = encoders_local
            feature_cols = feature_cols_local
            model_win = model_win_local
            model_pod = model_pod_local
            feature_store = feature_store_local

            _inference_assets_loaded = True
            _inference_assets_loaded_at = datetime.now().isoformat()
            _inference_assets_load_error = None
            logger.info("✓ Inference assets loaded (lazy)")
        except Exception as e:
            _inference_assets_loaded = False
            _inference_assets_load_error = str(e)
            logger.error(f"Inference asset load failed: {e}")
            logger.error(traceback.format_exc())
            raise InferenceAssetsUnavailableError("Inference assets failed to load") from e


# Initialize Supabase prediction logger (used by most endpoints)
prediction_logger = get_prediction_logger(config)
logger.info(f"✓ Prediction logger initialized (mode: {prediction_logger.mode})")

# Initialize standings cache (optional)
standings_cache = try_create_cache(config)
STANDINGS_CACHE_TTL = timedelta(days=365)
if standings_cache is not None:
    logger.info("✓ Standings cache enabled (Supabase)")
else:
    logger.info("Standings cache disabled (no Supabase configured)")

# Historical training dataset loading removed for production simplicity.
# We rely on qualifying + FeatureStore snapshots + Supabase pipeline tables.
# Keep an empty DataFrame with expected columns so legacy endpoints don't crash.
hist_data = pd.DataFrame(
    columns=[
        "race_key",
        "race_year",
        "event",
        "circuit",
        "event_date",
        "driver",
        "team",
        "qualifying_position",
        "finishing_position",
        "points",
        "EloRating",
        "RecentFormAvg",
        "CircuitHistoryAvg",
        "DriverExperienceScore",
        "TeamPerfScore",
    ]
)
logger.info("⚠ Historical training dataset disabled (qualifying-only production mode)")
logger.info("Inference models will load on-demand")

# ---------- helper: simple file cache ----------
def cache_path_for_key(prefix: str, key: str, ext: str):
    """Generate cache path for a given key"""
    h = hashlib.sha1(key.encode()).hexdigest()
    return IMG_CACHE / f"{prefix}_{h}.{ext}"

# ---------- helper: fetch image from Wikipedia (page image) ----------
def _wikipedia_page_image_url(entity_name: str, *, fallback_size: int = 600):
    """Return a representative image URL for a Wikipedia page.

    Tries exact title lookup first; falls back to a search query.
    Prefers `original` image when available, otherwise uses `thumbnail`.
    """

    headers = {
        # Wikimedia APIs may return 403 for generic/missing user agents.
        "User-Agent": os.getenv("WIKIMEDIA_USER_AGENT")
        or "f1-hackathon/1.0 (https://github.com; dev)",
        "Accept": "application/json",
    }

    def _query_title(title: str):
        if not title:
            return None

        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            headers=headers,
            params={
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "pageimages",
                # Ask for both; not all pages have `original`.
                "piprop": "original|thumbnail",
                "pithumbsize": int(fallback_size),
            },
            timeout=8,
        )
        r.raise_for_status()
        pages = (r.json() or {}).get("query", {}).get("pages", {})
        for _, page in (pages or {}).items():
            orig = (page or {}).get("original") or {}
            thumb = (page or {}).get("thumbnail") or {}
            return orig.get("source") or thumb.get("source")
        return None

    title = (entity_name or "").strip()
    if not title:
        return None

    # 1) Exact title lookup
    try:
        url = _query_title(title)
        if url:
            return url
    except Exception as e:
        logger.debug(f"Wikipedia image title lookup failed for '{title}': {e}")

    # 2) Search fallback
    try:
        s = requests.get(
            "https://en.wikipedia.org/w/api.php",
            headers=headers,
            params={
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": title,
                "srlimit": 1,
            },
            timeout=8,
        )
        s.raise_for_status()
        results = (s.json() or {}).get("query", {}).get("search", [])
        if results and isinstance(results, list):
            best_title = (results[0] or {}).get("title")
            if best_title:
                return _query_title(str(best_title))
    except Exception as e:
        logger.debug(f"Wikipedia image search failed for '{title}': {e}")

    return None


def fetch_wikipedia_image(entity_name: str, fallback_size=600):
    """
    Returns local path to cached image (jpg/png/webp/gif/svg) or None.
    Uses MediaWiki 'pageimages' and caches the downloaded asset.
    """
    key = (entity_name or "").strip()
    if not key:
        return None

    h = hashlib.sha1(key.encode()).hexdigest()

    # Return any existing cached file regardless of extension.
    for ext in ("jpg", "jpeg", "png", "webp", "gif", "svg"):
        existing = IMG_CACHE / f"wiki_{h}.{ext}"
        if existing.exists():
            return str(existing)

    img_url = None
    try:
        img_url = _wikipedia_page_image_url(key, fallback_size=int(fallback_size))
    except Exception:
        img_url = None

    if not img_url:
        return None

    allowed_ext = {"jpg", "jpeg", "png", "webp", "gif", "svg"}
    try:
        suffix = Path(urlparse(img_url).path).suffix.lower()
        ext = suffix.lstrip(".") if suffix else "jpg"
        if ext not in allowed_ext:
            ext = "jpg"
    except Exception:
        ext = "jpg"

    p = IMG_CACHE / f"wiki_{h}.{ext}"

    try:
        r = requests.get(
            img_url,
            timeout=12,
            stream=True,
            headers={
                "User-Agent": os.getenv("WIKIMEDIA_USER_AGENT")
                or "f1-hackathon/1.0 (https://github.com; dev)",
            },
        )
        r.raise_for_status()
        with open(p, "wb") as fh:
            for chunk in r.iter_content(1024 * 8):
                if chunk:
                    fh.write(chunk)
        return str(p)
    except Exception as e:
        logger.debug(f"Wikipedia image download failed for '{key}': {e}")

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
            year = datetime.now().year
        
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


def compute_basic_features(df):
    """
    🔥 CRITICAL FUNCTION: Computes all features dynamically from merged data
    
    This is THE KEY to making predictions realistic instead of all rookies.
    It calculates features using full historical context for each driver.
    
    Takes merged dataset (historical 2018-2024 + current race qualifying)
    Returns same dataset with computed features added.
    
    Features computed:
    - RecentFormAvg: Rolling average of last 5 races (finish position)
    - CircuitHistoryAvg: Career average finish position at this circuit
    - DriverExperienceScore: Normalized by total career races
    - TeamPerfScore: How good is the team this season?
    - EloRating: Elo rating (if available)
    """
    df = df.copy()

    # Defensive: allow qualifying-only datasets (no historical rows)
    if 'event_date' not in df.columns:
        df['event_date'] = pd.NaT
    else:
        df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')

    df = df.sort_values(['driver', 'race_year', 'event_date']).reset_index(drop=True)
    
    # Convert finishing position to numeric
    df['finishing_position_num'] = pd.to_numeric(df['finishing_position'], errors='coerce')
    median_finish = df['finishing_position_num'].median()
    if pd.isna(median_finish):
        # Qualifying-only inference has no finishing positions; use mid-pack default.
        median_finish = 10.0
    
    logger.debug(f"[compute_basic_features] Input: {len(df)} rows, {df['driver'].nunique()} drivers")
    
    # ═══════════════════════════════════════════════════════════════════
    # FEATURE 1: RecentFormAvg - Rolling average of last 5 races
    # ═══════════════════════════════════════════════════════════════════
    # This is the driver's recent form - how they've been finishing
    # Lower is better (1st place = 1, 20th place = 20)
    df['RecentFormAvg'] = (
        df.groupby('driver')['finishing_position_num']
          .transform(lambda x: x.shift(1)                    # Exclude current race
                                .rolling(5, min_periods=1)   # Last 5 races
                                .mean())
          .fillna(median_finish)
    )
    
    # ═══════════════════════════════════════════════════════════════════
    # FEATURE 2: CircuitHistoryAvg - Career average at this circuit
    # ═══════════════════════════════════════════════════════════════════
    # How has this driver done at this specific circuit historically?
    df['CircuitHistoryAvg'] = (
        df.groupby(['driver', 'circuit'])['finishing_position_num']
          .transform(lambda x: x.shift(1)                    # Exclude current race
                                .expanding(min_periods=1)    # All previous races
                                .mean())
          .fillna(median_finish)
    )
    
    # ═══════════════════════════════════════════════════════════════════
    # FEATURE 3: DriverExperienceScore - How many races have they done?
    # ═══════════════════════════════════════════════════════════════════
    # Normalized by max races in dataset (e.g., veteran = 1.0, rookie = 0.1)
    df['TotalRaces'] = df.groupby('driver').cumcount()
    max_races = df['TotalRaces'].max() if df['TotalRaces'].notna().any() else 1
    df['DriverExperienceScore'] = (df['TotalRaces'] / max(1, max_races)).fillna(0)
    
    # ═══════════════════════════════════════════════════════════════════
    # FEATURE 4: TeamPerfScore - How good is the team this year?
    # ═══════════════════════════════════════════════════════════════════
    # Calculate average finishing position for each team in each year
    team_perf_yearly = (
        df.dropna(subset=['finishing_position_num'])
          .groupby(['race_year', 'team'])['finishing_position_num']
          .mean()
          .reset_index()
          .rename(columns={'finishing_position_num': 'team_avg_finish'})
    )
    
    df = df.merge(team_perf_yearly, on=['race_year', 'team'], how='left')
    
    # Normalize: best team = 1.0, worst team = 0.0
    if df['team_avg_finish'].notna().any():
        df['TeamPerfScore'] = (
            df.groupby('race_year')['team_avg_finish']
              .transform(lambda x: (x.max() - x) / (x.max() - x.min() + 0.001)
                         if x.max() != x.min() else 0.5)
        )
    else:
        df['TeamPerfScore'] = 0.5
    
    # ═══════════════════════════════════════════════════════════════════
    # FEATURE 5: EloRating - If available, else default 1500
    # ═══════════════════════════════════════════════════════════════════
    if 'elo_before' in df.columns:
        df['EloRating'] = pd.to_numeric(df['elo_before'], errors='coerce').fillna(1500.0)
    elif 'EloRating' not in df.columns:
        df['EloRating'] = 1500.0
    
    # ═══════════════════════════════════════════════════════════════════
    # Fill any remaining NaN values with defaults
    # ═══════════════════════════════════════════════════════════════════
    for col in ['RecentFormAvg', 'CircuitHistoryAvg', 'DriverExperienceScore', 
                'TeamPerfScore', 'EloRating']:
        if col not in df.columns:
            df[col] = 0 if col != 'EloRating' else 1500.0
        else:
            if col == 'EloRating':
                df[col] = df[col].fillna(1500.0)
            elif col == 'TeamPerfScore':
                df[col] = df[col].fillna(0.5)
            else:
                df[col] = df[col].fillna(0)
    
    logger.debug(f"[compute_basic_features] Output: Features computed for {df['driver'].nunique()} drivers")
    
    return df


def infer_from_qualifying(qual_df, race_key, race_year, event, circuit, skip_cache=False):
    """
    Generate predictions from qualifying data using DYNAMIC feature computation.
    
    🔥 KEY CHANGE: Now merges qualifying with historical data BEFORE computing features
    This gives features proper context (driver experience, form, circuit history)
    instead of defaulting all new drivers to "rookie" status.
    
    Flow:
    1. Prepare qualifying data (add metadata)
    2. MERGE with historical data (2018-2024)
    3. Compute ALL features dynamically using merged dataset
    4. Extract race rows and make predictions
    
    Args:
        skip_cache: If True, skip Supabase/external cache writes (for batch operations like season review)
    """
    import time
    start = time.time()

    # Lazy-load models + encoders + FeatureStore only when inference is requested.
    ensure_inference_assets_loaded()

    race_rows = build_race_rows_from_qualifying(qual_df, race_key, race_year, event, circuit)

    # Step 6: Prepare feature matrix
    X = race_rows[feature_cols].fillna(0)
    
    logger.debug(f"[infer] Features: {list(X.columns)}")
    logger.debug(f"[infer] Feature matrix shape: {X.shape}")
    
    # Step 7: Make predictions
    # These are binary probabilities ("this driver wins"), not a true multi-class
    # distribution across all drivers. For UI/percentages, normalize across the
    # grid so outputs are interpretable and sum to 100%.
    race_rows["p_win_raw"] = model_win.predict_proba(X)[:, 1]
    race_rows["p_pod_raw"] = model_pod.predict_proba(X)[:, 1]

    win_sum = float(race_rows["p_win_raw"].sum()) if "p_win_raw" in race_rows.columns else 0.0
    pod_sum = float(race_rows["p_pod_raw"].sum()) if "p_pod_raw" in race_rows.columns else 0.0

    if win_sum > 0:
        race_rows["p_win"] = race_rows["p_win_raw"] / win_sum
    else:
        race_rows["p_win"] = 1.0 / max(1, len(race_rows))

    if pod_sum > 0:
        race_rows["p_pod"] = race_rows["p_pod_raw"] / pod_sum
    else:
        race_rows["p_pod"] = 1.0 / max(1, len(race_rows))
    
    logger.debug(f"[infer] Predictions made in {time.time() - start:.2f}s")
    
    # Step 8: Hybrid podium ranking (60% podium ML + 40% qualifying)
    max_grid = int(race_rows["qualifying_position"].max(skipna=True)) if race_rows["qualifying_position"].notna().any() else 20
    qual = race_rows["qualifying_position"].fillna(max_grid)
    qual_score = 1 - ((qual - 1) / max(1, max_grid - 1))
    race_rows["hybrid_score"] = 0.6 * race_rows["p_pod"] + 0.4 * qual_score
    
    # Step 9: Determine winner and top 3
    winner = race_rows.sort_values("p_win", ascending=False).iloc[0]
    hybrid_board = race_rows.sort_values("hybrid_score", ascending=False)
    
    # Calculate confidence (keep decimals; avoids truncating small non-zero values to 0%)
    winner_pct = round(float(winner["p_win"]) * 100, 2)
    winner_confidence = get_confidence_level(winner_pct)
    
    logger.info(f"[infer] Prediction: {winner['driver']} {winner_pct}% ({winner_confidence['level']})")
    
    # Log to MLflow (skip for batch operations like season review)
    if not skip_cache:
        try:
            from services.mlflow_manager import log_prediction as _mlflow_log_prediction

            _mlflow_log_prediction(
                race_name=event,
                predicted_winner=winner["driver"],
                confidence=winner_pct,
                model_version="v3"
            )
        except Exception as e:
            logger.debug(f"MLflow prediction logging skipped: {e}")
    
    # Queue to Supabase (batch sync) - skip for batch operations like season review
    if not skip_cache:
        try:
            qual_cache = get_qualifying_cache(config)
            qual_cache.cache_qualifying(
                race_key=race_key,
                race_year=race_year,
                qualifying_data=race_rows.sort_values("hybrid_score", ascending=False)[["driver", "team", "qualifying_position", "p_win", "p_pod", "hybrid_score"]].to_dict('records')
            )
        except Exception as e:
            logger.debug(f"Qualifying cache queueing failed: {e}")
    
    return {
        "winner_prediction": {
            "driver": winner["driver"],
            "team": winner.get("team", None),
            "p_win": float(winner["p_win"]),
            "percentage": winner_pct,
            "confidence": winner_confidence["level"],
            "confidence_color": winner_confidence["color"]
        },
        "top3_prediction": hybrid_board.head(3)[["driver", "team", "hybrid_score", "p_pod"]].apply(
            lambda row: {
                "driver": row["driver"],
                "team": row["team"],
                "hybrid_score": float(row["hybrid_score"]),
                "percentage": round(float(row["p_pod"]) * 100, 2),
                "confidence": get_confidence_level(round(float(row["p_pod"]) * 100, 2))["level"],
                "confidence_color": get_confidence_level(round(float(row["p_pod"]) * 100, 2))["color"]
            },
            axis=1
        ).tolist(),
        "full_predictions": race_rows.sort_values("hybrid_score", ascending=False)[[
            "driver", "team", "qualifying_position", "p_win", "p_pod", "hybrid_score"
        ]].apply(
            lambda row: {
                "driver": row["driver"],
                "team": row["team"],
                "grid": int(row["qualifying_position"]) if pd.notna(row["qualifying_position"]) else 0,
                "win_prob": round(float(row["p_win"]) * 100, 1),
                "podium_prob": round(float(row["p_pod"]) * 100, 1),
                "hybrid_score": round(float(row["hybrid_score"]), 3)
            },
            axis=1
        ).tolist()
    }


def build_race_rows_from_qualifying(qual_df, race_key, race_year, event, circuit):
    """Build the per-driver feature rows for a single race.

    This is the reusable part of inference: prepare/normalize qualifying rows,
    merge with history or fall back to FeatureStore snapshots, compute engineered
    features, extract the race rows, and encode categoricals.

    Returns a DataFrame that contains `feature_cols`.
    """
    # Step 1: Prepare qualifying data
    q = qual_df.copy()
    q["race_key"] = race_key
    q["race_year"] = race_year
    q["event"] = event
    q["circuit"] = circuit

    # Ensure columns used by feature computation exist (qualifying-only inference)
    if "event_date" not in q.columns:
        q["event_date"] = pd.NaT
    if "finishing_position" not in q.columns:
        q["finishing_position"] = np.nan

    # Ensure required columns exist
    if "qualifying_position" not in q.columns and "qualifyingposition" not in q.columns:
        q["qualifying_position"] = range(1, len(q) + 1)

    if "qualifying_position" not in q.columns:
        q.rename(columns={"qualifyingposition": "qualifying_position"}, inplace=True)

    if "team" not in q.columns:
        q["team"] = "Unknown"

    # Normalize labels BEFORE merging with history so feature computation can find matches.
    try:
        driver_classes = _encoder_classes(encoders.get("driver"))
        team_classes = _encoder_classes(encoders.get("team"))
        circuit_classes = _encoder_classes(encoders.get("circuit"))

        if "driver" in q.columns:
            q["driver"] = q["driver"].astype(str).map(lambda v: _normalize_driver_code(v, driver_classes))
        if "team" in q.columns:
            q["team"] = q["team"].astype(str).map(lambda v: _normalize_team_name(v, team_classes))
        q["circuit"] = _normalize_circuit_label(circuit, event, circuit_classes)
    except Exception as e:
        logger.debug(f"[infer] Label normalization skipped: {e}")

    # Step 2: Merge with historical data if available.
    # If historical data isn't available (common in slim deployments), fall back
    # to FeatureStore snapshots so engineered features remain meaningful.
    try:
        hist_available = hist_data is not None and (hasattr(hist_data, "empty") and not hist_data.empty)
        if not hist_available:
            merged = q.copy()

            # FeatureStore-backed engineered features (fast, no parquet dependency at runtime)
            try:
                merged["RecentFormAvg"] = 0.0
                merged["CircuitHistoryAvg"] = 10.0
                merged["DriverExperienceScore"] = 0.0
                merged["TeamPerfScore"] = 0.5
                merged["EloRating"] = 1500.0

                for idx, row in merged.iterrows():
                    driver_code = str(row.get("driver") or "").upper()
                    team_name = str(row.get("team") or "Unknown")
                    circuit_name = str(row.get("circuit") or "")

                    d = feature_store.get_driver_features(driver_code) if driver_code else {}
                    merged.at[idx, "RecentFormAvg"] = float(d.get("RecentFormAvg", 10.0) or 10.0)
                    merged.at[idx, "DriverExperienceScore"] = float(d.get("DriverExperienceScore", 0.0) or 0.0)
                    merged.at[idx, "EloRating"] = float(d.get("EloRating", 1500.0) or 1500.0)

                    try:
                        merged.at[idx, "CircuitHistoryAvg"] = float(feature_store.get_circuit_history(driver_code, circuit_name))
                    except Exception:
                        merged.at[idx, "CircuitHistoryAvg"] = 10.0

                    try:
                        merged.at[idx, "TeamPerfScore"] = float(feature_store.get_team_perf_score(team_name, int(race_year)))
                    except Exception:
                        merged.at[idx, "TeamPerfScore"] = 0.5
            except Exception as fs_err:
                logger.warning(f"[infer] FeatureStore fallback failed: {fs_err}")

        else:
            logger.debug(f"[infer] Merging {len(q)} qualifying with {len(hist_data)} historical rows...")

            # Add missing required columns to qualifying
            for col in hist_data.columns:
                if col not in q.columns:
                    if col == "finishing_position":
                        q[col] = np.nan  # To be predicted
                    elif col == "event_date":
                        q[col] = pd.NaT
                    else:
                        q[col] = None

            merged = pd.concat([hist_data, q[hist_data.columns.tolist()]], ignore_index=True, sort=False)

            logger.debug(f"[infer] Merged dataset: {len(merged)} rows")

            # Step 3: Compute engineered features dynamically from merged history
            try:
                logger.debug("[infer] Computing features from merged data...")
                merged = compute_basic_features(merged)
                logger.debug("[infer] Feature computation complete")
            except Exception as e:
                logger.warning(f"[infer] Feature computation failed: {e}")
                for col in ["RecentFormAvg", "CircuitHistoryAvg", "DriverExperienceScore", "TeamPerfScore", "EloRating"]:
                    if col not in merged.columns:
                        merged[col] = 0 if col != "EloRating" else 1500.0

    except Exception as e:
        logger.warning(f"[infer] Merge/feature pipeline failed, continuing with defaults: {e}")
        merged = q.copy()
        for col in ["RecentFormAvg", "CircuitHistoryAvg", "DriverExperienceScore", "TeamPerfScore", "EloRating"]:
            if col not in merged.columns:
                merged[col] = 0 if col != "EloRating" else 1500.0

    # Step 4: Extract only the target race rows
    race_rows = merged[merged["race_key"] == race_key].copy()

    if len(race_rows) == 0:
        race_rows = merged[merged["event"] == event].copy()

    if len(race_rows) == 0:
        race_rows = merged.tail(len(q)).copy()

    logger.debug(f"[infer] Extracted {len(race_rows)} rows for race prediction")

    # Step 5: Encode categorical features
    for c in ["driver", "team", "circuit"]:
        if c not in race_rows.columns:
            race_rows[f"{c}_enc"] = 0
            continue

        enc_obj = encoders.get(c)
        if enc_obj is None:
            race_rows[f"{c}_enc"] = 0
            continue

        if hasattr(enc_obj, "classes_"):
            classes = list(enc_obj.classes_)
        elif isinstance(enc_obj, (list, tuple)):
            classes = list(enc_obj)
        else:
            classes = []

        index = {str(v): i for i, v in enumerate(classes)}
        unknown = len(classes)
        race_rows[f"{c}_enc"] = [index.get(str(v), unknown) for v in race_rows[c].astype(str)]

        try:
            if len(race_rows) > 0 and unknown > 0:
                unk_count = int((race_rows[f"{c}_enc"] == unknown).sum())
                unk_frac = unk_count / max(1, len(race_rows))
                if unk_frac >= 0.5:
                    logger.warning(f"[infer] High unknown rate for {c}: {unk_count}/{len(race_rows)}")
        except Exception:
            pass

    # Ensure feature columns exist
    for col in feature_cols:
        if col not in race_rows.columns:
            race_rows[col] = 0

    return race_rows


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
    fs_health = None
    fs_initialized = feature_store is not None
    if fs_initialized:
        try:
            fs_health = feature_store.health_check()
        except Exception:
            fs_health = None
            fs_initialized = False

    return jsonify({
        "status": "healthy",
        "version": "2.0.0",
        "architecture": "optimized",
        "feature_store": {
            "initialized": fs_initialized,
            "drivers_cached": (fs_health or {}).get("drivers_in_memory", 0),
            "parquet_available": (fs_health or {}).get("parquet_exists", False),
            "redis_connected": (fs_health or {}).get("redis_connected", False),
            "lru_cache": (fs_health or {}).get("lru_cache_info", {})
        },
        "inference_assets": {
            "loaded": bool(_inference_assets_loaded),
            "loaded_at": _inference_assets_loaded_at,
            "last_error": bool(_inference_assets_load_error),
        },
        "prediction_logger": {
            "mode": prediction_logger.mode,
            "supabase_connected": prediction_logger.mode == "supabase"
        },
        "data": {
            "historical_format": "parquet" if (fs_health or {}).get("parquet_exists", False) else "csv",
            "compression": "5.5x" if (fs_health or {}).get("parquet_exists", False) else "none"
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
        return _internal_error("Error getting accuracy stats", e)


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
        return _internal_error("Error getting prediction history", e)


@app.route('/api/update-actual-winner', methods=['POST'])
def update_actual_winner():
    """
    Update the actual winner for a race after it completes.
    Expects JSON: {"race_name": "São Paulo Grand Prix", "actual_winner": "NOR"}
    """
    try:
        auth_error = _require_write_key()
        if auth_error is not None:
            return auth_error

        data = request.json or {}
        race_name = data.get('race_name')
        race_year = data.get('race_year')
        actual_winner = data.get('actual_winner')
        
        if not race_name or not actual_winner:
            return jsonify({
                "success": False,
                "error": "Both race_name and actual_winner are required"
            }), 400

        success = prediction_logger.update_actual_winner(race_name, actual_winner, race_year=race_year)
        
        return jsonify({
            "success": success,
            "message": f"Updated actual winner for {race_name} to {actual_winner}" if success else "Update failed"
        })
    except Exception as e:
        return _internal_error("Error updating actual winner", e)


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

        def bad_request(message: str):
            return jsonify({"success": False, "error": message}), 400

        def _ensure_str(field: str, max_len: int = 200) -> str:
            val = data.get(field)
            if val is None:
                raise ValueError(f"Missing required field: {field}")
            s = str(val).strip()
            if not s:
                raise ValueError(f"Missing required field: {field}")
            if len(s) > max_len:
                raise ValueError(f"{field} too long")
            return s

        # Validate required fields
        try:
            data["race_key"] = _ensure_str("race_key", max_len=120)
            data["event"] = _ensure_str("event", max_len=120)
            data["circuit"] = _ensure_str("circuit", max_len=120)
        except ValueError as ve:
            return bad_request(str(ve))

        # Validate race_year as int within plausible range
        race_year_raw = data.get("race_year")
        try:
            race_year_int = int(race_year_raw)
        except Exception:
            return bad_request("race_year must be an integer")

        current_year = datetime.now().year
        if race_year_int < 1950 or race_year_int > current_year + 1:
            return bad_request(f"race_year must be between 1950 and {current_year + 1}")
        data["race_year"] = race_year_int
        
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
                    qualifying = fetch_qualifying_from_ergast(int(ry), circuit=circuit, event=event)
                    if not qualifying:
                        return jsonify({"success": False, "error": "No qualifying data available"}), 404
            except Exception as e:
                return _internal_error("Could not fetch qualifying", e)

        if not isinstance(qualifying, list):
            return bad_request("qualifying must be a list")
        if len(qualifying) > 30:
            return bad_request("qualifying list too large")

        # Convert qualifying data to DataFrame
        qual_df = pd.DataFrame(qualifying)
        if qual_df.empty:
            return bad_request("qualifying is empty")
        
        # Generate predictions
        try:
            predictions = infer_from_qualifying(
                qual_df,
                data["race_key"],
                data["race_year"],
                data["event"],
                data["circuit"],
            )
        except InferenceAssetsUnavailableError as ie:
            return jsonify({
                "success": False,
                "error": str(ie),
            }), 503
        
        # Log prediction to database (Supabase/CSV)
        try:
            prediction_logger.log_prediction(
                race_name=data["event"],
                predicted_winner=predictions["winner_prediction"]["driver"],
                race_year=data.get("race_year"),
                circuit=data.get("circuit"),
                confidence=predictions["winner_prediction"]["percentage"],
                model_version="xgb_v3",
                full_predictions=predictions
            )
            logger.info(f"✓ Prediction logged for {data['event']}")
        except Exception as log_err:
            logger.warning(f"Could not log prediction: {log_err}")
        
        return jsonify({
            "success": True,
            "data": predictions
        })
    
    except Exception as e:
        return _internal_error("Prediction error", e)


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
        return _internal_error("Qualifying lookup error", e)


def get_races_with_predictions_and_history():
    """
    Get last 5 completed races with predictions and history.
    OPTIMIZED: Uses cache first, skips expensive FastF1 calls when possible
    Returns faster even if not all data is complete - targets <15s response
    Falls back to previous year if current year has no completed races.
    """
    try:
        logger.info("Building race history with predictions...")
        
        year = datetime.now().year
        try:
            schedule = fastf1.get_event_schedule(year)
            schedule['EventDate'] = pd.to_datetime(schedule['EventDate'])
        except Exception as sched_err:
            logger.warning(f"Could not get {year} schedule: {sched_err}")
            # Try previous year
            year = year - 1
            schedule = fastf1.get_event_schedule(year)
            schedule['EventDate'] = pd.to_datetime(schedule['EventDate'])
        
        today = pd.to_datetime(datetime.now().date())
        
        # Get last 5 completed races
        past_races = schedule[schedule['EventDate'] <= today].sort_values('EventDate', ascending=False)
        
        # If no completed races in current year, fall back to previous year
        if past_races.empty and year == datetime.now().year:
            logger.info(f"No completed races in {year}, falling back to {year - 1}")
            year = year - 1
            schedule = fastf1.get_event_schedule(year)
            schedule['EventDate'] = pd.to_datetime(schedule['EventDate'])
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
                
                # 1. Try to get qualifying data from PREDICTION CACHE FIRST (fast, in-memory, no HTTP)
                qual_data = None
                
                try:
                    # Use QualifyingCache instead of Supabase queries
                    qual_cache = get_qualifying_cache()
                    cached_qual = qual_cache.get_cached_qualifying(race_key)
                    
                    if cached_qual:
                        if isinstance(cached_qual, str):
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
                        
                        logger.info(f"  ✓ Got qualifying from MEMORY cache (no HTTP): {len(qual_data) if isinstance(qual_data, list) else 1} drivers")
                        
                except Exception as e:
                    logger.debug(f"Memory cache lookup failed: {e}")
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
        logger.info("Getting next race prediction (Supabase-backed)...")

        def _stable_qualifying_hash(rows):
            try:
                rows_norm = []
                for r in rows or []:
                    if not isinstance(r, dict):
                        continue
                    rows_norm.append(
                        {
                            "driver": str(r.get("driver") or "").upper(),
                            "team": r.get("team"),
                            "qualifying_position": int(r.get("qualifying_position")) if r.get("qualifying_position") is not None else None,
                            "qualifying_lap_time_s": float(r.get("qualifying_lap_time_s")) if r.get("qualifying_lap_time_s") is not None else None,
                        }
                    )
                rows_norm.sort(key=lambda x: (x.get("qualifying_position") or 999, x.get("driver") or ""))
                payload = json.dumps(rows_norm, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                import hashlib

                return hashlib.sha1(payload.encode("utf-8")).hexdigest()
            except Exception:
                return None

        def _get_supabase_client():
            try:
                from supabase import create_client as _create_client
            except Exception:
                return None

            sb_url = getattr(config, "SUPABASE_URL", None)
            # Prefer service key when available (server-side) to bypass RLS.
            sb_key = getattr(config, "SUPABASE_SERVICE_KEY", None) or getattr(config, "SUPABASE_KEY", None)
            if not sb_url or not sb_key:
                return None
            try:
                return _create_client(sb_url, sb_key)
            except Exception:
                return None

        # Next race metadata (FastF1 first, Ergast fallback)
        nr = get_next_race_from_fastf1(year=None)
        if not nr:
            nr = ergast_next_race() or {}
        if not nr:
            return None

        race_name = nr.get("event_name") or nr.get("event") or nr.get("race") or "Unknown"
        race_year = int(nr.get("year") or nr.get("race_year") or datetime.now().year)
        race_date = nr.get("date")
        circuit = nr.get("circuit") or race_name

        # Round number is best-effort (used only for display)
        race_round = None
        try:
            schedule = fastf1.get_event_schedule(race_year)
            schedule["EventDate"] = pd.to_datetime(schedule["EventDate"])
            # Match by event name (best-effort)
            if "EventName" in schedule.columns:
                names = schedule["EventName"].astype(str)
            elif "Event" in schedule.columns:
                names = schedule["Event"].astype(str)
            else:
                names = pd.Series([""] * len(schedule))

            match = schedule[names.str.lower() == str(race_name).lower()]
            if not match.empty and "RoundNumber" in match.columns:
                race_round = int(match.iloc[0].get("RoundNumber") or 0) or None
        except Exception:
            race_round = None

        # Build a race_key that matches the pipeline/jobs convention.
        race_key = nr.get("race_key")
        if not race_key and race_round:
            race_key = f"{race_year}_{race_round}_{race_name}".replace(" ", "_")

        # If qualifying exists in Supabase for this race_key, prefer computing the
        # prediction live (this avoids serving a stale/low-quality prediction that
        # may have been generated by a job running without proper history).
        qual_rows = None
        qual_hash = None
        if race_key:
            try:
                sb = _get_supabase_client()
                if sb is not None:
                    qresp = (
                        sb.table("qualifying_raw")
                        .select("driver,team,qualifying_position,qualifying_lap_time_s,ingested_at")
                        .eq("race_key", race_key)
                        .execute()
                    )
                    qual_rows = qresp.data or []
                    if isinstance(qual_rows, list) and len(qual_rows) > 0:
                        qual_hash = _stable_qualifying_hash(qual_rows)
                        logger.info(
                            "Found qualifying_raw for %s: %s row(s)",
                            race_key,
                            len(qual_rows),
                        )
            except Exception as e:
                logger.debug(f"Supabase qualifying_raw lookup failed: {e}")

        row = None
        try:
            row = prediction_logger.get_latest_prediction(race_name, race_year=race_year)
        except Exception:
            row = None

        # If we have qualifying_raw for the upcoming race, we can recompute.
        # In production, avoid live inference in HTTP handlers; rely on the cron/worker
        # to precompute and store predictions.
        if config.FLASK_ENV != "production" and qual_rows and isinstance(qual_rows, list) and len(qual_rows) >= 10:
            try:
                qual_df = pd.DataFrame(qual_rows)
                # Ensure expected columns exist
                if "qualifying_position" not in qual_df.columns and "qualifyingposition" in qual_df.columns:
                    qual_df = qual_df.rename(columns={"qualifyingposition": "qualifying_position"})
                predictions = infer_from_qualifying(
                    qual_df,
                    race_key,
                    race_year,
                    race_name,
                    circuit,
                    skip_cache=True,
                )

                winner = (predictions or {}).get("winner_prediction") or {}
                predicted_winner_now = winner.get("driver")
                confidence_now = winner.get("percentage")

                # Best-effort: store/update the predictions row so future requests are fast.
                try:
                    prediction_logger.log_prediction(
                        race_name=race_name,
                        predicted_winner=predicted_winner_now,
                        race_year=race_year,
                        circuit=circuit,
                        confidence=confidence_now,
                        model_version="xgb_v3",
                        full_predictions={
                            **(predictions or {}),
                            "race_key": race_key,
                            "round": race_round,
                            "date": race_date,
                            "status": "ready",
                            "source": "qualifying_raw",
                            "qualifying_hash": qual_hash,
                        },
                        allow_update=True,
                    )
                except Exception as e:
                    logger.debug(f"Could not persist refreshed next-race prediction: {e}")

                # Force response to use the freshly computed prediction
                row = {
                    "predicted": predicted_winner_now,
                    "confidence": confidence_now,
                    "full_predictions": {
                        **(predictions or {}),
                        "race_key": race_key,
                        "round": race_round,
                        "date": race_date,
                        "status": "ready",
                        "source": "qualifying_raw",
                        "qualifying_hash": qual_hash,
                    },
                }
                logger.info("Refreshed next-race prediction from qualifying_raw for %s", race_key)

            except Exception as e:
                logger.warning(f"Live next-race inference from qualifying_raw failed: {e}")

        used_stale_fallback = False

        # If we don't have a prediction yet for the upcoming race, fall back to the
        # most recent stored prediction so the UI can continue to show driver cards.
        if not (row and row.get("predicted")):
            try:
                fallback = prediction_logger.get_most_recent_prediction_with_full_predictions()
                if not fallback:
                    fallback = prediction_logger.get_most_recent_prediction()
            except Exception:
                fallback = None

            if fallback and fallback.get("predicted"):
                row = fallback
                used_stale_fallback = True

                # Override display metadata to match the fallback row (avoid mixing race labels).
                race_name = fallback.get("race") or race_name
                circuit = fallback.get("circuit") or circuit
                try:
                    if fallback.get("race_year") is not None:
                        race_year = int(fallback.get("race_year"))
                except Exception:
                    pass

                # Best-effort: round/date from stored full_predictions if present.
                fp_raw = fallback.get("full_predictions")
                if isinstance(fp_raw, str):
                    try:
                        fp_raw = json.loads(fp_raw)
                    except Exception:
                        fp_raw = None
                if isinstance(fp_raw, dict):
                    try:
                        if fp_raw.get("round") is not None:
                            race_round = int(fp_raw.get("round"))
                    except Exception:
                        pass
                    if fp_raw.get("date"):
                        race_date = fp_raw.get("date")
                else:
                    # Fall back to the stored timestamp if we don't have a dedicated date.
                    if fallback.get("timestamp"):
                        race_date = str(fallback.get("timestamp"))

        predicted_winner = "TBA"
        predicted_confidence = 0
        predicted_top3 = []
        full_predictions = []
        status = "pending"

        if row and row.get("predicted"):
            status = "stale" if used_stale_fallback else "ready"
            predicted_winner = row.get("predicted")

            conf_val = row.get("confidence")
            try:
                predicted_confidence = int(float(conf_val)) if conf_val is not None else 0
            except Exception:
                predicted_confidence = 0

            fp = row.get("full_predictions")
            if isinstance(fp, str):
                try:
                    fp = json.loads(fp)
                except Exception:
                    fp = None

            if isinstance(fp, dict):
                fp_outer = fp
                fp_data = fp
                if isinstance(fp_outer.get("predictions"), dict):
                    fp_data = fp_outer.get("predictions")

                # Use stored metadata when available
                try:
                    round_val = fp_outer.get("round")
                    if round_val is None and isinstance(fp_data, dict):
                        round_val = fp_data.get("round")
                    if round_val is not None:
                        race_round = int(round_val)
                except Exception:
                    pass
                date_val = fp_outer.get("date")
                if not date_val and isinstance(fp_data, dict):
                    date_val = fp_data.get("date")
                if date_val:
                    race_date = date_val

                t3 = []
                if isinstance(fp_data, dict):
                    t3 = fp_data.get("top3_prediction") or fp_data.get("predicted_top3") or []
                if isinstance(t3, list):
                    if t3 and isinstance(t3[0], dict):
                        predicted_top3 = [d.get("driver") for d in t3 if isinstance(d, dict) and d.get("driver")][:3]
                    else:
                        predicted_top3 = [str(x) for x in t3 if x][:3]

                if isinstance(fp_data, dict):
                    full_predictions = fp_data.get("full_predictions") or []
                    if not isinstance(full_predictions, list):
                        full_predictions = []

        # date formatting
        if isinstance(race_date, str):
            date_str = race_date[:10]
        else:
            try:
                date_str = pd.to_datetime(race_date).strftime("%Y-%m-%d")
            except Exception:
                date_str = datetime.now().strftime("%Y-%m-%d")

        return {
            "round": race_round,
            "race": race_name,
            "circuit": circuit,
            "date": date_str,
            "qualifying_count": 0,
            "predicted_winner": predicted_winner,
            "predicted_confidence": predicted_confidence,
            "predicted_top3": predicted_top3,
            "full_predictions": full_predictions,
            "status": status,
        }

    except Exception as e:
        logger.error(f"Error getting next race prediction: {e}")
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
        
        # Quick off-season check: see if any races have completed this year
        # This avoids expensive Supabase/FastF1 calls when we're in off-season
        year = datetime.now().year
        try:
            schedule = fastf1.get_event_schedule(year)
            schedule['EventDate'] = pd.to_datetime(schedule['EventDate'])
            today = pd.to_datetime(datetime.now().date())
            
            # Filter out testing events - only count actual race rounds
            actual_races = schedule[schedule['RoundNumber'] > 0] if 'RoundNumber' in schedule.columns else schedule
            completed_races = actual_races[actual_races['EventDate'] <= today]
            has_completed_races = len(completed_races) > 0
            
            logger.info(f"Quick check: {len(completed_races)} completed races in {year}")
        except Exception as sched_err:
            logger.warning(f"Could not check {year} schedule: {sched_err}")
            has_completed_races = False
        
        # If no completed races this year, go straight to season review
        if not has_completed_races:
            logger.info("Off-season detected (no completed races) - returning season review directly")
            season_review = get_season_review()
            
            # Still try to get next race info (quick, just schedule lookup - already fetched above)
            next_race_info = None
            try:
                future_races = actual_races[actual_races['EventDate'] > today].sort_values('EventDate')
                if not future_races.empty:
                    nxt = future_races.iloc[0]
                    next_race_info = {
                        "round": int(nxt.get('RoundNumber', 1)),
                        "race": nxt.get('EventName', 'Unknown'),
                        "circuit": nxt.get('EventName', 'Unknown'),
                        "date": nxt['EventDate'].strftime('%Y-%m-%d'),
                        "status": "pending",
                        "predicted_winner": "TBA",
                        "predicted_confidence": 0,
                        "predicted_top3": [],
                        "full_predictions": []
                    }
            except Exception:
                pass
            
            return jsonify({
                "success": True,
                "race_history": [],
                "next_race": next_race_info,
                "is_off_season": True,
                "season_review": season_review,
                "accuracy": {"percentage": 0, "correct_predictions": 0, "total_races": 0}
            })
        
        # Normal in-season flow (Supabase-backed)
        next_race = get_next_race_prediction()

        # Build last 5 races from prediction logs (Supabase), not from cached FastF1/history.
        race_history = []
        try:
            df = prediction_logger.get_prediction_history(limit=200)
            if not df.empty:
                # Normalize timestamp
                ts_col = "timestamp" if "timestamp" in df.columns else None
                if ts_col:
                    df["timestamp_dt"] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
                else:
                    df["timestamp_dt"] = pd.NaT

                # Only completed races (have actual)
                if "actual" not in df.columns:
                    completed = df.iloc[0:0]
                else:
                    completed = df[df["actual"].notna()]
                completed = completed.sort_values("timestamp_dt", ascending=False)

                for _, row in completed.head(5).iterrows():
                    race_name = row.get("race")
                    predicted = row.get("predicted")
                    actual = row.get("actual")

                    confidence_val = row.get("confidence")
                    try:
                        confidence = int(float(confidence_val)) if confidence_val is not None and str(confidence_val) != "nan" else 0
                    except Exception:
                        confidence = 0

                    correct = row.get("correct")
                    try:
                        correct = bool(correct) if correct is not None else (str(predicted).upper() == str(actual).upper())
                    except Exception:
                        correct = False

                    ts = row.get("timestamp_dt")
                    try:
                        date_str = ts.strftime("%Y-%m-%d") if pd.notna(ts) else str(row.get("timestamp") or "")[:10]
                    except Exception:
                        date_str = str(row.get("timestamp") or "")[:10]

                    race_history.append({
                        "race": race_name or "Unknown",
                        "circuit": row.get("circuit") or race_name or "Unknown",
                        "predicted_winner": predicted or "N/A",
                        "actual_winner": actual or "TBA",
                        "correct": correct,
                        "confidence": confidence,
                        "date": date_str or datetime.now().strftime("%Y-%m-%d"),
                    })
        except Exception as hist_err:
            logger.warning(f"Could not build race history from logs: {hist_err}")
            race_history = []

        correct_predictions = sum(1 for r in race_history if r.get("correct"))
        accuracy_pct = int((correct_predictions / len(race_history)) * 100) if race_history else 0
        
        logger.info(f"Accuracy: {accuracy_pct}% ({correct_predictions}/{len(race_history) if race_history else 0})")
        
        response = jsonify({
            "success": True,
            "race_history": race_history or [],
            "next_race": next_race,
            "is_off_season": False,
            "season_review": None,
            "accuracy": {
                "percentage": accuracy_pct,
                "correct_predictions": correct_predictions,
                "total_races": len(race_history) if race_history else 0
            }
        })
        
        return response
    
    except Exception as e:
        return _internal_error("/api/predict/sao-paulo failed", e)


# =============================================================================
# SEASON REVIEW ENDPOINT - Shows previous season prediction accuracy
# =============================================================================

# Cache for season review results (historical data doesn't change)
_season_review_cache = {}

def _load_season_review_from_supabase(year):
    """Try to load a stored season review from the Supabase predictions table."""
    try:
        from supabase import create_client as _create_client
        
        supabase_url = config.SUPABASE_URL
        supabase_key = getattr(config, 'SUPABASE_SERVICE_KEY', None) or getattr(config, 'SUPABASE_KEY', None)
        
        if not supabase_url or not supabase_key:
            return None
        
        sb = _create_client(supabase_url, supabase_key)
        
        # Check for stored predictions for this year
        races_result = sb.table('predictions').select('*').eq('race_year', year).neq('race', f'__SEASON_SUMMARY_{year}__').order('timestamp').execute()
        
        if not races_result.data or len(races_result.data) == 0:
            return None
        
        logger.info(f"Loading {len(races_result.data)} stored predictions for {year} from Supabase")
        
        # Build races list from stored predictions
        season_results = []
        correct_count = 0
        podium_correct_count = 0
        
        for row in races_result.data:
            import json as _json
            full_pred = {}
            if row.get('full_predictions'):
                try:
                    full_pred = _json.loads(row['full_predictions']) if isinstance(row['full_predictions'], str) else row['full_predictions']
                except:
                    full_pred = {}
            
            is_correct = row.get('correct', False)
            is_podium_correct = full_pred.get('podium_correct', False)
            
            if is_correct:
                correct_count += 1
            if is_podium_correct:
                podium_correct_count += 1
            
            season_results.append({
                "round": full_pred.get('round', len(season_results) + 1),
                "race": row.get('race', ''),
                "circuit": row.get('circuit', ''),
                "date": full_pred.get('date', row.get('timestamp', '')),
                "predicted_winner": row.get('predicted', 'N/A'),
                "predicted_top3": full_pred.get('predicted_top3', []),
                "confidence": row.get('confidence', 0),
                "actual_winner": row.get('actual', ''),
                "actual_podium": full_pred.get('actual_podium', []),
                "correct": is_correct,
                "podium_correct": is_podium_correct,
                "status": full_pred.get('status', 'complete')
            })
        
        total_races = len([r for r in season_results if r["status"] == "complete"])
        accuracy = int((correct_count / total_races) * 100) if total_races > 0 else 0
        podium_accuracy = int((podium_correct_count / total_races) * 100) if total_races > 0 else 0
        
        # Get available years from hist_data
        available_years = sorted(hist_data['race_year'].unique()) if not hist_data.empty else [year]
        
        result = {
            "year": year,
            "races": season_results,
            "stats": {
                "total_races": total_races,
                "correct_predictions": correct_count,
                "accuracy_percentage": accuracy,
                "podium_correct": podium_correct_count,
                "podium_accuracy_percentage": podium_accuracy,
                "total_with_data": len(season_results)
            },
            "available_years": [int(y) for y in available_years]
        }
        
        logger.info(f"✓ Loaded season review for {year} from Supabase: {correct_count}/{total_races} correct ({accuracy}%)")
        return result
        
    except Exception as e:
        logger.debug(f"Could not load season review from Supabase: {e}")
        return None

def get_season_review(year=None):
    """
    Build a full season review from historical data.
    
    Priority:
    1. In-memory cache (instant)
    2. Supabase predictions table (fast DB read, ~1-2s)
    3. Compute from scratch using ML model (slow, ~20s)
    
    If year is None, defaults to the most recent year in hist_data.
    Results are cached since historical seasons don't change.
    """
    global _season_review_cache
    
    try:
        if hist_data.empty:
            logger.warning("No historical data available for season review")
            return {"races": [], "year": year or 0, "accuracy": 0}
        
        # Determine the review year
        available_years = sorted(hist_data['race_year'].unique())
        if year is None:
            year = int(available_years[-1])  # Latest year in data
        
        # Priority 1: In-memory cache
        if year in _season_review_cache:
            logger.info(f"Returning cached season review for {year}")
            return _season_review_cache[year]
        
        # Priority 2: Supabase stored predictions
        db_result = _load_season_review_from_supabase(year)
        if db_result and db_result.get('races'):
            _season_review_cache[year] = db_result  # Cache in memory too
            return db_result
        
        if year not in available_years:
            logger.warning(f"Year {year} not found in historical data. Available: {available_years}")
            return {"races": [], "year": year, "accuracy": 0, "available_years": [int(y) for y in available_years]}
        
        logger.info(f"Building season review for {year}...")
        
        # Filter data for the target year
        year_data = hist_data[hist_data['race_year'] == year].copy()
        
        # Get unique races (events) in chronological order
        if 'event_date' in year_data.columns:
            year_data['event_date'] = pd.to_datetime(year_data['event_date'], errors='coerce')
            race_order = (
                year_data.dropna(subset=['event_date'])
                .groupby('event')['event_date']
                .first()
                .sort_values()
            )
            unique_races = race_order.index.tolist()
        else:
            unique_races = year_data['event'].unique().tolist()
        
        logger.info(f"Found {len(unique_races)} races in {year}")
        
        season_results = []
        correct_count = 0
        podium_correct_count = 0
        
        for race_event in unique_races:
            try:
                race_data = year_data[year_data['event'] == race_event].copy()
                
                if race_data.empty:
                    continue
                
                # Get actual winner
                winners = race_data[race_data['finishing_position'] == 1]
                if winners.empty:
                    # Try numeric conversion
                    race_data['fp_num'] = pd.to_numeric(race_data['finishing_position'], errors='coerce')
                    winners = race_data[race_data['fp_num'] == 1]
                
                actual_winner = winners.iloc[0]['driver'] if not winners.empty else 'Unknown'
                
                # Get actual podium (top 3)
                race_data['fp_num'] = pd.to_numeric(race_data['finishing_position'], errors='coerce')
                actual_podium = (
                    race_data.dropna(subset=['fp_num'])
                    .nsmallest(3, 'fp_num')['driver']
                    .tolist()
                )
                
                # Get race date
                race_date = None
                if 'event_date' in race_data.columns:
                    dates = race_data['event_date'].dropna()
                    if not dates.empty:
                        race_date = pd.to_datetime(dates.iloc[0]).strftime('%Y-%m-%d')
                
                # Get circuit
                circuit = race_data['circuit'].iloc[0] if 'circuit' in race_data.columns else race_event
                
                # Extract qualifying data for prediction
                qual_rows = race_data[race_data['qualifying_position'].notna()].copy()
                
                if qual_rows.empty:
                    # No qualifying data - skip prediction for this race
                    season_results.append({
                        "round": len(season_results) + 1,
                        "race": race_event,
                        "circuit": circuit,
                        "date": race_date,
                        "predicted_winner": "N/A",
                        "predicted_top3": [],
                        "confidence": 0,
                        "actual_winner": actual_winner,
                        "actual_podium": actual_podium,
                        "correct": False,
                        "podium_correct": False,
                        "status": "no_qualifying_data"
                    })
                    continue
                
                # Build qualifying DataFrame for model
                race_key = f"{year}_{len(season_results)+1}_{race_event.replace(' ', '_')}"
                qual_df = qual_rows[['driver', 'team', 'qualifying_position']].copy()
                qual_df['race_key'] = race_key
                qual_df['race_year'] = year
                qual_df['event'] = race_event
                qual_df['circuit'] = circuit
                
                # Run prediction through the model (skip_cache=True for batch season review)
                try:
                    predictions = infer_from_qualifying(
                        qual_df, race_key, year, race_event, circuit, skip_cache=True
                    )
                    
                    predicted_winner = predictions["winner_prediction"]["driver"]
                    predicted_confidence = predictions["winner_prediction"]["percentage"]
                    predicted_top3 = [d["driver"] for d in predictions["top3_prediction"][:3]]
                    
                except Exception as pred_err:
                    logger.warning(f"  Prediction failed for {race_event}: {pred_err}")
                    predicted_winner = "Error"
                    predicted_confidence = 0
                    predicted_top3 = []
                
                # Check if prediction was correct
                is_correct = (
                    str(predicted_winner).upper() == str(actual_winner).upper()
                )
                
                # Check if predicted winner was on actual podium
                is_podium_correct = any(
                    str(predicted_winner).upper() == str(p).upper()
                    for p in actual_podium
                )
                
                if is_correct:
                    correct_count += 1
                if is_podium_correct:
                    podium_correct_count += 1
                
                season_results.append({
                    "round": len(season_results) + 1,
                    "race": race_event,
                    "circuit": circuit,
                    "date": race_date,
                    "predicted_winner": predicted_winner,
                    "predicted_top3": predicted_top3,
                    "confidence": predicted_confidence,
                    "actual_winner": actual_winner,
                    "actual_podium": actual_podium,
                    "correct": is_correct,
                    "podium_correct": is_podium_correct,
                    "status": "complete"
                })
                
                logger.info(f"  {race_event}: Predicted={predicted_winner} | Actual={actual_winner} | {'✓' if is_correct else '✗'}")
                
            except Exception as e:
                logger.error(f"  Error processing {race_event}: {e}")
                continue
        
        total_races = len([r for r in season_results if r["status"] == "complete"])
        accuracy = int((correct_count / total_races) * 100) if total_races > 0 else 0
        podium_accuracy = int((podium_correct_count / total_races) * 100) if total_races > 0 else 0
        
        logger.info(f"✓ Season review for {year}: {correct_count}/{total_races} correct ({accuracy}%)")
        
        result = {
            "year": year,
            "races": season_results,
            "stats": {
                "total_races": total_races,
                "correct_predictions": correct_count,
                "accuracy_percentage": accuracy,
                "podium_correct": podium_correct_count,
                "podium_accuracy_percentage": podium_accuracy,
                "total_with_data": len(season_results)
            },
            "available_years": [int(y) for y in available_years]
        }
        
        # Cache the result (historical data doesn't change)
        _season_review_cache[year] = result
        
        return result
        
    except Exception as e:
        logger.error(f"Error building season review: {e}")
        logger.error(traceback.format_exc())
        return {"races": [], "year": year or 0, "stats": {"accuracy_percentage": 0}, "error": "Internal server error"}


@app.route('/api/season-review', methods=['GET'])
@app.route('/api/season-review/<int:year>', methods=['GET'])
def season_review_endpoint(year=None):
    """
    Season Review Endpoint
    Returns full season prediction accuracy for a given year.
    If no year specified, returns the most recent season in the dataset.
    
    Query params:
      - year: Season year (optional, defaults to latest in data)
    
    Response:
      - year: The season year
      - races: Array of race results with predicted vs actual
      - stats: Overall accuracy statistics
      - available_years: List of years available for review
    """
    try:
        # Allow year from query param or URL path
        if year is None:
            year = request.args.get('year', type=int)
        
        logger.info(f"=== Season Review Request (year={year}) ===")
        
        review = get_season_review(year)
        
        return jsonify({
            "success": True,
            **review
        })
        
    except Exception as e:
        return _internal_error("Season review endpoint failed", e)
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
    Fetch current season driver standings from Ergast API (actual current points)
    Returns list of drivers with current points and team info
    """
    try:
        current_year = datetime.now().year
        logger.info(f"Fetching {current_year} driver standings from Ergast...")
        
        # Use FastF1's Ergast wrapper to get latest standings
        from fastf1.ergast import Ergast
        ergast = Ergast()
        
        # Get latest standings for current season (None means latest round)
        standings_data = ergast.get_driver_standings(season=current_year, round=None)
        
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
    Fetch current season constructor standings from Ergast API (actual current points)
    Returns list of constructors with current points
    """
    try:
        current_year = datetime.now().year
        logger.info(f"Fetching {current_year} constructor standings from Ergast...")
        
        # Use FastF1's Ergast wrapper to get latest constructor standings
        from fastf1.ergast import Ergast
        ergast = Ergast()
        
        # Get latest standings for current season (None means latest round)
        standings_data = ergast.get_constructor_standings(season=current_year, round=None)
        
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
        
        # Check rounds in reverse order for current season
        current_year = datetime.now().year
        rounds_to_check = list(range(24, 0, -1))
        
        session = None
        latest_round = None
        
        logger.info(f"Searching for latest completed race with results ({current_year})...")
        
        for round_num in rounds_to_check:
            try:
                s = fastf1.get_session(current_year, round_num, 'R')
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
    Get up to 5 latest races with predictions vs actual results.

    Source of truth:
    - Supabase `predictions` table when configured
    - local `monitoring/predictions.csv` fallback
    """
    # Handle CORS preflight (Flask-CORS handles this automatically)
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
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
        
        logger.info("Fetching race history from prediction logs...")

        max_races = 5
        lookback_rows = 250

        def _as_bool(value):
            if pd.isna(value):
                return None
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(int(value))
            s = str(value).strip().lower()
            if s in {"true", "t", "1", "yes", "y"}:
                return True
            if s in {"false", "f", "0", "no", "n"}:
                return False
            return None

        races = []
        try:
            df = prediction_logger.get_prediction_history(limit=lookback_rows)

            if df is None or df.empty:
                return jsonify({
                    "success": True,
                    "data": [],
                    "source": "empty",
                    "message": "No prediction history available yet"
                })

            # Normalize column names across possible sources
            rename_map = {
                "race_name": "race",
                "predicted_winner": "predicted",
                "actual_winner": "actual",
                "is_correct": "correct",
                "predicted_confidence": "confidence",
            }
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

            required_cols = {"race", "predicted", "timestamp"}
            if not required_cols.issubset(set(df.columns)):
                logger.warning(f"Prediction history missing required cols: {sorted(required_cols - set(df.columns))}")
                return jsonify({
                    "success": True,
                    "data": [],
                    "source": "empty",
                    "message": "Prediction history schema mismatch"
                })

            # Only include rows where actual winner is known (so UI shows predicted vs actual)
            if "actual" in df.columns:
                df = df[df["actual"].notna()]
                df = df[df["actual"].astype(str).str.strip().str.upper().ne("TBA")]

            df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values("timestamp_dt", ascending=False)

            # Distinct races (keep latest prediction row per race)
            df = df.drop_duplicates(subset=["race"], keep="first").head(max_races)

            for _, row in df.iterrows():
                race_name = str(row.get("race") or "").strip()
                predicted = str(row.get("predicted") or "N/A").strip()
                actual = str(row.get("actual") or "N/A").strip() if "actual" in row else "N/A"

                confidence_val = row.get("confidence") if "confidence" in row else 0
                try:
                    confidence = int(float(confidence_val)) if confidence_val is not None and str(confidence_val) != "nan" else 0
                except Exception:
                    confidence = 0

                correct = _as_bool(row.get("correct")) if "correct" in row else None
                if correct is None and actual not in {"", "N/A", "TBA"} and predicted not in {"", "N/A"}:
                    correct = predicted.upper() == actual.upper()
                if correct is None:
                    correct = False

                ts = row.get("timestamp_dt")
                date_str = None
                if pd.notna(ts):
                    try:
                        date_str = ts.strftime("%Y-%m-%d")
                    except Exception:
                        date_str = None
                if not date_str:
                    date_str = str(row.get("timestamp") or "")[:10] or datetime.now().strftime("%Y-%m-%d")

                races.append({
                    "race": race_name or "Unknown",
                    "circuit": race_name or "Unknown",
                    "predicted_winner": predicted,
                    "actual_winner": actual,
                    "correct": bool(correct),
                    "confidence": confidence,
                    "date": date_str,
                })

        except Exception as e:
            logger.error(f"Prediction log read error: {e}")
            traceback.print_exc()

        logger.info(f"Race history ready: {len(races)} race(s) from prediction logs")
        
        # Save to memory cache for next request (short TTL)
        if races:
            cache.set(CACHE_KEYS["RACE_HISTORY"], races)
            logger.info("✓ Cached race history in memory")
        
        return jsonify({
            "success": True,
            "data": races,
            "source": f"predictions_{getattr(prediction_logger, 'mode', 'unknown')}"
        })
    
    except Exception as e:
        return _internal_error("Race history endpoint failed", e)


# ---------- New Ergast + Wikipedia endpoints ----------
@app.route("/api/next-race", methods=["GET"])
def api_next_race():
    """Get next race information from FastF1 with circuit image from Wikipedia"""
    # Get next race from FastF1
    nr = get_next_race_from_fastf1(year=None)
    
    if not nr:
        # Fallback to Ergast if FastF1 doesn't have data
        nr = ergast_next_race() or {}
    
    if not nr:
        return jsonify({"success": False, "error": "No next race found"}), 404
    
    # Try to fetch a circuit image via Wikipedia.
    # Prefer local cached image (/images/...), but fall back to a remote Wikimedia URL.
    event_name = (nr.get("event_name") or nr.get("event") or "").strip()
    circuit_name = (nr.get("circuit") or "").strip()
    ergast_circuit_name = None
    if event_name and (not circuit_name or circuit_name == event_name):
        try:
            erg = ergast_next_race() or {}
            erg_event = (erg.get("event") or "").strip()
            if erg_event and erg_event.lower() == event_name.lower():
                ergast_circuit_name = (erg.get("circuit") or "").strip() or None
        except Exception:
            ergast_circuit_name = None
    race_year = nr.get("year") or nr.get("race_year")
    try:
        race_year = int(race_year) if race_year is not None else None
    except Exception:
        race_year = None

    candidates: list[str] = []
    def _add_candidate(value: str | None):
        v = (value or "").strip()
        if v and v not in candidates:
            candidates.append(v)

    _add_candidate(event_name)
    if race_year and event_name and not event_name.startswith(str(race_year)):
        _add_candidate(f"{race_year} {event_name}")
    if circuit_name and circuit_name != event_name:
        _add_candidate(circuit_name)
    if ergast_circuit_name and ergast_circuit_name != event_name and ergast_circuit_name != circuit_name:
        _add_candidate(ergast_circuit_name)

    for cand in candidates:
        # Fast path: return an already-cached local image (no network calls)
        h = hashlib.sha1(cand.encode()).hexdigest()
        for ext in ("jpg", "jpeg", "png", "webp", "gif", "svg"):
            fp = IMG_CACHE / f"wiki_{h}.{ext}"
            if fp.exists():
                return jsonify({
                    "success": True,
                    "race": nr,
                    "circuit_image_url": f"/images/{fp.name}",
                })

    for cand in candidates:
        # Prefer a remote Wikimedia URL to keep this endpoint fast in production.
        remote = _wikipedia_page_image_url(cand)
        if remote:
            return jsonify({
                "success": True,
                "race": nr,
                "circuit_image_url": remote,
            })

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

        season_year = datetime.now().year
        cache_source = None
        response_source = None
        actual_standings = None

        # Prefer Supabase cache when available (keeps requests fast + reduces Ergast load)
        if standings_cache is not None:
            cached = standings_cache.get_fresh(season=season_year, category="driver")
            if cached and isinstance(cached.payload, list) and len(cached.payload) > 0:
                actual_standings = cached.payload
                cache_source = cached.source
                response_source = f"Supabase cache ({cache_source or 'unknown'})"
                logger.info("✓ Using cached driver standings from Supabase (%s)", cache_source or "unknown")
        
        # If cache miss, fetch live and refresh the cache.
        if not actual_standings:
            actual_standings = get_2025_driver_standings_from_fastf1()
            if actual_standings and standings_cache is not None:
                standings_cache.upsert(
                    season=season_year,
                    category="driver",
                    payload=actual_standings,
                    source="FastF1",
                    ttl=STANDINGS_CACHE_TTL,
                )
            if actual_standings:
                response_source = "FastF1"

        # Final fallback: direct Ergast calls (no pandas/fastf1 dependency)
        if not actual_standings:
            logger.warning("No standings data from FastF1, using Ergast fallback")
            ergast_data = ergast_standings(str(season_year))
            drivers = ergast_data.get("drivers", [])

            actual_standings = []
            for d in drivers:
                actual_standings.append(
                    {
                        "position": d.get("position"),
                        "name": d.get("driver_name"),
                        "team": d.get("constructor"),
                        "points": int(d.get("points", 0)),
                        "code": d.get("driver_code"),
                    }
                )

            if actual_standings and standings_cache is not None:
                standings_cache.upsert(
                    season=season_year,
                    category="driver",
                    payload=actual_standings,
                    source="Ergast",
                    ttl=STANDINGS_CACHE_TTL,
                )
            response_source = "Ergast (fallback)"
        
        # Calculate predicted points for each driver (next 5 races)
        result_data = []
        for driver in actual_standings:
            driver_code = driver.get("code") or driver.get("driver_code")
            predicted_additional = predict_driver_points_for_future_races(
                driver_code,
                driver.get("points", 0),
                num_races=5
            )
            predicted_total = driver.get("points", 0) + predicted_additional
            
            result_data.append({
                "position": driver.get("position"),
                "driverName": driver.get("name") or driver.get("driver_name"),
                "teamName": driver.get("team") or driver.get("constructor"),
                "points": int(driver.get("points", 0)),
                "predictedPoints": int(predicted_total),
                "headshotUrl": None,
                "teamColor": "#1e3a8a"  # TODO: Add team color mapping
            })
        
        logger.info(f"✓ Driver standings complete: {len(result_data)} drivers")
        return jsonify({
            "success": True,
            "source": response_source or "FastF1",
            "data": result_data
        })
        
    except Exception as e:
        return _internal_error("Driver standings endpoint failed", e)

@app.route("/api/constructor-standings", methods=["GET"])
def api_constructor_standings():
    """
    Get 2025 constructor standings with actual points from Ergast
    and predicted future points based on aggregated driver predictions
    """
    try:
        logger.info("Fetching constructor standings endpoint...")

        season_year = datetime.now().year
        cache_source = None
        response_source = None
        actual_standings = None

        if standings_cache is not None:
            cached = standings_cache.get_fresh(season=season_year, category="constructor")
            if cached and isinstance(cached.payload, list) and len(cached.payload) > 0:
                actual_standings = cached.payload
                cache_source = cached.source
                response_source = f"Supabase cache ({cache_source or 'unknown'})"
                logger.info("✓ Using cached constructor standings from Supabase (%s)", cache_source or "unknown")
        
        # Cache miss: fetch live and refresh the cache.
        if not actual_standings:
            actual_standings = get_2025_constructor_standings_from_ergast()
            if actual_standings and standings_cache is not None:
                standings_cache.upsert(
                    season=season_year,
                    category="constructor",
                    payload=actual_standings,
                    source="Ergast",
                    ttl=STANDINGS_CACHE_TTL,
                )
            if actual_standings:
                response_source = "Ergast"
        
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
            "source": response_source or "Ergast",
            "data": result_data
        })
        
    except Exception as e:
        return _internal_error("Constructor standings endpoint failed", e)

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
        return _internal_error("Latest race circuit endpoint failed", e)


# Cache for qualifying session to avoid reloading
_cached_qualifying_session = None
_cached_qualifying_timestamp = None
_qualifying_session_lock = __import__('threading').Lock()

def get_cached_qualifying_session():
    """Load and cache the latest qualifying session with real FastF1 data"""
    global _cached_qualifying_session, _cached_qualifying_timestamp
    import time
    from datetime import datetime
    
    current_time = time.time()
    # Refresh cache every 30 minutes
    with _qualifying_session_lock:
        if _cached_qualifying_session is not None and (current_time - _cached_qualifying_timestamp) < 1800:
            logger.debug("Returning cached qualifying session")
            return _cached_qualifying_session
    
    try:
        logger.info("Loading latest qualifying session from FastF1...")
        
        # Prioritize current season, then prior seasons
        years_to_try = [2026, 2025, 2024]
        
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
                                    with _qualifying_session_lock:
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


# Cache for race session to avoid reloading (development use)
_cached_race_session = None
_cached_race_timestamp = None
_race_session_lock = __import__('threading').Lock()


def get_cached_race_session():
    """Load and cache the latest completed race session with telemetry."""
    global _cached_race_session, _cached_race_timestamp
    import time
    from datetime import datetime

    current_time = time.time()
    # Refresh cache every 30 minutes
    with _race_session_lock:
        if _cached_race_session is not None and (current_time - _cached_race_timestamp) < 1800:
            logger.debug("Returning cached race session")
            return _cached_race_session

    try:
        logger.info("Loading latest race session from FastF1...")

        years_to_try = [2026, 2025, 2024]

        for year in years_to_try:
            try:
                logger.info(f"Attempting to load race sessions for year {year}...")
                schedule = fastf1.get_event_schedule(year)
                schedule['EventDate'] = pd.to_datetime(schedule['EventDate'])
                today = pd.to_datetime(datetime.now().date())

                past_events = schedule[schedule['EventDate'] <= today]
                past_events = past_events.sort_values('EventDate', ascending=False)

                if len(past_events) == 0:
                    logger.debug(f"No completed events found for {year}")
                    continue

                logger.info(f"Found {len(past_events)} completed events for {year}")

                for _, event in past_events.iterrows():
                    event_name = event.get('EventName') or event.get('Event')
                    round_num = event.get('RoundNumber')
                    event_date = event.get('EventDate')

                    try:
                        logger.info(f"Attempting to load {event_name} (Round {round_num}) {year} race... ({event_date})")

                        session = fastf1.get_session(year, round_num, 'R')
                        session.load(laps=True, telemetry=True, weather=False)

                        if session.results is not None and len(session.results) > 0:
                            # Verify we can get telemetry from at least one driver
                            test_driver = session.results.iloc[0]['Abbreviation']
                            test_laps = session.laps.pick_drivers([test_driver])
                            if not test_laps.empty:
                                test_fast = test_laps.pick_fastest()
                                test_tel = test_fast.get_telemetry() if test_fast is not None else None
                                if test_tel is not None and not getattr(test_tel, 'empty', True):
                                    logger.info(
                                        f"✓ Successfully loaded {event_name} {year} race session with telemetry"
                                    )
                                    with _race_session_lock:
                                        _cached_race_session = session
                                        _cached_race_timestamp = current_time
                                    return session

                    except Exception as e:
                        logger.debug(f"Could not load {event_name}: {str(e)[:100]}")
                        continue

            except Exception as e:
                logger.debug(f"Error loading {year} schedule: {str(e)[:100]}")
                continue

        logger.error("Could not load any race session from FastF1")
        return None

    except Exception as e:
        logger.error(f"Error loading race session: {str(e)[:200]}")
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
        return _internal_error("Latest qualifying session endpoint failed", e)


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
        return _internal_error("Driver telemetry endpoint failed", e)


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
        
        return _internal_error("Qualifying circuit telemetry endpoint failed", e)


@app.route('/api/race-circuit-telemetry')
def race_circuit_telemetry():
    """Returns telemetry data for top 6 race finishers (cache-first).

    Query params: year, event (optional, defaults to latest)

    CACHING STRATEGY:
    - Latest session: Prefer Supabase cache when available
    - Production: Cache-only (prevents worker timeout)
    - Development: Cache if present, else loads fresh via FastF1
    """
    try:
        year = request.args.get('year', type=int)
        event = request.args.get('event', type=str)

        logger.info(f"Loading race telemetry for {year} {event}")

        # Cache-first for latest-session queries; production is cache-only.
        try:
            logger.info("✓ Checking Supabase for cached race telemetry")
            race_cache = get_race_telemetry_cache(config)
            cached_result = race_cache.get_latest_cached_race_telemetry()

            if cached_result is not None and ((not year and not event) or config.FLASK_ENV == "production"):
                logger.info("✓ Returning cached race telemetry from Supabase")
                return jsonify({
                    "success": True,
                    "drivers": cached_result if isinstance(cached_result, list) else [],
                    "source": "supabase_cache"
                }), 200
        except Exception as e:
            logger.debug(f"Supabase race telemetry cache check failed: {e}")

        # Cache miss on production: return helpful message instead of timeout
        if config.FLASK_ENV == "production":
            logger.warning("⚠️  No cached race telemetry found in production")
            return jsonify({
                "success": False,
                "error": "Race telemetry not yet cached",
                "message": "Race telemetry cache is populated after each race using: python scripts/cache_race_telemetry.py",
                "source": "cache_miss"
            }), 404

        # Development: allow fresh loads (no timeout risk locally)
        if year and event:
            logger.info(f"Development mode: loading fresh race data for {year} {event}")
            session = fastf1.get_session(year, event, 'R')
        else:
            session = get_cached_race_session()

        if session is None:
            return jsonify({
                "success": False,
                "error": "Could not load race session"
            }), 404

        session.load(laps=True, telemetry=True, weather=False)

        results = session.results.sort_values('Position')
        top_6_drivers = results.head(6)['Abbreviation'].tolist()

        drivers_data = []

        for idx, driver_code in enumerate(top_6_drivers):
            try:
                driver_row = session.results[session.results['Abbreviation'] == driver_code].iloc[0]

                driver_laps = session.laps.pick_drivers([driver_code])
                if driver_laps.empty:
                    continue

                fastest_lap = driver_laps.pick_fastest()
                telemetry = fastest_lap.get_telemetry()

                if telemetry.empty:
                    continue

                lap_time = fastest_lap['LapTime'].total_seconds() if pd.notna(fastest_lap['LapTime']) else 0

                s1 = fastest_lap.get('Sector1Time', np.nan)
                s2 = fastest_lap.get('Sector2Time', np.nan)
                s3 = fastest_lap.get('Sector3Time', np.nan)

                sector1_s = s1.total_seconds() if pd.notna(s1) else 0
                sector2_s = s2.total_seconds() if pd.notna(s2) else 0
                sector3_s = s3.total_seconds() if pd.notna(s3) else 0

                max_accel = telemetry['Acceleration'].max() if 'Acceleration' in telemetry.columns else 0
                avg_accel = telemetry['Acceleration'].mean() if 'Acceleration' in telemetry.columns else 0

                brake_events = 0
                if 'Brake' in telemetry.columns:
                    brake_events = (telemetry['Brake'] > 0).sum()

                circuit_trace = {
                    "x": telemetry['X'].astype(float).tolist(),
                    "y": telemetry['Y'].astype(float).tolist(),
                    "speed": telemetry['Speed'].astype(float).tolist()
                }

                driver_data = {
                    "code": str(driver_code),
                    "name": str(driver_row['FullName']),
                    "team": str(driver_row['TeamName']),
                    "race_position": int(driver_row['Position']) if pd.notna(driver_row['Position']) else None,
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
            "year": int(session.event.get('Year', session.event.get('year', datetime.now().year))),
            "event": session.event.get('EventName', 'Unknown'),
            "circuit": session.event.get('CircuitName', 'Unknown'),
            "round": int(session.event.get('RoundNumber', 0))
        }

        return jsonify({
            "success": True,
            "session_info": session_info,
            "drivers": drivers_data,
            "source": "fastf1_fresh"
        }), 200

    except Exception as e:
        logger.error(f"Error in race_circuit_telemetry: {str(e)[:200]}")
        logger.error(traceback.format_exc())
        return _internal_error("Race circuit telemetry endpoint failed", e)


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
        return _internal_error("Driver circuit details endpoint failed", e)


# ========== MLflow Model Registry Endpoints ==========

@app.route("/api/model-registry", methods=["GET"])
def model_registry():
    """
    Get complete model registry from MLflow
    Shows all registered models, versions, and metrics
    """
    try:
        from services.mlflow_manager import get_model_registry as _get_model_registry

        registry = _get_model_registry()
        return jsonify({
            "success": True,
            "data": registry
        }), 200
    except Exception as e:
        return _internal_error("Model registry endpoint failed", e)


@app.route("/api/model-metrics", methods=["GET"])
def model_metrics():
    """
    Get prediction accuracy metrics and model performance
    Includes overall accuracy, recent accuracy, and trends
    """
    try:
        from services.mlflow_manager import get_prediction_accuracy as _get_prediction_accuracy

        accuracy = _get_prediction_accuracy()
        return jsonify({
            "success": True,
            "data": accuracy,
            "timestamp": datetime.now().isoformat()
        }), 200
    except Exception as e:
        return _internal_error("Model metrics endpoint failed", e)


@app.route("/api/mlflow-status", methods=["GET"])
def mlflow_status():
    """
    Health check for MLflow and model status
    Returns current model versions and MLflow UI URL
    """
    try:
        from services.mlflow_manager import get_model_registry as _get_model_registry

        registry = _get_model_registry()
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
        return _internal_error("MLflow status endpoint failed", e)


# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    # Start background scheduler for auto-caching qualifying data
    logger.info("Starting background scheduler for automatic qualifying cache...")
    from services.scheduler import start_scheduler, stop_scheduler

    start_scheduler()
    
    try:
        # Run with Flask development server (for local development only)
        # In production, use: gunicorn -w 4 -b 0.0.0.0:5000 api:app
        app.run(
            debug=config.DEBUG,
            host=config.HOST,
            port=config.PORT,
            threaded=True
        )
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        stop_scheduler()

