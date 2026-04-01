"""F1 Prediction API - Database Layer (Simplified)

Supabase is used for:
1. Prediction logs (accuracy tracking)
2. Qualifying cache (avoid FastF1 rate limits)
3. Standings cache (optional; reduces live Ergast calls)

Historical race data stays in local Parquet files.
Driver features are pre-computed snapshots.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)

# Try to import Supabase client
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    logger.info("Supabase client not installed. Using local CSV mode.")


class PredictionLogger:
    """
    Handles prediction logging for accuracy tracking.
    Uses Supabase in production, local CSV in development.
    """
    
    def __init__(self, config):
        self.config = config
        self.supabase: Optional[Client] = None
        self._mode = "csv"
        
        self._initialize()
    
    def _initialize(self):
        """Initialize Supabase connection if configured"""
        if self.config.USE_SUPABASE and SUPABASE_AVAILABLE:
            try:
                supabase_service_key = getattr(self.config, "SUPABASE_SERVICE_KEY", None)
                supabase_key = supabase_service_key or getattr(self.config, "SUPABASE_KEY", None)
                if not self.config.SUPABASE_URL or not supabase_key:
                    raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY/SUPABASE_SERVICE_KEY")
                self.supabase = create_client(
                    self.config.SUPABASE_URL,
                    supabase_key
                )
                self._mode = "supabase"
                logger.info(
                    "✓ Connected to Supabase for prediction logging (%s key)",
                    "service" if supabase_service_key else "anon",
                )
            except Exception as e:
                logger.warning(f"Supabase connection failed: {e}. Using local CSV.")
                self._mode = "csv"
        else:
            self._mode = "csv"
            logger.info("✓ Using local CSV for prediction logging")
    
    @property
    def mode(self) -> str:
        return self._mode
    
    # =========================================================================
    # PREDICTION LOGGING
    # =========================================================================
    
    def log_prediction(
        self,
        race_name: str,
        predicted_winner: str,
        *,
        race_year: Optional[int] = None,
        circuit: Optional[str] = None,
        actual_winner: Optional[str] = None,
        confidence: Optional[float] = None,
        model_version: str = "v3",
        full_predictions: Optional[Any] = None,
        allow_update: bool = True,
    ) -> bool:
        """Log a prediction for tracking accuracy.

        In Supabase mode this is idempotent-ish: if a row for (race, race_year)
        exists and is missing actuals, update it instead of inserting a duplicate.
        """
        timestamp = datetime.now().isoformat()

        # Match schema.sql column names
        prediction: Dict[str, Any] = {
            "timestamp": timestamp,
            "race": race_name,
            "race_year": int(race_year) if race_year is not None else None,
            "circuit": circuit,
            "predicted": predicted_winner,
            "confidence": confidence,
            "model_version": model_version,
            "actual": actual_winner,
            "correct": (predicted_winner == actual_winner) if actual_winner else None,
            "full_predictions": full_predictions,
        }

        if self._mode == "supabase":
            try:
                # If we have a race_year, try to update an existing incomplete row first.
                existing_row = None
                if allow_update and race_year is not None:
                    resp = (
                        self.supabase.table("predictions")
                        .select("id, actual, correct")
                        .eq("race", race_name)
                        .eq("race_year", int(race_year))
                        .order("timestamp", desc=True)
                        .limit(1)
                        .execute()
                    )
                    if resp.data:
                        existing_row = resp.data[0]

                if existing_row and (existing_row.get("actual") in (None, "") or existing_row.get("correct") is None):
                    row_id = existing_row.get("id")
                    if row_id:
                        self.supabase.table("predictions").update(prediction).eq("id", row_id).execute()
                        logger.debug("✓ Updated existing prediction row in Supabase")
                        return True

                self.supabase.table("predictions").insert(prediction).execute()
                logger.debug("✓ Inserted prediction row to Supabase")
                return True
            except Exception as e:
                logger.error(f"Supabase insert/update failed: {e}")
                return self._log_to_csv(prediction)

        return self._log_to_csv(prediction)
    
    def _log_to_csv(self, prediction: dict) -> bool:
        """Fallback: log to local CSV"""
        try:
            log_path = Path("monitoring/predictions.csv")
            log_path.parent.mkdir(exist_ok=True)
            
            df = pd.DataFrame([prediction])
            if log_path.exists():
                df.to_csv(log_path, mode='a', header=False, index=False)
            else:
                df.to_csv(log_path, mode='w', header=True, index=False)
            
            return True
        except Exception as e:
            logger.error(f"CSV logging failed: {e}")
            return False
    
    def update_actual_winner(self, race_name: str, actual_winner: str, *, race_year: Optional[int] = None) -> bool:
        """Update prediction(s) with actual winner after race completes."""
        if self._mode == "supabase":
            try:
                # Update rows with missing actual.
                q = (
                    self.supabase.table("predictions")
                    .update({"actual": actual_winner, "correct": None})
                    .eq("race", race_name)
                    .is_("actual", "null")
                )
                if race_year is not None:
                    q = q.eq("race_year", int(race_year))
                q.execute()
                
                # Update correct flag where predicted matches actual
                q_true = self.supabase.table("predictions").update({"correct": True}).eq("race", race_name).eq("predicted", actual_winner)
                if race_year is not None:
                    q_true = q_true.eq("race_year", int(race_year))
                q_true.execute()
                
                # Update correct flag where predicted doesn't match actual
                q_false = (
                    self.supabase.table("predictions")
                    .update({"correct": False})
                    .eq("race", race_name)
                    .neq("predicted", actual_winner)
                    .is_("correct", "null")
                )
                if race_year is not None:
                    q_false = q_false.eq("race_year", int(race_year))
                q_false.execute()
                
                return True
            except Exception as e:
                logger.error(f"Failed to update actual winner: {e}")
                return False
        
        # CSV mode: would need to read, update, write
        logger.warning("Cannot update actual winner in CSV mode")
        return False

    def get_latest_prediction(self, race_name: str, *, race_year: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Fetch the most recent prediction row for a race."""
        if self._mode == "supabase":
            try:
                q = self.supabase.table("predictions").select("*").eq("race", race_name)
                if race_year is not None:
                    q = q.eq("race_year", int(race_year))
                resp = q.order("timestamp", desc=True).limit(1).execute()
                if resp.data:
                    return resp.data[0]
            except Exception as e:
                logger.error(f"Failed to fetch latest prediction (Supabase REST): {e}")

        # CSV mode
        try:
            df = self.get_prediction_history(limit=5000)
            if df is None or df.empty:
                return None
            df = df[df["race"] == race_name]
            if race_year is not None and "race_year" in df.columns:
                df = df[df["race_year"].astype("Int64") == int(race_year)]
            if df.empty:
                return None
            row = df.sort_values("timestamp", ascending=False).iloc[0].to_dict()
            for key, value in list(row.items()):
                try:
                    if pd.isna(value):
                        row[key] = None
                except Exception:
                    pass
            return row
        except Exception:
            return None

    def get_most_recent_prediction(self) -> Optional[Dict[str, Any]]:
        """Fetch the most recent prediction row across all races.

        Works in both Supabase and CSV mode.
        """
        try:
            df = self.get_prediction_history(limit=1)
            if df is None or df.empty:
                return None

            row = df.iloc[0].to_dict()

            # Normalize pandas NaN/NaT to None
            for key, value in list(row.items()):
                try:
                    if pd.isna(value):
                        row[key] = None
                except Exception:
                    pass
            return row
        except Exception as e:
            logger.error(f"Failed to fetch most recent prediction: {e}")
            return None

    def get_most_recent_prediction_with_full_predictions(self, *, limit: int = 200) -> Optional[Dict[str, Any]]:
        """Fetch the most recent prediction row that includes driver-level predictions.

        Some historical rows only store summary metadata (winner/top3) in `full_predictions`.
        The frontend driver cards require a non-empty `full_predictions` list.
        """
        try:
            df = self.get_prediction_history(limit=int(limit))
            if df is None or df.empty:
                return None

            for _, r in df.iterrows():
                row = r.to_dict()

                # Normalize pandas NaN/NaT to None
                for key, value in list(row.items()):
                    try:
                        if pd.isna(value):
                            row[key] = None
                    except Exception:
                        pass

                predicted = row.get("predicted")
                if not predicted:
                    continue

                fp = row.get("full_predictions")
                if isinstance(fp, str):
                    try:
                        fp = json.loads(fp)
                    except Exception:
                        fp = None

                # Support multiple shapes:
                # - { ..., "full_predictions": [ ... ] }
                # - { ..., "predictions": { "full_predictions": [ ... ] } }
                fp_data = fp
                if isinstance(fp, dict) and isinstance(fp.get("predictions"), dict):
                    fp_data = fp.get("predictions")

                if isinstance(fp_data, dict):
                    preds = fp_data.get("full_predictions")
                    if isinstance(preds, list) and len(preds) > 0:
                        return row

            return None
        except Exception as e:
            logger.error(f"Failed to fetch most recent prediction with full_predictions: {e}")
            return None

    def get_predictions_missing_actual(self, *, limit: int = 200) -> List[Dict[str, Any]]:
        """Return prediction rows that have not been backfilled with actual winner yet."""
        if self._mode == "supabase":
            try:
                resp = (
                    self.supabase.table("predictions")
                    .select("id, timestamp, race, race_year, circuit, predicted")
                    .is_("actual", "null")
                    .order("timestamp", desc=True)
                    .limit(limit)
                    .execute()
                )
                data = list(resp.data or [])
                return data
            except Exception as e:
                logger.error(f"Failed to fetch predictions missing actual (Supabase REST): {e}")

        # CSV mode
        try:
            df = self.get_prediction_history(limit=5000)
            if df is None or df.empty:
                return []
            if "actual" not in df.columns:
                return []
            missing = df[df["actual"].isna()].copy()
            if missing.empty:
                return []
            missing = missing.sort_values("timestamp", ascending=False).head(int(limit))
            rows: List[Dict[str, Any]] = []
            for _, r in missing.iterrows():
                row = r.to_dict()
                for key, value in list(row.items()):
                    try:
                        if pd.isna(value):
                            row[key] = None
                    except Exception:
                        pass
                rows.append(row)
            return rows
        except Exception:
            return []
    
    def get_prediction_history(self, limit: int = 100) -> pd.DataFrame:
        """Get prediction history for accuracy analysis"""
        if self._mode == "supabase":
            try:
                response = self.supabase.table('predictions') \
                    .select('*') \
                    .order('timestamp', desc=True) \
                    .limit(limit) \
                    .execute()
                return pd.DataFrame(list(response.data or []))
            except Exception as e:
                logger.error(f"Query failed: {e}")
                return pd.DataFrame()
        
        # Fallback to CSV
        log_path = Path("monitoring/predictions.csv")
        if log_path.exists():
            return pd.read_csv(log_path).tail(limit)
        return pd.DataFrame()
    
    def get_accuracy_stats(self) -> Dict[str, Any]:
        """Calculate prediction accuracy statistics"""
        df = self.get_prediction_history(limit=1000)
        
        if df.empty or 'correct' not in df.columns:
            return {
                "total_predictions": 0,
                "correct_predictions": 0,
                "accuracy_pct": 0,
                "recent_accuracy_pct": 0,
                "trend": []
            }
        
        # Filter rows where we have actual results
        completed = df[df['correct'].notna()]
        
        if completed.empty:
            return {
                "total_predictions": len(df),
                "correct_predictions": 0,
                "accuracy_pct": 0,
                "recent_accuracy_pct": 0,
                "trend": []
            }
        
        completed['correct'] = completed['correct'].astype(bool)
        
        total = len(completed)
        correct = completed['correct'].sum()
        accuracy = (correct / total * 100) if total > 0 else 0
        
        # Recent accuracy (last 10)
        recent = completed.tail(10)
        recent_acc = (recent['correct'].sum() / len(recent) * 100) if len(recent) > 0 else 0
        
        # Trend (last 20)
        trend = completed.tail(20)['correct'].astype(int).tolist()
        
        return {
            "total_predictions": total,
            "correct_predictions": int(correct),
            "accuracy_pct": round(accuracy, 2),
            "recent_accuracy_pct": round(recent_acc, 2),
            "trend": trend
        }


class QualifyingCache:
    """
    Cache qualifying results to avoid FastF1 API rate limits.
    Uses Supabase in production, Redis/local in development.
    Falls back to file cache if Supabase unavailable.
    """
    
    def __init__(self, config):
        self.config = config
        self.supabase: Optional[Client] = None
        self._local_cache: Dict[str, dict] = {}
        self._mode = "local"
        
        self._initialize()
        self._load_file_cache()  # Load from f1_cache directory
    
    def _initialize(self):
        """Initialize cache backend"""
        if self.config.USE_DATABASE and SUPABASE_AVAILABLE:
            try:
                supabase_key = getattr(self.config, "SUPABASE_SERVICE_KEY", None) or getattr(self.config, "SUPABASE_KEY", None)
                if not self.config.SUPABASE_URL or not supabase_key:
                    raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY/SUPABASE_SERVICE_KEY")
                self.supabase = create_client(
                    self.config.SUPABASE_URL,
                    supabase_key
                )
                self._mode = "supabase"
                logger.info("✓ Using Supabase for qualifying cache")
            except Exception as e:
                logger.warning(f"Supabase connection failed: {e}")
                self._mode = "local"
    
    def _load_file_cache(self):
        """Load qualifying data from f1_cache directory into memory"""
        try:
            from pathlib import Path
            cache_dir = Path(__file__).parent.parent / 'f1_cache'
            
            if not cache_dir.exists():
                return
            
            # Load all qualifying JSON files
            for year_dir in cache_dir.iterdir():
                if year_dir.is_dir():
                    for json_file in year_dir.glob('*_qualifying.json'):
                        try:
                            race_key = json_file.stem.rsplit('_qualifying', 1)[0]
                            with open(json_file, 'r') as f:
                                import json
                                qual_data = json.load(f)
                                expires_at = datetime.now() + timedelta(hours=24)
                                self._local_cache[race_key] = {
                                    'data': qual_data,
                                    'expires_at': expires_at
                                }
                                logger.debug(f"✓ Loaded {race_key} from file cache")
                        except Exception as e:
                            logger.debug(f"Could not load {json_file}: {e}")
        except Exception as e:
            logger.debug(f"File cache loading failed: {e}")
    
    def cache_qualifying(self, race_key: str, race_year: int, 
                        qualifying_data: list, ttl_hours: int = 24):
        """Cache qualifying results"""
        expires_at = datetime.now() + timedelta(hours=ttl_hours)
        
        if self._mode == "supabase":
            try:
                self.supabase.table('qualifying_cache').upsert({
                    'race_key': race_key,
                    'race_year': race_year,
                    'qualifying_data': qualifying_data,  # JSONB stores as-is, no json.dumps needed
                    'cached_at': datetime.now().isoformat(),
                    'expires_at': expires_at.isoformat()
                }).execute()
                logger.debug(f"✓ Cached qualifying for {race_key}")
            except Exception as e:
                logger.error(f"Cache write failed: {e}")
                # Fallback to local
                self._local_cache[race_key] = {
                    'data': qualifying_data,
                    'expires_at': expires_at
                }
        else:
            self._local_cache[race_key] = {
                'data': qualifying_data,
                'expires_at': expires_at
            }
    
    def get_cached_qualifying(self, race_key: str) -> Optional[list]:
        """Get cached qualifying results if not expired"""
        if self._mode == "supabase":
            try:
                response = self.supabase.table('qualifying_cache') \
                    .select('qualifying_data, expires_at') \
                    .eq('race_key', race_key) \
                    .gt('expires_at', datetime.now().isoformat()) \
                    .execute()
                
                if response.data:
                    # JSONB comes back as native list/dict, no json.loads needed
                    return response.data[0]['qualifying_data']
            except Exception as e:
                logger.error(f"Cache read failed: {e}")
        
        # Check local cache
        if race_key in self._local_cache:
            cached = self._local_cache[race_key]
            if cached['expires_at'] > datetime.now():
                return cached['data']
            else:
                del self._local_cache[race_key]
        
        return None
    
    def get_latest_cached_qualifying(self) -> Optional[list]:
        """Get the LATEST cached qualifying results (most recent by cached_at)"""
        if self._mode == "supabase":
            try:
                response = self.supabase.table('qualifying_cache') \
                    .select('qualifying_data, expires_at, race_key') \
                    .gt('expires_at', datetime.now().isoformat()) \
                    .order('cached_at', desc=True) \
                    .limit(1) \
                    .execute()
                
                if response.data:
                    logger.info(f"✓ Found latest cached qualifying: {response.data[0].get('race_key')}")
                    return response.data[0]['qualifying_data']

                # If nothing is currently valid, fall back to the most recent cached
                # entry even if it is expired. This keeps the UI populated until a
                # new cache run updates Supabase.
                stale = self.supabase.table('qualifying_cache') \
                    .select('qualifying_data, expires_at, race_key') \
                    .order('cached_at', desc=True) \
                    .limit(1) \
                    .execute()
                if stale.data:
                    logger.info(f"⚠ Using stale cached qualifying: {stale.data[0].get('race_key')}")
                    return stale.data[0]['qualifying_data']
            except Exception as e:
                logger.error(f"Cache read failed: {e}")
        
        # Check local cache for most recent entry
        if self._local_cache:
            latest_key = max(self._local_cache.keys(), 
                           key=lambda k: self._local_cache[k]['expires_at'])
            cached = self._local_cache[latest_key]
            if cached['expires_at'] > datetime.now():
                return cached['data']
        
        return None
    
    def clear_expired(self):
        """Clean up expired cache entries"""
        if self._mode == "supabase":
            try:
                self.supabase.table('qualifying_cache') \
                    .delete() \
                    .lt('expires_at', datetime.now().isoformat()) \
                    .execute()
            except Exception as e:
                logger.error(f"Cache cleanup failed: {e}")
        
        # Clean local cache
        now = datetime.now()
        expired = [k for k, v in self._local_cache.items() if v['expires_at'] < now]
        for k in expired:
            del self._local_cache[k]


# =============================================================================
# SIMPLIFIED SCHEMA (What Supabase actually needs)
# =============================================================================
SUPABASE_SCHEMA = """
-- Prediction logs (for accuracy tracking)
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    race_name TEXT NOT NULL,
    predicted_winner TEXT NOT NULL,
    actual_winner TEXT,
    correct BOOLEAN,
    confidence FLOAT,
    model_version TEXT DEFAULT 'v3'
);

CREATE INDEX IF NOT EXISTS idx_predictions_race ON predictions(race_name);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp DESC);

-- Qualifying cache (avoid FastF1 rate limits)
CREATE TABLE IF NOT EXISTS qualifying_cache (
    race_key TEXT PRIMARY KEY,
    race_year INT NOT NULL,
    data JSONB NOT NULL,
    fetched_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_qualifying_expires ON qualifying_cache(expires_at);

-- Enable Row Level Security
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE qualifying_cache ENABLE ROW LEVEL SECURITY;

-- Allow read for all, write only for authenticated service role
CREATE POLICY "Allow read on predictions" ON predictions FOR SELECT USING (true);
CREATE POLICY "Allow insert on predictions" ON predictions FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow read on qualifying_cache" ON qualifying_cache FOR SELECT USING (true);
CREATE POLICY "Allow insert/update on qualifying_cache" ON qualifying_cache FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow update on qualifying_cache" ON qualifying_cache FOR UPDATE USING (true);
"""


# =============================================================================
# SINGLETON INSTANCES
# =============================================================================
_prediction_logger: Optional[PredictionLogger] = None
_qualifying_cache: Optional[QualifyingCache] = None

def get_prediction_logger(config=None) -> PredictionLogger:
    """Get or create prediction logger singleton"""
    global _prediction_logger
    if _prediction_logger is None:
        if config is None:
            from utils.config import config as app_config
            config = app_config
        _prediction_logger = PredictionLogger(config)
    return _prediction_logger

def get_qualifying_cache(config=None) -> QualifyingCache:
    """Get or create qualifying cache singleton"""
    global _qualifying_cache
    if _qualifying_cache is None:
        if config is None:
            from utils.config import config as app_config
            config = app_config
        _qualifying_cache = QualifyingCache(config)
    return _qualifying_cache
