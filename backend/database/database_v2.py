"""
F1 Prediction API - Database Layer (Simplified)

Supabase is used ONLY for:
1. Prediction logs (accuracy tracking)
2. Qualifying cache (avoid FastF1 rate limits)

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
        if self.config.USE_DATABASE and SUPABASE_AVAILABLE:
            try:
                self.supabase = create_client(
                    self.config.SUPABASE_URL,
                    self.config.SUPABASE_KEY
                )
                self._mode = "supabase"
                logger.info("✓ Connected to Supabase for prediction logging")
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
    
    def log_prediction(self, race_name: str, predicted_winner: str,
                      actual_winner: str = None, confidence: float = None,
                      model_version: str = "v3") -> bool:
        """Log a prediction for tracking accuracy"""
        # Match schema.sql column names
        prediction = {
            'timestamp': datetime.now().isoformat(),
            'race': race_name,
            'predicted': predicted_winner,
            'actual': actual_winner,
            'correct': predicted_winner == actual_winner if actual_winner else None,
            'confidence': confidence,
            'model_version': model_version
        }
        
        if self._mode == "supabase":
            try:
                self.supabase.table('predictions').insert(prediction).execute()
                logger.debug(f"✓ Logged prediction to Supabase")
                return True
            except Exception as e:
                logger.error(f"Supabase insert failed: {e}")
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
    
    def update_actual_winner(self, race_name: str, actual_winner: str) -> bool:
        """Update prediction with actual winner after race completes"""
        if self._mode == "supabase":
            try:
                # Find prediction for this race and update actual winner
                response = self.supabase.table('predictions') \
                    .update({
                        'actual': actual_winner,
                        'correct': None  # Will be computed next
                    }) \
                    .eq('race', race_name) \
                    .is_('actual', 'null') \
                    .execute()
                
                # Update correct flag where predicted matches actual
                self.supabase.table('predictions') \
                    .update({'correct': True}) \
                    .eq('race', race_name) \
                    .eq('predicted', actual_winner) \
                    .execute()
                
                # Update correct flag where predicted doesn't match actual
                self.supabase.table('predictions') \
                    .update({'correct': False}) \
                    .eq('race', race_name) \
                    .neq('predicted', actual_winner) \
                    .is_('correct', 'null') \
                    .execute()
                
                return True
            except Exception as e:
                logger.error(f"Failed to update actual winner: {e}")
                return False
        
        # CSV mode: would need to read, update, write
        logger.warning("Cannot update actual winner in CSV mode")
        return False
    
    def get_prediction_history(self, limit: int = 100) -> pd.DataFrame:
        """Get prediction history for accuracy analysis"""
        if self._mode == "supabase":
            try:
                response = self.supabase.table('predictions') \
                    .select('*') \
                    .order('timestamp', desc=True) \
                    .limit(limit) \
                    .execute()
                return pd.DataFrame(response.data)
            except Exception as e:
                logger.error(f"Query failed: {e}")
        
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
    """
    
    def __init__(self, config):
        self.config = config
        self.supabase: Optional[Client] = None
        self._local_cache: Dict[str, dict] = {}
        self._mode = "local"
        
        self._initialize()
    
    def _initialize(self):
        """Initialize cache backend"""
        if self.config.USE_DATABASE and SUPABASE_AVAILABLE:
            try:
                self.supabase = create_client(
                    self.config.SUPABASE_URL,
                    self.config.SUPABASE_KEY
                )
                self._mode = "supabase"
                logger.info("✓ Using Supabase for qualifying cache")
            except Exception as e:
                logger.warning(f"Supabase connection failed: {e}")
                self._mode = "local"
    
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

-- Allow public read/write for now (tighten in production)
CREATE POLICY "Allow all on predictions" ON predictions FOR ALL USING (true);
CREATE POLICY "Allow all on qualifying_cache" ON qualifying_cache FOR ALL USING (true);
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
            from config import config as app_config
            config = app_config
        _prediction_logger = PredictionLogger(config)
    return _prediction_logger

def get_qualifying_cache(config=None) -> QualifyingCache:
    """Get or create qualifying cache singleton"""
    global _qualifying_cache
    if _qualifying_cache is None:
        if config is None:
            from config import config as app_config
            config = app_config
        _qualifying_cache = QualifyingCache(config)
    return _qualifying_cache
