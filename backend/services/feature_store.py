"""
F1 Prediction API - Feature Store
Efficient feature retrieval with caching layers:
1. In-memory LRU cache (fastest, per-process)
2. Redis cache (shared across instances, optional)
3. Parquet file (cold storage, columnar for fast filtered reads)
"""

import os
import json
import logging
from pathlib import Path
from functools import lru_cache
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Try to import Redis (optional for production)
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.info("Redis not installed. Using in-memory cache only.")


class FeatureStore:
    """
    Efficient feature retrieval with multi-layer caching:
    
    Layer 1: LRU cache (in-memory, 128 drivers)
    Layer 2: Redis (shared across instances, 7-day TTL)
    Layer 3: Parquet file (columnar, filtered reads)
    
    Usage:
        store = FeatureStore(config)
        features = store.get_driver_features("VER")
    """
    
    def __init__(self, config):
        self.config = config
        self.redis_client: Optional[redis.Redis] = None
        
        # Paths
        self.parquet_path = config.DATA_PATH.with_suffix('.parquet')
        self.features_snapshot_path = config.MODEL_DIR / "driver_features_snapshot.parquet"
        self.circuit_features_path = config.MODEL_DIR / "circuit_features_snapshot.parquet"
        self.team_features_path = config.MODEL_DIR / "team_features_snapshot.parquet"
        
        # Cache settings
        self.redis_ttl = 7 * 24 * 3600  # 7 days
        self.qualifying_ttl = 24 * 3600  # 24 hours
        
        # In-memory cache for features (loaded once at startup)
        self._driver_features_cache: Optional[pd.DataFrame] = None
        self._circuit_features_cache: Optional[pd.DataFrame] = None
        self._team_features_cache: Optional[pd.DataFrame] = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize connections and load snapshots"""
        # Connect to Redis if configured
        if REDIS_AVAILABLE and self.config.REDIS_URL:
            try:
                self.redis_client = redis.from_url(
                    self.config.REDIS_URL,
                    decode_responses=True
                )
                self.redis_client.ping()
                logger.info("✓ Connected to Redis cache")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Using in-memory only.")
                self.redis_client = None
        
        # Load driver features snapshot if exists
        if self.features_snapshot_path.exists():
            self._driver_features_cache = pd.read_parquet(self.features_snapshot_path)
            logger.info(f"✓ Loaded driver features snapshot: {len(self._driver_features_cache)} drivers")
        else:
            logger.warning(f"Driver features snapshot not found at {self.features_snapshot_path}")
            logger.info("  Run 'python build_feature_snapshots.py' to generate")
        
        # Load circuit features snapshot if exists
        if self.circuit_features_path.exists():
            self._circuit_features_cache = pd.read_parquet(self.circuit_features_path)
            logger.info(f"✓ Loaded circuit features snapshot: {len(self._circuit_features_cache)} driver-circuit pairs")
        else:
            logger.warning(f"Circuit features snapshot not found at {self.circuit_features_path}")
        
        # Load team features snapshot if exists
        if self.team_features_path.exists():
            self._team_features_cache = pd.read_parquet(self.team_features_path)
            logger.info(f"✓ Loaded team features snapshot: {len(self._team_features_cache)} team-year pairs")
    
    # =========================================================================
    # DRIVER FEATURES (Hot Path - Used in every prediction)
    # =========================================================================
    
    def get_driver_features(self, driver_code: str) -> Dict[str, Any]:
        """
        Get pre-computed features for a driver.
        Returns dict with: RecentFormAvg, EloRating, TotalRaces, etc.
        
        Cache layers:
        1. In-memory DataFrame (loaded at startup)
        2. Redis (if configured)
        3. Compute from Parquet (fallback)
        """
        # Layer 1: In-memory snapshot
        if self._driver_features_cache is not None:
            driver_row = self._driver_features_cache[
                self._driver_features_cache['driver'] == driver_code
            ]
            if not driver_row.empty:
                return driver_row.iloc[0].to_dict()
        
        # Layer 2: Redis cache
        if self.redis_client:
            cache_key = f"driver_features:{driver_code}"
            cached = self.redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        
        # Layer 3: Compute from Parquet (slow path)
        features = self._compute_driver_features_from_parquet(driver_code)
        
        # Cache in Redis for next time
        if self.redis_client and features:
            self.redis_client.setex(
                f"driver_features:{driver_code}",
                self.redis_ttl,
                json.dumps(features)
            )
        
        return features
    
    @lru_cache(maxsize=128)
    def get_driver_recent_form(self, driver_code: str, n_races: int = 10) -> pd.DataFrame:
        """
        Get last N races for ONE driver (not entire dataset).
        Uses Parquet predicate pushdown for efficient filtered reads.
        """
        # Try Redis first
        if self.redis_client:
            cache_key = f"driver_form:{driver_code}:{n_races}"
            cached = self.redis_client.get(cache_key)
            if cached:
                return pd.read_json(cached)
        
        # Read from Parquet with filter (columnar = fast)
        if self.parquet_path.exists():
            try:
                df = pd.read_parquet(
                    self.parquet_path,
                    filters=[("driver", "==", driver_code)],
                    columns=["race_year", "event", "circuit", "finishing_position", 
                            "qualifying_position", "points", "event_date"]
                )
                df = df.sort_values("event_date", ascending=False).head(n_races)
                
                # Cache in Redis
                if self.redis_client:
                    self.redis_client.setex(
                        f"driver_form:{driver_code}:{n_races}",
                        self.redis_ttl,
                        df.to_json()
                    )
                
                return df
            except Exception as e:
                logger.error(f"Parquet read failed: {e}")
        
        # Fallback to CSV (slow)
        return self._get_driver_form_from_csv(driver_code, n_races)
    
    def _compute_driver_features_from_parquet(self, driver_code: str) -> Dict[str, Any]:
        """Compute features from Parquet file (fallback path)"""
        if not self.parquet_path.exists():
            logger.warning(f"Parquet file not found: {self.parquet_path}")
            return self._get_default_features(driver_code)
        
        try:
            # Read only this driver's data
            df = pd.read_parquet(
                self.parquet_path,
                filters=[("driver", "==", driver_code)]
            )
            
            if df.empty:
                return self._get_default_features(driver_code)
            
            df = df.sort_values("event_date", ascending=False)
            
            # Compute features
            recent_5 = df.head(5)["finishing_position"].mean()
            total_races = len(df)
            avg_finish = df["finishing_position"].mean()
            
            return {
                "driver": driver_code,
                "RecentFormAvg": recent_5 if pd.notna(recent_5) else 10.0,
                "TotalRaces": total_races,
                "DriverExperienceScore": min(total_races / 200, 1.0),  # Normalize to max ~200 races
                "EloRating": df.iloc[0].get("EloRating", 1500.0) if "EloRating" in df.columns else 1500.0,
                "AvgFinishPosition": avg_finish if pd.notna(avg_finish) else 10.0
            }
            
        except Exception as e:
            logger.error(f"Failed to compute features for {driver_code}: {e}")
            return self._get_default_features(driver_code)
    
    def _get_driver_form_from_csv(self, driver_code: str, n_races: int) -> pd.DataFrame:
        """Fallback: read from CSV (slow)"""
        csv_path = self.config.DATA_PATH
        if not csv_path.exists():
            return pd.DataFrame()
        
        df = pd.read_csv(csv_path)
        driver_df = df[df["driver"] == driver_code].sort_values("event_date", ascending=False)
        return driver_df.head(n_races)
    
    def _get_default_features(self, driver_code: str) -> Dict[str, Any]:
        """Default features for unknown drivers (rookies)"""
        return {
            "driver": driver_code,
            "RecentFormAvg": 10.0,  # Median position
            "TotalRaces": 0,
            "DriverExperienceScore": 0.0,
            "EloRating": 1500.0,  # Starting ELO
            "AvgFinishPosition": 10.0
        }
    
    # =========================================================================
    # CIRCUIT FEATURES
    # =========================================================================
    
    def get_circuit_history(self, driver_code: str, circuit: str) -> float:
        """
        Get driver's average finishing position at a specific circuit.
        Uses fuzzy matching since training data has full event names
        (e.g., "FORMULA 1 GRAND PRIX DE MONACO 2018") but API receives
        simple names like "Monaco" or "Las Vegas".
        
        Returns median (10.0) if no history.
        """
        # Try in-memory circuit cache (DataFrame lookup with fuzzy match)
        if self._circuit_features_cache is not None:
            # First try exact match
            match = self._circuit_features_cache[
                (self._circuit_features_cache['driver'] == driver_code) &
                (self._circuit_features_cache['circuit'] == circuit)
            ]
            if not match.empty:
                return float(match.iloc[0].get('CircuitAvgFinish', 10.0))
            
            # Fuzzy match: circuit name contains the search term (case-insensitive)
            # e.g., "Monaco" matches "FORMULA 1 GRAND PRIX DE MONACO 2018"
            match = self._circuit_features_cache[
                (self._circuit_features_cache['driver'] == driver_code) &
                (self._circuit_features_cache['circuit'].str.contains(circuit, case=False, na=False))
            ]
            if not match.empty:
                # Average across all matching circuits (e.g., Monaco 2018, Monaco 2019, etc.)
                avg = match['CircuitAvgFinish'].mean()
                return float(avg)
        
        # Try Redis
        cache_key = f"circuit:{driver_code}:{circuit}"
        if self.redis_client:
            cached = self.redis_client.get(cache_key)
            if cached:
                return float(cached)
        
        # Compute from Parquet with fuzzy match (slow path)
        if self.parquet_path.exists():
            try:
                # Read all data for this driver
                df = pd.read_parquet(
                    self.parquet_path,
                    filters=[("driver", "==", driver_code)],
                    columns=["circuit", "finishing_position"]
                )
                
                # Fuzzy filter by circuit name
                df = df[df['circuit'].str.contains(circuit, case=False, na=False)]
                
                if not df.empty:
                    avg = df["finishing_position"].mean()
                    
                    # Cache result in Redis
                    if self.redis_client:
                        self.redis_client.setex(cache_key, self.redis_ttl, str(avg))
                    
                    return avg
            except Exception as e:
                logger.debug(f"Circuit history lookup failed: {e}")
        
        return 10.0  # Default median
    
    # =========================================================================
    # TEAM FEATURES
    # =========================================================================
    
    def get_team_perf_score(self, team: str, race_year: int) -> float:
        """
        Get team performance score for a given year.
        Uses pre-computed snapshot, falls back to previous year if current not available.
        Returns 0.5 (median) if no history.
        """
        # Try in-memory team cache (DataFrame lookup)
        if self._team_features_cache is not None:
            # Try exact year match first
            match = self._team_features_cache[
                (self._team_features_cache['team'] == team) &
                (self._team_features_cache['race_year'] == race_year)
            ]
            if not match.empty:
                return float(match.iloc[0].get('TeamPerfScore', 0.5))
            
            # Fallback to previous year
            match = self._team_features_cache[
                (self._team_features_cache['team'] == team) &
                (self._team_features_cache['race_year'] == race_year - 1)
            ]
            if not match.empty:
                return float(match.iloc[0].get('TeamPerfScore', 0.5))
            
            # Try partial team name match (e.g., "Red Bull" matches "Red Bull Racing")
            match = self._team_features_cache[
                (self._team_features_cache['team'].str.contains(team, case=False, na=False)) &
                (self._team_features_cache['race_year'] >= race_year - 1)
            ]
            if not match.empty:
                # Get most recent
                match = match.sort_values('race_year', ascending=False)
                return float(match.iloc[0].get('TeamPerfScore', 0.5))
        
        # Try Redis
        cache_key = f"team:{team}:{race_year}"
        if self.redis_client:
            cached = self.redis_client.get(cache_key)
            if cached:
                return float(cached)
        
        return 0.5  # Default median team performance
    
    # =========================================================================
    # QUALIFYING CACHE (Supabase or Redis)
    # =========================================================================
    
    def cache_qualifying(self, race_key: str, qualifying_data: list, ttl: int = None):
        """Cache qualifying results to avoid FastF1 rate limits"""
        ttl = ttl or self.qualifying_ttl
        
        if self.redis_client:
            self.redis_client.setex(
                f"qualifying:{race_key}",
                ttl,
                json.dumps(qualifying_data)
            )
            logger.debug(f"Cached qualifying for {race_key}")
    
    def get_cached_qualifying(self, race_key: str) -> Optional[list]:
        """Get cached qualifying results"""
        if self.redis_client:
            cached = self.redis_client.get(f"qualifying:{race_key}")
            if cached:
                logger.debug(f"Cache hit for qualifying:{race_key}")
                return json.loads(cached)
        return None
    
    # =========================================================================
    # BULK FEATURE RETRIEVAL (For prediction)
    # =========================================================================
    
    def get_features_for_drivers(self, driver_codes: list, circuit: str) -> pd.DataFrame:
        """
        Get features for multiple drivers efficiently.
        Used during prediction to avoid N+1 queries.
        """
        features_list = []
        
        for driver in driver_codes:
            driver_features = self.get_driver_features(driver)
            driver_features["CircuitHistoryAvg"] = self.get_circuit_history(driver, circuit)
            features_list.append(driver_features)
        
        return pd.DataFrame(features_list)
    
    # =========================================================================
    # CACHE MANAGEMENT
    # =========================================================================
    
    def clear_driver_cache(self, driver_code: str = None):
        """Clear cache for a driver (after race results update)"""
        if driver_code:
            # Clear specific driver
            self.get_driver_recent_form.cache_clear()
            if self.redis_client:
                self.redis_client.delete(f"driver_features:{driver_code}")
                # Clear all form caches for this driver
                for key in self.redis_client.scan_iter(f"driver_form:{driver_code}:*"):
                    self.redis_client.delete(key)
        else:
            # Clear all
            self.get_driver_recent_form.cache_clear()
            if self.redis_client:
                for key in self.redis_client.scan_iter("driver_*"):
                    self.redis_client.delete(key)
    
    def health_check(self) -> Dict[str, Any]:
        """Check feature store status"""
        return {
            "parquet_exists": self.parquet_path.exists(),
            "features_snapshot_exists": self.features_snapshot_path.exists(),
            "redis_connected": self.redis_client is not None,
            "drivers_in_memory": len(self._driver_features_cache) if self._driver_features_cache is not None else 0,
            "lru_cache_info": self.get_driver_recent_form.cache_info()._asdict()
        }


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================
_store_instance: Optional[FeatureStore] = None

def get_feature_store(config=None) -> FeatureStore:
    """Get or create feature store singleton"""
    global _store_instance
    
    if _store_instance is None:
        if config is None:
            from config import config as app_config
            config = app_config
        _store_instance = FeatureStore(config)
    
    return _store_instance
