"""
In-Memory Prediction Cache

Optimizes prediction system by caching all data in memory instead of querying Supabase repeatedly.
This solves the architectural issue where every API request made 3+ HTTP calls to Supabase.

Architecture:
1. Qualifying cache (24h TTL): Cached from FastF1 API, reused across requests
2. Static data (forever TTL): Drivers, teams, circuits loaded once at startup
3. Prediction queue (batch): Queue predictions, sync to Supabase every 5 minutes or when >50 items
4. LRU eviction (max 1000): Keep memory footprint bounded

Benefits:
- Eliminate Supabase queries for qualifying data (3 HTTP -> 0 per request)
- Batch predictions to Supabase (reduce write operations by 50x)
- Instant lookups (memory vs network latency)
- Thread-safe for concurrent requests
"""

from collections import OrderedDict
from datetime import datetime, timedelta
from threading import Lock
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger("prediction_cache")


class PredictionCache:
    """In-memory cache for predictions and qualifying data with smart batching."""
    
    def __init__(self, max_size: int = 1000, sync_interval: int = 300, queue_threshold: int = 50):
        """
        Args:
            max_size: Maximum predictions to store (LRU eviction beyond this)
            sync_interval: Seconds between automatic syncs to Supabase (default 5 min)
            queue_threshold: Sync when this many predictions queued (or interval expires)
        """
        # Prediction cache: race_key -> prediction_data
        self.predictions = OrderedDict()
        
        # Qualifying cache: race_key -> {"data": qual_data, "timestamp": datetime}
        self.qualifying = {}
        self.qualifying_ttl = {}  # race_key -> expiry_datetime
        
        # Static reference data (loaded once at startup)
        self.static_drivers = []
        self.static_teams = []
        self.static_circuits = []
        
        # Queue for batch Supabase sync
        self.pending_syncs = []
        self.last_sync_time = datetime.now()
        
        # Configuration
        self.max_size = max_size
        self.sync_interval = sync_interval
        self.queue_threshold = queue_threshold
        
        # Thread safety
        self.lock = Lock()
        
        logger.info("✓ PredictionCache initialized")
    
    # ==================== PREDICTIONS ====================
    
    def add_prediction(self, race_key: str, prediction_data: Dict[str, Any]) -> None:
        """Add prediction to cache and queue for batch sync."""
        with self.lock:
            # Store in memory cache
            self.predictions[race_key] = {
                "data": prediction_data,
                "timestamp": datetime.now()
            }
            
            # Move to end (LRU ordering)
            self.predictions.move_to_end(race_key)
            
            # Evict oldest if over max size
            if len(self.predictions) > self.max_size:
                self.predictions.popitem(last=False)
            
            # Queue for Supabase sync
            self.pending_syncs.append({
                "race_key": race_key,
                "data": prediction_data,
                "timestamp": datetime.now()
            })
            
            logger.debug(f"  Added prediction: {race_key} | Queue size: {len(self.pending_syncs)}")
    
    def get_prediction(self, race_key: str) -> Optional[Dict[str, Any]]:
        """Get prediction from memory cache (instant, no HTTP)."""
        with self.lock:
            if race_key in self.predictions:
                pred = self.predictions[race_key]
                # Move to end (LRU)
                self.predictions.move_to_end(race_key)
                return pred.get("data")
        return None
    
    def get_all_predictions(self) -> List[Dict[str, Any]]:
        """Get all cached predictions."""
        with self.lock:
            return [p["data"] for p in self.predictions.values()]
    
    # ==================== QUALIFYING ====================
    
    def add_qualifying(self, race_key: str, qual_data: Any, ttl_hours: int = 24) -> None:
        """Add qualifying data with TTL (default 24 hours)."""
        with self.lock:
            self.qualifying[race_key] = {
                "data": qual_data,
                "timestamp": datetime.now()
            }
            self.qualifying_ttl[race_key] = datetime.now() + timedelta(hours=ttl_hours)
            logger.debug(f"  Cached qualifying: {race_key} (TTL: {ttl_hours}h)")
    
    def get_qualifying(self, race_key: str) -> Optional[Any]:
        """Get qualifying data from memory (instant, no HTTP to Supabase!)."""
        with self.lock:
            if race_key not in self.qualifying:
                return None
            
            # Check if expired
            if datetime.now() > self.qualifying_ttl.get(race_key, datetime.now()):
                # Expired, remove and return None
                del self.qualifying[race_key]
                if race_key in self.qualifying_ttl:
                    del self.qualifying_ttl[race_key]
                logger.debug(f"  Qualifying expired: {race_key}")
                return None
            
            # Still valid
            return self.qualifying[race_key]["data"]
    
    def get_latest_qualifying(self) -> Optional[Any]:
        """Get most recent qualifying data (by timestamp)."""
        with self.lock:
            if not self.qualifying:
                return None
            
            # Find most recent non-expired entry
            latest_key = None
            latest_time = None
            
            for key, entry in self.qualifying.items():
                # Skip if expired
                if datetime.now() > self.qualifying_ttl.get(key, datetime.now()):
                    continue
                
                if latest_time is None or entry["timestamp"] > latest_time:
                    latest_key = key
                    latest_time = entry["timestamp"]
            
            if latest_key:
                return self.qualifying[latest_key]["data"]
        
        return None
    
    # ==================== STATIC DATA ====================
    
    def load_static_data(self, drivers: List[str], teams: List[str], circuits: List[str]) -> None:
        """Load static reference data once at startup (never changes during season)."""
        with self.lock:
            self.static_drivers = drivers
            self.static_teams = teams
            self.static_circuits = circuits
            logger.info(f"✓ Static data loaded: {len(drivers)} drivers, {len(teams)} teams, {len(circuits)} circuits")
    
    def get_drivers(self) -> List[str]:
        """Get all drivers (no HTTP call)."""
        with self.lock:
            return self.static_drivers.copy()
    
    def get_teams(self) -> List[str]:
        """Get all teams (no HTTP call)."""
        with self.lock:
            return self.static_teams.copy()
    
    def get_circuits(self) -> List[str]:
        """Get all circuits (no HTTP call)."""
        with self.lock:
            return self.static_circuits.copy()
    
    # ==================== BATCH SYNC ====================
    
    def should_sync_to_supabase(self) -> bool:
        """Check if it's time to sync queued predictions to Supabase."""
        with self.lock:
            # Sync if:
            # 1. Queue has >50 predictions, OR
            # 2. Interval expired (every 5 minutes)
            queue_large = len(self.pending_syncs) >= self.queue_threshold
            time_expired = (datetime.now() - self.last_sync_time).total_seconds() >= self.sync_interval
            
            return queue_large or time_expired
    
    def get_pending_syncs(self) -> List[Dict[str, Any]]:
        """Get pending predictions for batch Supabase write."""
        with self.lock:
            syncs = self.pending_syncs.copy()
            self.pending_syncs.clear()
            self.last_sync_time = datetime.now()
            
            if syncs:
                logger.info(f"  ✓ Batching {len(syncs)} predictions to Supabase")
            
            return syncs
    
    # ==================== STATS & HEALTH ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                "predictions_cached": len(self.predictions),
                "qualifying_cached": len(self.qualifying),
                "pending_syncs": len(self.pending_syncs),
                "static_drivers": len(self.static_drivers),
                "static_teams": len(self.static_teams),
                "static_circuits": len(self.static_circuits),
                "max_size": self.max_size,
                "should_sync": self.should_sync_to_supabase()
            }
    
    def clear(self) -> None:
        """Clear all caches (useful for testing or manual reset)."""
        with self.lock:
            self.predictions.clear()
            self.qualifying.clear()
            self.qualifying_ttl.clear()
            self.pending_syncs.clear()
            logger.info("  Caches cleared")


# Global singleton instance
_cache_instance: Optional[PredictionCache] = None


def get_prediction_cache() -> PredictionCache:
    """Get or create the global prediction cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = PredictionCache()
    return _cache_instance


def reset_cache_for_testing() -> None:
    """Reset cache for testing purposes."""
    global _cache_instance
    _cache_instance = None
