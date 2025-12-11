"""
Simple file-based caching for expensive FastF1 queries
Stores JSON data in /backend/f1_cache/ directory
Persists across Render restarts, no external dependencies needed
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = Path("./f1_cache")
CACHE_DIR.mkdir(exist_ok=True, parents=True)


class FileCache:
    """Simple file-based cache using JSON"""
    
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"✓ File cache initialized at: {self.cache_dir}")
    
    def _get_cache_file(self, key: str) -> Path:
        """Get cache file path for a key"""
        # Sanitize key to valid filename
        safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
        return self.cache_dir / f"{safe_key}.json"
    
    def get(self, key: str, ttl_seconds: int = 1800) -> Optional[Any]:
        """
        Get cached data if it exists and is fresh
        Args:
            key: Cache key
            ttl_seconds: Time-to-live in seconds (default 30 mins)
        Returns:
            Cached data or None if not found/expired
        """
        try:
            cache_file = self._get_cache_file(key)
            
            if not cache_file.exists():
                logger.debug(f"✗ Cache MISS: {key} (file not found)")
                return None
            
            # Check if cache is still fresh
            file_mtime = cache_file.stat().st_mtime
            file_age_seconds = (datetime.now().timestamp() - file_mtime)
            
            if file_age_seconds > ttl_seconds:
                logger.debug(f"✗ Cache EXPIRED: {key} (age: {file_age_seconds:.0f}s > {ttl_seconds}s)")
                return None
            
            # Load and return cached data
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            logger.debug(f"✓ Cache HIT: {key} (age: {file_age_seconds:.0f}s)")
            return data
            
        except Exception as e:
            logger.warning(f"Cache get error for {key}: {e}")
            return None
    
    def set(self, key: str, data: Any) -> bool:
        """
        Save data to cache file
        Args:
            key: Cache key
            data: Data to cache (must be JSON serializable)
        Returns:
            True if successful
        """
        try:
            cache_file = self._get_cache_file(key)
            
            # Write to temp file first, then move (atomic write)
            temp_file = cache_file.with_suffix('.tmp')
            
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Atomic rename
            temp_file.replace(cache_file)
            
            file_size_kb = cache_file.stat().st_size / 1024
            logger.debug(f"✓ Cached to file: {key} ({file_size_kb:.1f} KB)")
            return True
            
        except Exception as e:
            logger.error(f"Cache set error for {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete cache file"""
        try:
            cache_file = self._get_cache_file(key)
            if cache_file.exists():
                cache_file.unlink()
                logger.debug(f"✓ Deleted cache: {key}")
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    def clear_expired(self) -> int:
        """Remove all expired cache files"""
        removed = 0
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                file_age = datetime.now().timestamp() - cache_file.stat().st_mtime
                # Delete if older than 24 hours
                if file_age > 86400:
                    cache_file.unlink()
                    removed += 1
            
            if removed > 0:
                logger.info(f"✓ Cleaned {removed} expired cache files")
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")
        
        return removed
    
    def health_check(self) -> dict:
        """Get cache health status"""
        try:
            cache_files = list(self.cache_dir.glob("*.json"))
            total_size_kb = sum(f.stat().st_size for f in cache_files) / 1024
            
            return {
                "status": "healthy",
                "cache_dir": str(self.cache_dir),
                "cached_items": len(cache_files),
                "total_size_kb": round(total_size_kb, 1)
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


# Global cache instance
_cache_instance: Optional[FileCache] = None


def get_file_cache() -> FileCache:
    """Get or create global file cache instance"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = FileCache()
    return _cache_instance


# Cache key constants
CACHE_KEYS = {
    "RACE_HISTORY": "race_history",
    "QUALIFYING_TELEMETRY": "qualifying_telemetry_latest",
    "LATEST_RACE_CIRCUIT": "latest_race_circuit",
}

# Default TTL values (in seconds)
CACHE_TTL = {
    "RACE_HISTORY": 1800,              # 30 minutes
    "QUALIFYING_TELEMETRY": 1800,      # 30 minutes
    "LATEST_RACE_CIRCUIT": 3600,       # 1 hour
}
