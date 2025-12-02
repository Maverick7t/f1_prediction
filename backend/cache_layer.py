"""
Redis/Local Caching Layer for F1 Prediction API
Handles caching of expensive FastF1 queries with graceful fallback to local cache
"""

import json
import logging
import pickle
from datetime import datetime, timedelta
from typing import Any, Optional
from functools import wraps
import redis

from config import config

logger = logging.getLogger(__name__)


class CacheLayer:
    """
    Hybrid caching layer: Redis for production, local memory for development
    Caches expensive FastF1 queries to avoid timeouts on Render free tier
    """
    
    def __init__(self):
        self.redis_client = None
        self.local_cache = {}
        self.cache_timestamps = {}
        
        # Try to connect to Redis if configured
        if config.USE_REDIS:
            try:
                self.redis_client = redis.from_url(
                    config.REDIS_URL,
                    decode_responses=False,
                    socket_connect_timeout=5,
                    socket_keepalive=True
                )
                # Test connection
                self.redis_client.ping()
                logger.info("✓ Redis cache connected")
            except Exception as e:
                logger.warning(f"⚠️ Redis connection failed: {e}. Using local cache.")
                self.redis_client = None
        else:
            logger.info("ℹ️ Redis not configured. Using local memory cache.")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (Redis → Local)"""
        try:
            # Try Redis first
            if self.redis_client:
                try:
                    value = self.redis_client.get(key)
                    if value:
                        logger.debug(f"✓ Cache HIT (Redis): {key}")
                        return pickle.loads(value)
                except Exception as e:
                    logger.debug(f"Redis get failed: {e}")
            
            # Fallback to local cache
            if key in self.local_cache:
                logger.debug(f"✓ Cache HIT (Local): {key}")
                return self.local_cache[key]
            
            logger.debug(f"✗ Cache MISS: {key}")
            return None
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> bool:
        """Set value in cache with TTL"""
        try:
            if value is None:
                return False
            
            # Try Redis first
            if self.redis_client:
                try:
                    self.redis_client.setex(
                        key,
                        ttl_seconds,
                        pickle.dumps(value)
                    )
                    logger.debug(f"✓ Cached to Redis: {key} (TTL: {ttl_seconds}s)")
                    return True
                except Exception as e:
                    logger.debug(f"Redis set failed: {e}")
            
            # Fallback to local cache
            self.local_cache[key] = value
            self.cache_timestamps[key] = datetime.now() + timedelta(seconds=ttl_seconds)
            logger.debug(f"✓ Cached to Local: {key} (TTL: {ttl_seconds}s)")
            return True
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            if self.redis_client:
                self.redis_client.delete(key)
            
            if key in self.local_cache:
                del self.local_cache[key]
            
            if key in self.cache_timestamps:
                del self.cache_timestamps[key]
            
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    def clear_expired_local(self):
        """Remove expired items from local cache"""
        now = datetime.now()
        expired_keys = [
            k for k, exp_time in self.cache_timestamps.items()
            if exp_time < now
        ]
        for key in expired_keys:
            self.delete(key)
    
    def health_check(self) -> dict:
        """Health check for cache layer"""
        return {
            "redis_connected": bool(self.redis_client),
            "local_cache_items": len(self.local_cache),
            "cache_mode": "redis" if self.redis_client else "local"
        }


# Global cache instance
_cache_instance: Optional[CacheLayer] = None


def get_cache() -> CacheLayer:
    """Get or create global cache instance"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CacheLayer()
    return _cache_instance


def cached(key_prefix: str, ttl_seconds: int = 3600):
    """
    Decorator to cache function results
    Usage: @cached("race_history", ttl_seconds=1800)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache()
            
            # Build cache key from function name and arguments
            cache_key = f"{key_prefix}:{func.__name__}"
            if args:
                cache_key += f":{':'.join(str(a)[:20] for a in args)}"
            if kwargs:
                cache_key += f":{':'.join(f'{k}={v}' for k, v in sorted(kwargs.items()))}"
            
            # Try to get from cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            # Call function and cache result
            try:
                result = func(*args, **kwargs)
                if result is not None:
                    cache.set(cache_key, result, ttl_seconds)
                return result
            except Exception as e:
                logger.error(f"Error in cached function {func.__name__}: {e}")
                # Try to return stale cache as fallback
                return cache.get(cache_key)
        
        return wrapper
    return decorator


def cache_key_for(prefix: str, *args) -> str:
    """Build a cache key from prefix and arguments"""
    parts = [prefix]
    for arg in args:
        if arg is not None:
            parts.append(str(arg)[:30])
    return ":".join(parts)


# Cache key constants
CACHE_KEYS = {
    "RACE_HISTORY": "race_history",
    "QUALIFYING_TELEMETRY": "qualifying_telemetry",
    "LATEST_RACE_CIRCUIT": "latest_race_circuit",
    "DRIVER_STANDINGS": "driver_standings",
    "CONSTRUCTOR_STANDINGS": "constructor_standings",
    "NEXT_RACE": "next_race",
    "QUALIFYING_SESSION": "qualifying_session",
}

# Default TTL values (in seconds)
CACHE_TTL = {
    "RACE_HISTORY": 1800,              # 30 minutes
    "QUALIFYING_TELEMETRY": 1800,      # 30 minutes
    "LATEST_RACE_CIRCUIT": 3600,       # 1 hour
    "DRIVER_STANDINGS": 600,           # 10 minutes
    "CONSTRUCTOR_STANDINGS": 600,      # 10 minutes
    "NEXT_RACE": 86400,                # 24 hours
    "QUALIFYING_SESSION": 3600,        # 1 hour
}
