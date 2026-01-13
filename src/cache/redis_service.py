"""
Redis Caching Service with In-Memory Fallback
Handle caching of diagnostic results to improve performance.
Works without Redis by falling back to in-memory cache.
"""

import json
import hashlib
import time
from typing import Optional, Any, Dict
from loguru import logger

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis package not installed, using in-memory cache")

from src.config import get_settings


class InMemoryCache:
    """
    In-memory cache fallback when Redis is unavailable.
    Supports TTL (time-to-live) for entries.
    """
    
    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, tuple] = {}  # key -> (value, expiry_time)
        self._max_size = max_size
        logger.info("InMemoryCache initialized (max_size={})".format(max_size))
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        if key not in self._cache:
            return None
        
        value, expiry = self._cache[key]
        
        if expiry and time.time() > expiry:
            # Expired
            del self._cache[key]
            return None
        
        return value
    
    def set(self, key: str, value: Any, ttl_seconds: int = 3600):
        """Set value with optional TTL"""
        # Evict oldest entries if cache is full
        if len(self._cache) >= self._max_size:
            # Remove 10% oldest entries
            keys_to_remove = list(self._cache.keys())[:self._max_size // 10]
            for k in keys_to_remove:
                del self._cache[k]
        
        expiry = time.time() + ttl_seconds if ttl_seconds else None
        self._cache[key] = (value, expiry)
    
    def delete(self, key: str):
        """Delete a key"""
        self._cache.pop(key, None)
    
    def clear(self):
        """Clear all entries"""
        self._cache.clear()
    
    def ping(self) -> bool:
        """Always returns True for in-memory"""
        return True


class RedisService:
    """Service for caching data in Redis with in-memory fallback"""
    
    def __init__(self):
        settings = get_settings()
        self.enabled = settings.enable_caching
        self.client = None
        self.using_fallback = False
        
        if self.enabled and REDIS_AVAILABLE:
            try:
                self.client = redis.from_url(settings.redis_url, decode_responses=True)
                self.client.ping()
                logger.info("Connected to Redis cache")
            except Exception as e:
                logger.warning(f"Redis unavailable: {e}. Using in-memory fallback.")
                self.client = InMemoryCache()
                self.using_fallback = True
        else:
            # Use in-memory fallback
            self.client = InMemoryCache()
            self.using_fallback = True
            logger.info("Using in-memory cache (Redis not configured)")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.enabled or not self.client:
            return None
            
        try:
            if self.using_fallback:
                return self.client.get(key)
            else:
                value = self.client.get(key)
                if value:
                    return json.loads(value)
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
        return None
        
    def set(self, key: str, value: Any, ttl_seconds: int = 3600):
        """Set value in cache with TTL"""
        if not self.enabled or not self.client:
            return
            
        try:
            if self.using_fallback:
                self.client.set(key, value, ttl_seconds)
            else:
                json_val = json.dumps(value)
                self.client.setex(key, ttl_seconds, json_val)
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")

    @staticmethod
    def generate_cache_key(prefix: str, data: Dict) -> str:
        """Generate deterministic cache key from dict data"""
        # Sort keys for consistency
        data_str = json.dumps(data, sort_keys=True)
        hash_val = hashlib.sha256(data_str.encode()).hexdigest()[:16]
        return f"{prefix}:{hash_val}"
    
    @property
    def is_connected(self) -> bool:
        """Check if cache is available"""
        return self.client is not None


# Singleton instance
_redis_service = None

def get_redis_service() -> RedisService:
    """Get singleton redis service"""
    global _redis_service
    if _redis_service is None:
        _redis_service = RedisService()
    return _redis_service


# Alias for easier imports
get_cache = get_redis_service


if __name__ == "__main__":
    # Test the cache
    cache = get_redis_service()
    print(f"Using fallback: {cache.using_fallback}")
    
    # Test set/get
    cache.set("test_key", {"value": 123, "data": "test"})
    result = cache.get("test_key")
    print(f"Cache test: {result}")

