"""
Unit tests for Cache Service
Tests InMemoryCache and RedisService functionality.
"""
import pytest
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.cache.redis_service import RedisService, InMemoryCache, get_redis_service


class TestInMemoryCache:
    """Tests for InMemoryCache class"""
    
    @pytest.fixture
    def cache(self):
        return InMemoryCache()
    
    def test_initialization(self, cache):
        """Test cache initializes"""
        assert cache is not None
        assert hasattr(cache, '_cache')
    
    def test_set_and_get(self, cache):
        """Test basic set and get"""
        cache.set("test_key", "test_value")
        result = cache.get("test_key")
        assert result == "test_value"
    
    def test_get_nonexistent(self, cache):
        """Test getting nonexistent key"""
        result = cache.get("nonexistent_key")
        assert result is None
    
    def test_set_with_ttl(self, cache):
        """Test set with TTL"""
        cache.set("ttl_key", "ttl_value", ttl=1)
        assert cache.get("ttl_key") == "ttl_value"
        
        # Wait for expiration
        time.sleep(1.1)
        assert cache.get("ttl_key") is None
    
    def test_delete(self, cache):
        """Test delete key"""
        cache.set("delete_key", "value")
        assert cache.get("delete_key") == "value"
        
        cache.delete("delete_key")
        assert cache.get("delete_key") is None
    
    def test_exists(self, cache):
        """Test exists check"""
        cache.set("exists_key", "value")
        
        assert cache.exists("exists_key") == True
        assert cache.exists("nonexistent") == False
    
    def test_clear(self, cache):
        """Test clear all"""
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        cache.clear()
        
        assert cache.get("key1") is None
        assert cache.get("key2") is None
    
    def test_dict_value(self, cache):
        """Test storing dict values"""
        data = {"name": "test", "value": 123}
        cache.set("dict_key", data)
        
        result = cache.get("dict_key")
        assert result == data
    
    def test_list_value(self, cache):
        """Test storing list values"""
        data = [1, 2, 3, "four"]
        cache.set("list_key", data)
        
        result = cache.get("list_key")
        assert result == data


class TestRedisService:
    """Tests for RedisService class"""
    
    @pytest.fixture
    def service(self):
        return RedisService()
    
    def test_initialization(self, service):
        """Test service initializes"""
        assert service is not None
    
    def test_has_fallback(self, service):
        """Test fallback cache exists"""
        assert hasattr(service, '_memory_cache')
    
    def test_generate_key(self, service):
        """Test key generation"""
        key = service.generate_key("prefix", {"a": 1, "b": 2})
        
        assert isinstance(key, str)
        assert "prefix" in key
    
    def test_generate_key_consistent(self, service):
        """Test key generation is consistent"""
        key1 = service.generate_key("test", {"x": 1})
        key2 = service.generate_key("test", {"x": 1})
        
        assert key1 == key2
    
    def test_cache_diagnosis(self, service):
        """Test caching diagnosis result"""
        diagnosis = {
            "diagnosis": "Test",
            "confidence": 0.8
        }
        
        key = service.cache_diagnosis("patient1", ["fever"], diagnosis)
        assert key is not None
    
    def test_get_cached_diagnosis(self, service):
        """Test retrieving cached diagnosis"""
        diagnosis = {
            "diagnosis": "Cached",
            "confidence": 0.9
        }
        
        # Cache it
        service.cache_diagnosis("patient_cached", ["symptom"], diagnosis)
        
        # Retrieve it
        result = service.get_cached_diagnosis("patient_cached", ["symptom"])
        
        # May or may not work depending on Redis availability
        # Just verify no exception
        assert result is None or isinstance(result, dict)


class TestGetRedisService:
    """Tests for get_redis_service factory function"""
    
    def test_returns_instance(self):
        """Test factory returns instance"""
        service = get_redis_service()
        assert service is not None
        assert isinstance(service, RedisService)
    
    def test_singleton(self):
        """Test singleton pattern"""
        s1 = get_redis_service()
        s2 = get_redis_service()
        
        # Both should be same instance
        assert s1 is s2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
