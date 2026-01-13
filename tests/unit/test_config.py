"""
Unit tests for Configuration Module
Tests settings, environment loading, and validation.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import Settings, get_settings


class TestSettings:
    """Tests for Settings class"""
    
    def test_settings_loads(self):
        """Test settings can be loaded"""
        settings = Settings()
        assert settings is not None
    
    def test_has_app_name(self):
        """Test app name is set"""
        settings = Settings()
        assert hasattr(settings, 'app_name')
        assert len(settings.app_name) > 0
    
    def test_has_confidence_threshold(self):
        """Test confidence threshold exists"""
        settings = Settings()
        assert hasattr(settings, 'confidence_threshold')
        assert 0 <= settings.confidence_threshold <= 1
    
    def test_has_api_settings(self):
        """Test API settings exist"""
        settings = Settings()
        
        assert hasattr(settings, 'api_host') or hasattr(settings, 'host')
        assert hasattr(settings, 'api_port') or hasattr(settings, 'port')


class TestGetSettings:
    """Tests for get_settings function"""
    
    def test_returns_settings(self):
        """Test returns Settings instance"""
        settings = get_settings()
        assert settings is not None
        assert isinstance(settings, Settings)
    
    def test_singleton_pattern(self):
        """Test settings uses singleton/caching"""
        s1 = get_settings()
        s2 = get_settings()
        
        # Should be same object (cached)
        assert s1 is s2


class TestEnvironmentVariables:
    """Tests for environment variable handling"""
    
    def test_default_values(self):
        """Test default values are set"""
        settings = Settings()
        
        # Should have reasonable defaults
        assert settings.confidence_threshold > 0
    
    def test_settings_not_none(self):
        """Test critical settings are not None"""
        settings = Settings()
        
        # Core settings should exist
        assert settings.app_name is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
