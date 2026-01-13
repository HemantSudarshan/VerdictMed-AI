"""
Tests for FastAPI endpoints
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


@pytest.fixture
def client():
    """Create test client"""
    from src.api.main import app
    return TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_returns_200(self, client):
        """Health endpoint should return 200"""
        response = client.get("/health")
        assert response.status_code == 200
        
    def test_health_returns_status(self, client):
        """Health endpoint should return status"""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestRootEndpoint:
    """Test root endpoint"""
    
    def test_root_returns_api_info(self, client):
        """Root endpoint should return API info"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "VerdictMed" in data["name"]


class TestAuthProtection:
    """Test API authentication"""
    
    def test_diagnose_without_key_returns_401(self, client):
        """Diagnosis endpoint should reject requests without API key"""
        response = client.post(
            "/api/v1/diagnose",
            json={"symptoms": "fever and cough"}
        )
        assert response.status_code == 401
    
    def test_diagnose_with_invalid_key_returns_401(self, client):
        """Diagnosis endpoint should reject invalid API key"""
        response = client.post(
            "/api/v1/diagnose",
            json={"symptoms": "fever and cough"},
            headers={"X-API-Key": "invalid-key"}
        )
        assert response.status_code == 401
    
    def test_diagnose_with_valid_key_accepted(self, client):
        """Diagnosis endpoint should accept valid API key"""
        # Using the dev test key from auth.py
        response = client.post(
            "/api/v1/diagnose",
            json={"symptoms": "fever and cough"},
            headers={"X-API-Key": "dev-test-key"}
        )
        # Should not be 401 (might be 500 if agent fails, but auth passed)
        assert response.status_code != 401


class TestSymptomsEndpoint:
    """Test symptoms listing endpoint"""
    
    def test_symptoms_returns_list(self, client):
        """Symptoms endpoint should return list of symptoms"""
        response = client.get("/api/v1/symptoms")
        assert response.status_code == 200
        data = response.json()
        assert "symptoms" in data
        assert isinstance(data["symptoms"], list)
        assert len(data["symptoms"]) > 0


class TestDisclaimerEndpoint:
    """Test disclaimer endpoint"""
    
    def test_disclaimer_returns_text(self, client):
        """Disclaimer endpoint should return disclaimer text"""
        response = client.get("/api/v1/disclaimer")
        assert response.status_code == 200
        data = response.json()
        assert "disclaimer" in data
        assert "NOT a replacement" in data["disclaimer"]
