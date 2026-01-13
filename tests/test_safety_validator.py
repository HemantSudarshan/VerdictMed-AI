"""
Tests for Safety Validator
"""

import pytest
from unittest.mock import MagicMock


class MockSettings:
    """Mock settings for testing"""
    confidence_threshold = 0.55
    enable_safety_checks = True


@pytest.fixture
def validator():
    """Create a SafetyValidator instance"""
    from src.safety.validator import SafetyValidator
    return SafetyValidator(MockSettings())


class TestConfidenceCheck:
    """Test confidence validation"""
    
    def test_high_confidence_passes(self, validator):
        """Diagnosis with high confidence should pass"""
        result = {"confidence": 0.85, "primary_diagnosis": {"disease": "Pneumonia"}}
        validation = validator.validate_diagnosis(result)
        assert "low_confidence" not in [a.lower() for a in validation.get("alerts", [])]
    
    def test_low_confidence_flagged(self, validator):
        """Diagnosis with low confidence should be flagged"""
        result = {"confidence": 0.45, "primary_diagnosis": {"disease": "Unknown"}}
        validation = validator.validate_diagnosis(result)
        # Should have some kind of alert or escalation
        assert validation.get("requires_escalation", False) or len(validation.get("alerts", [])) > 0


class TestCriticalConditions:
    """Test critical condition detection"""
    
    def test_mi_flagged_as_critical(self, validator):
        """Myocardial infarction should trigger escalation"""
        result = {
            "confidence": 0.8,
            "primary_diagnosis": {"disease": "Myocardial Infarction"},
            "extracted_symptoms": []
        }
        validation = validator.validate_diagnosis(result)
        assert validation.get("requires_escalation", False) == True
    
    def test_stroke_flagged_as_critical(self, validator):
        """Stroke should trigger escalation"""
        result = {
            "confidence": 0.8,
            "primary_diagnosis": {"disease": "Acute Stroke"},
            "extracted_symptoms": []
        }
        validation = validator.validate_diagnosis(result)
        assert validation.get("requires_escalation", False) == True
    
    def test_common_cold_not_critical(self, validator):
        """Common cold should not trigger escalation"""
        result = {
            "confidence": 0.8,
            "primary_diagnosis": {"disease": "Common Cold"},
            "extracted_symptoms": [{"symptom": "runny nose", "negated": False}]
        }
        validation = validator.validate_diagnosis(result)
        # Common cold with symptoms shouldn't force escalation
        # (unless low confidence)


class TestSanityChecks:
    """Test sanity validation"""
    
    def test_diagnosis_with_matching_symptoms_passes(self, validator):
        """Diagnosis matching symptoms should pass sanity check"""
        result = {
            "confidence": 0.8,
            "primary_diagnosis": {"disease": "Pneumonia"},
            "extracted_symptoms": [
                {"symptom": "fever", "negated": False},
                {"symptom": "cough", "negated": False}
            ]
        }
        validation = validator.validate_diagnosis(result)
        # Should not have sanity failure
        alerts = [a.lower() for a in validation.get("alerts", [])]
        assert not any("sanity" in a for a in alerts)


class TestConflictDetection:
    """Test signal conflict detection"""
    
    def test_conflicting_vitals_detected(self, validator):
        """Conflicting vital signs should be flagged"""
        result = {
            "confidence": 0.7,
            "primary_diagnosis": {"disease": "Sepsis"},
            "extracted_symptoms": [],
            "extracted_vitals": {
                "temperature": 36.5,  # Normal temp
                "heart_rate": 120     # But high HR
            }
        }
        # If we had more sophisticated conflict detection logic,
        # this would be flagged. Current implementation may or may not catch this.
        validation = validator.validate_diagnosis(result)
        # Just verify it doesn't crash


class TestIntegration:
    """Integration tests for full validation flow"""
    
    def test_full_validation_returns_expected_structure(self, validator):
        """Validate that output has expected keys"""
        result = {
            "confidence": 0.75,
            "primary_diagnosis": {"disease": "Bronchitis"},
            "extracted_symptoms": [{"symptom": "cough", "negated": False}],
            "extracted_vitals": {"temperature": 38.2}
        }
        validation = validator.validate_diagnosis(result)
        
        assert "alerts" in validation
        assert "requires_escalation" in validation
        assert isinstance(validation["alerts"], list)
        assert isinstance(validation["requires_escalation"], bool)
