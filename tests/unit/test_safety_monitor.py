"""
Unit tests for Safety Monitor
Tests fallback mechanisms, blocking logic, and audit logging.
"""

import pytest
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.safety.safety_monitor import SafetyMonitor, DiagnosisStatus, get_safety_monitor


class TestSafetyMonitor:
    """Tests for SafetyMonitor"""
    
    @pytest.fixture
    def monitor(self):
        return SafetyMonitor()
    
    def test_initialization(self, monitor):
        """Test monitor initializes correctly"""
        assert monitor.min_confidence == 0.55
        assert monitor.conflict_threshold == 0.5
        assert len(monitor.symptom_disease_rules) > 0
        assert len(monitor.alerts) == 0
    
    def test_rule_based_diagnosis_respiratory(self, monitor):
        """Test rule-based fallback for respiratory symptoms"""
        patient_data = {
            "symptoms": "fever cough shortness of breath"
        }
        
        result = monitor._rule_based_diagnose(patient_data)
        
        assert result["primary_diagnosis"]["disease"] == "Respiratory Infection"
        assert result["confidence"] > 0.5
    
    def test_rule_based_diagnosis_cardiac(self, monitor):
        """Test rule-based fallback for cardiac symptoms"""
        patient_data = {
            "symptoms": "chest pain shortness of breath sweating"
        }
        
        result = monitor._rule_based_diagnose(patient_data)
        
        assert "Cardiac" in result["primary_diagnosis"]["disease"]
        assert result["needs_escalation"] == True
    
    def test_rule_based_diagnosis_no_match(self, monitor):
        """Test fallback when no pattern matches"""
        patient_data = {
            "symptoms": "unusual symptom xyz"
        }
        
        result = monitor._rule_based_diagnose(patient_data)
        
        assert result["confidence"] == 0.0
        assert result["needs_escalation"] == True
        assert "NO_MATCH_FOUND" in result["safety_alerts"]
    
    def test_sanity_check_passes(self, monitor):
        """Test sanity check passes for valid diagnosis"""
        diagnosis = {
            "primary_diagnosis": {
                "disease": "Pneumonia"
            }
        }
        patient_data = {
            "symptoms": "cough fever shortness of breath"
        }
        
        assert monitor._sanity_check(diagnosis, patient_data) == True
    
    def test_sanity_check_fails_missing_symptoms(self, monitor):
        """Test sanity check fails when required symptoms missing"""
        diagnosis = {
            "primary_diagnosis": {
                "disease": "Myocardial Infarction"
            }
        }
        patient_data = {
            "symptoms": "headache fatigue"  # No chest pain
        }
        
        assert monitor._sanity_check(diagnosis, patient_data) == False
    
    def test_model_drift_detection_no_drift(self, monitor):
        """Test no drift detected when accuracy stable"""
        monitor.daily_metrics = {
            "baseline_accuracy": 0.92,
            "current_accuracy": 0.90
        }
        
        assert monitor._check_model_drift() == False
    
    def test_model_drift_detection_drift_detected(self, monitor):
        """Test drift detected when accuracy drops >5%"""
        monitor.daily_metrics = {
            "baseline_accuracy": 0.95,
            "current_accuracy": 0.85
        }
        
        assert monitor._check_model_drift() == True
    
    def test_alert_logging(self, monitor):
        """Test alert logging functionality"""
        monitor._log_alert("TEST_ALERT", "Test message")
        
        assert len(monitor.alerts) == 1
        assert monitor.alerts[0]["type"] == "TEST_ALERT"
        assert monitor.alerts[0]["message"] == "Test message"
    
    def test_audit_logging(self, monitor):
        """Test audit log creation"""
        monitor._audit_log(
            "REQ123",
            "SUCCESS",
            {"patient_id": "P001", "symptoms": "test"},
            {"diagnosis": {"disease": "Test"}, "confidence": 0.8}
        )
        
        assert len(monitor.audit_log) == 1
        assert monitor.audit_log[0]["request_id"] == "REQ123"
        assert monitor.audit_log[0]["action"] == "SUCCESS"
    
    def test_update_metrics(self, monitor):
        """Test metric updates for drift detection"""
        monitor.update_metrics(0.92)
        
        assert monitor.daily_metrics["baseline_accuracy"] == 0.92
        assert monitor.daily_metrics["current_accuracy"] == 0.92
        
        monitor.update_metrics(0.88)
        
        assert monitor.daily_metrics["baseline_accuracy"] == 0.92  # Unchanged
        assert monitor.daily_metrics["current_accuracy"] == 0.88
    
    def test_singleton_instance(self):
        """Test get_safety_monitor returns singleton"""
        mon1 = get_safety_monitor()
        mon2 = get_safety_monitor()
        
        assert mon1 is mon2


class TestAsyncDiagnoseWithSafety:
    """Tests for async diagnose_with_safety method"""
    
    @pytest.fixture
    def monitor(self):
        return SafetyMonitor()
    
    @pytest.mark.asyncio
    async def test_blocks_low_confidence(self, monitor):
        """Test that low confidence results in blocking"""
        # Mock a diagnosis agent that returns low confidence
        class MockAgent:
            async def run(self, data):
                return {
                    "primary_diagnosis": {"disease": "Unknown"},
                    "confidence": 0.3,  # Below 0.55 threshold
                    "safety_alerts": []
                }
        
        result = await monitor.diagnose_with_safety(
            {"symptoms": "test"},
            primary_agent=MockAgent()
        )
        
        assert result["status"] == DiagnosisStatus.BLOCKED
        assert "LOW_CONFIDENCE" in result["safety_alerts"]
        assert result["needs_escalation"] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
