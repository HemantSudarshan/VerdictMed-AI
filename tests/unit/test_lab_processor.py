"""
Unit tests for Lab Processor
Tests lab value interpretation against clinical thresholds.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.nlp.lab_processor import LabProcessor, AbnormalityLevel, get_lab_processor


class TestLabProcessor:
    """Tests for LabProcessor"""
    
    @pytest.fixture
    def processor(self):
        return LabProcessor()
    
    def test_initialization(self, processor):
        """Test processor initializes with reference ranges"""
        assert processor.references is not None
        assert len(processor.references) > 20  # Should have many labs defined
        assert "wbc" in processor.references
        assert "troponin" in processor.references
    
    def test_process_empty_labs(self, processor):
        """Test processing with no lab results"""
        result = processor.process({})
        
        assert result["abnormalities"] == []
        assert result["severity_score"] == 0.0
        assert "No lab results" in result["summary"]
    
    def test_process_normal_labs(self, processor):
        """Test processing with all normal values"""
        labs = {
            "wbc": 7.0,      # Normal: 4.5-11.0
            "hemoglobin": 14.0,  # Normal: 12.0-17.0
            "sodium": 140.0  # Normal: 136-145
        }
        
        result = processor.process(labs)
        
        assert result["abnormalities"] == []
        assert result["severity_score"] == 0.0
        assert "normal" in result["summary"].lower()
    
    def test_process_elevated_wbc(self, processor):
        """Test detection of elevated WBC (infection indicator)"""
        labs = {"wbc": 15.0}  # High: above 11.0
        
        result = processor.process(labs)
        
        assert len(result["abnormalities"]) == 1
        assert result["abnormalities"][0]["level"] == "high"
        assert result["severity_score"] > 0
    
    def test_process_critical_troponin(self, processor):
        """Test detection of critical troponin (cardiac damage)"""
        labs = {"troponin": 0.5}  # Critical high: above 0.4
        
        result = processor.process(labs)
        
        assert len(result["abnormalities"]) == 1
        assert result["abnormalities"][0]["level"] == "critical_high"
        assert result["severity_score"] >= 0.8
        assert any("CRITICAL" in f for f in result["flags"])
    
    def test_process_sepsis_indicators(self, processor):
        """Test sepsis pattern detection (elevated WBC + lactate)"""
        labs = {
            "wbc": 18.0,    # High
            "lactate": 4.5  # Elevated (>2)
        }
        
        result = processor.process(labs)
        
        assert any("SEPSIS" in f for f in result["flags"])
    
    def test_process_cardiac_markers(self, processor):
        """Test cardiac damage pattern detection"""
        labs = {
            "troponin": 0.1,  # Elevated
            "bnp": 500        # Elevated
        }
        
        result = processor.process(labs)
        
        assert any("CARDIAC" in f for f in result["flags"])
    
    def test_process_renal_dysfunction(self, processor):
        """Test kidney function detection"""
        labs = {
            "creatinine": 2.5,  # Elevated
            "bun": 35           # Elevated
        }
        
        result = processor.process(labs)
        
        assert any("RENAL" in f for f in result["flags"])
    
    def test_process_multiple_abnormalities(self, processor):
        """Test processing multiple abnormal values"""
        labs = {
            "wbc": 15.0,       # High
            "crp": 50.0,       # High
            "glucose": 250.0   # High
        }
        
        result = processor.process(labs)
        
        assert len(result["abnormalities"]) == 3
        assert result["severity_score"] > 0
        assert len(result["recommendations"]) > 0
    
    def test_unknown_lab_ignored(self, processor):
        """Test that unknown labs are handled gracefully"""
        labs = {"unknown_lab_xyz": 100}
        
        result = processor.process(labs)
        
        assert result["abnormalities"] == []
    
    def test_singleton_instance(self):
        """Test get_lab_processor returns singleton"""
        proc1 = get_lab_processor()
        proc2 = get_lab_processor()
        
        assert proc1 is proc2


class TestAbnormalityLevels:
    """Tests for abnormality level classification"""
    
    @pytest.fixture
    def processor(self):
        return LabProcessor()
    
    def test_critical_low_potassium(self, processor):
        """Test critical low value detection"""
        labs = {"potassium": 2.0}  # Critical low: below 2.5
        
        result = processor.process(labs)
        
        assert result["abnormalities"][0]["level"] == "critical_low"
    
    def test_low_hemoglobin(self, processor):
        """Test low (non-critical) value detection"""
        labs = {"hemoglobin": 10.0}  # Low but not critical
        
        result = processor.process(labs)
        
        assert result["abnormalities"][0]["level"] == "low"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
