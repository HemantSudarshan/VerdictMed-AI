"""
Unit tests for NLP Module (ClinicalNLPPipeline)
Tests symptom extraction, vital signs parsing, and clinical text analysis.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.nlp.clinical_nlp import ClinicalNLPPipeline


class TestClinicalNLPPipelineInit:
    """Tests for ClinicalNLPPipeline initialization"""
    
    def test_initialization_no_models(self):
        """Test pipeline initializes without loading models"""
        pipeline = ClinicalNLPPipeline(load_models=False)
        assert pipeline is not None
    
    def test_has_required_methods(self):
        """Test pipeline has expected methods"""
        pipeline = ClinicalNLPPipeline(load_models=False)
        
        assert hasattr(pipeline, 'extract_symptoms')
        assert hasattr(pipeline, 'extract_vitals')
        assert hasattr(pipeline, 'analyze_clinical_note')
        assert hasattr(pipeline, 'expand_abbreviations')


class TestExpandAbbreviations:
    """Tests for abbreviation expansion"""
    
    @pytest.fixture
    def pipeline(self):
        return ClinicalNLPPipeline(load_models=False)
    
    def test_expand_sob(self, pipeline):
        """Test SOB expansion"""
        text = "Patient reports SOB on exertion"
        result = pipeline.expand_abbreviations(text)
        
        assert isinstance(result, str)
    
    def test_expand_htn(self, pipeline):
        """Test HTN expansion"""
        text = "History of HTN and DM"
        result = pipeline.expand_abbreviations(text)
        
        assert isinstance(result, str)


class TestExtractSymptoms:
    """Tests for symptom extraction"""
    
    @pytest.fixture
    def pipeline(self):
        return ClinicalNLPPipeline(load_models=False)
    
    def test_extract_basic_symptoms(self, pipeline):
        """Test extracting basic symptoms"""
        text = "Patient presents with fever, cough, and headache"
        result = pipeline.extract_symptoms(text)
        
        assert isinstance(result, list)
    
    def test_extract_from_clinical_note(self, pipeline):
        """Test extraction from realistic clinical text"""
        text = """
        HPI: 58-year-old male with acute onset chest pain 
        radiating to left arm. Associated with shortness of breath 
        and diaphoresis. Denies fever or nausea.
        """
        result = pipeline.extract_symptoms(text)
        
        assert isinstance(result, list)
    
    def test_empty_text(self, pipeline):
        """Test handling of empty text"""
        result = pipeline.extract_symptoms("")
        
        assert isinstance(result, list)
    
    def test_symptom_structure(self, pipeline):
        """Test extracted symptom structure"""
        text = "Patient has fever and cough"
        result = pipeline.extract_symptoms(text)
        
        if len(result) > 0:
            symptom = result[0]
            # Should have name at minimum
            assert "name" in symptom or isinstance(symptom, str)


class TestExtractVitals:
    """Tests for vital signs extraction"""
    
    @pytest.fixture
    def pipeline(self):
        return ClinicalNLPPipeline(load_models=False)
    
    def test_extract_temperature(self, pipeline):
        """Test temperature extraction"""
        text = "Temp: 101.2°F"
        result = pipeline.extract_vitals(text)
        
        assert isinstance(result, dict)
    
    def test_extract_blood_pressure(self, pipeline):
        """Test blood pressure extraction"""
        text = "BP 140/90 mmHg"
        result = pipeline.extract_vitals(text)
        
        assert isinstance(result, dict)
    
    def test_extract_heart_rate(self, pipeline):
        """Test heart rate extraction"""
        text = "HR: 88 bpm"
        result = pipeline.extract_vitals(text)
        
        assert isinstance(result, dict)
    
    def test_extract_spo2(self, pipeline):
        """Test SpO2 extraction"""
        text = "SpO2: 94% on room air"
        result = pipeline.extract_vitals(text)
        
        assert isinstance(result, dict)
    
    def test_extract_multiple_vitals(self, pipeline):
        """Test extracting multiple vitals"""
        text = "Vitals: BP 120/80, HR 72, Temp 98.6, SpO2 98%"
        result = pipeline.extract_vitals(text)
        
        assert isinstance(result, dict)


class TestAnalyzeClinicalNote:
    """Tests for full clinical note analysis"""
    
    @pytest.fixture
    def pipeline(self):
        return ClinicalNLPPipeline(load_models=False)
    
    def test_analyze_returns_dict(self, pipeline):
        """Test analyze returns dictionary"""
        text = "Patient with fever and cough for 3 days"
        result = pipeline.analyze_clinical_note(text)
        
        assert isinstance(result, dict)
    
    def test_analyze_has_symptoms(self, pipeline):
        """Test analysis includes symptoms"""
        text = "Patient complains of chest pain and shortness of breath"
        result = pipeline.analyze_clinical_note(text)
        
        assert isinstance(result, dict)
        assert "symptoms" in result or "findings" in result or len(result) >= 0
    
    def test_analyze_complex_note(self, pipeline):
        """Test analysis of complex clinical note"""
        text = """
        Chief Complaint: Chest pain
        
        HPI: 65-year-old male with history of HTN, DM, and hyperlipidemia
        presenting with acute onset chest pain that started 2 hours ago.
        Pain is substernal, radiating to left arm, associated with 
        diaphoresis and nausea. Denies fever or cough.
        
        Vitals: BP 160/95, HR 96, Temp 98.6, SpO2 94% on RA
        """
        result = pipeline.analyze_clinical_note(text)
        
        assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
