"""
Unit tests for Knowledge Graph Module
Tests MockKnowledgeGraph functionality and disease matching.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.knowledge_graph.mock_kg import MockKnowledgeGraph, get_knowledge_graph


class TestMockKnowledgeGraphInit:
    """Tests for MockKnowledgeGraph initialization"""
    
    def test_initialization(self):
        """Test KG initializes with disease data"""
        kg = MockKnowledgeGraph()
        assert kg is not None
        assert len(kg.DISEASES) > 0
    
    def test_has_common_diseases(self):
        """Test common diseases are present"""
        kg = MockKnowledgeGraph()
        
        disease_names = [name.lower() for name in kg.DISEASES.keys()]
        
        assert any("pneumonia" in name for name in disease_names)
        assert any("diabetes" in name for name in disease_names)
    
    def test_disease_structure(self):
        """Test each disease has required fields"""
        kg = MockKnowledgeGraph()
        
        for name, disease in kg.DISEASES.items():
            assert "symptoms" in disease
            assert "icd10" in disease
            assert "severity" in disease
            assert isinstance(disease["symptoms"], list)


class TestFindDiseasesBySymptoms:
    """Tests for find_diseases_by_symptoms method"""
    
    @pytest.fixture
    def kg(self):
        return MockKnowledgeGraph()
    
    def test_single_symptom_match(self, kg):
        """Test matching with single symptom"""
        results = kg.find_diseases_by_symptoms(["fever"])
        
        assert isinstance(results, list)
        assert len(results) > 0
    
    def test_multiple_symptom_match(self, kg):
        """Test matching with multiple symptoms"""
        results = kg.find_diseases_by_symptoms(["fever", "cough", "shortness of breath"])
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        # Results should be sorted by match_ratio (descending)
        if len(results) > 1:
            assert results[0]["match_ratio"] >= results[1]["match_ratio"]
    
    def test_result_structure(self, kg):
        """Test result dictionary structure"""
        results = kg.find_diseases_by_symptoms(["chest pain"])
        
        if len(results) > 0:
            result = results[0]
            assert "disease" in result
            assert "match_ratio" in result
            assert "matched_symptoms" in result
            assert "severity" in result
            assert "icd10" in result
    
    def test_empty_symptoms(self, kg):
        """Test with empty symptom list"""
        results = kg.find_diseases_by_symptoms([])
        
        assert isinstance(results, list)
    
    def test_case_insensitivity(self, kg):
        """Test case-insensitive matching"""
        results_lower = kg.find_diseases_by_symptoms(["fever"])
        results_upper = kg.find_diseases_by_symptoms(["FEVER"])
        
        # Both should return results
        assert len(results_lower) > 0
        assert len(results_upper) > 0


class TestGetDifferentialDiagnosis:
    """Tests for get_differential_diagnosis method"""
    
    @pytest.fixture
    def kg(self):
        return MockKnowledgeGraph()
    
    def test_returns_list(self, kg):
        """Test returns list of diagnoses"""
        results = kg.get_differential_diagnosis(["chest pain", "shortness of breath"])
        
        assert isinstance(results, list)
    
    def test_limited_results(self, kg):
        """Test results are limited"""
        results = kg.get_differential_diagnosis(["fever", "cough"])
        
        assert len(results) <= 5


class TestCheckCriticalConditions:
    """Tests for check_critical_conditions method"""
    
    @pytest.fixture
    def kg(self):
        return MockKnowledgeGraph()
    
    def test_returns_list(self, kg):
        """Test returns list of critical conditions"""
        results = kg.check_critical_conditions(["chest pain", "shortness of breath"])
        
        assert isinstance(results, list)
    
    def test_critical_symptoms(self, kg):
        """Test critical symptoms trigger warnings"""
        # Using typical MI symptoms
        results = kg.check_critical_conditions([
            "chest pain", 
            "arm pain",
            "sweating"
        ])
        
        # Should identify potential critical conditions
        assert isinstance(results, list)
        # MI should be detected
        if len(results) > 0:
            severities = [r["severity"] for r in results]
            assert "critical" in severities


class TestGetDiseaseInfo:
    """Tests for get_disease_info method"""
    
    @pytest.fixture
    def kg(self):
        return MockKnowledgeGraph()
    
    def test_existing_disease(self, kg):
        """Test getting info for existing disease"""
        info = kg.get_disease_info("Pneumonia")
        
        assert info is not None
        assert "symptoms" in info
        assert "icd10" in info
    
    def test_nonexistent_disease(self, kg):
        """Test getting info for nonexistent disease"""
        info = kg.get_disease_info("NonexistentDisease123")
        
        assert info is None


class TestICD10Codes:
    """Tests for ICD-10 code handling"""
    
    @pytest.fixture
    def kg(self):
        return MockKnowledgeGraph()
    
    def test_all_diseases_have_icd10(self, kg):
        """Test all diseases have ICD-10 codes"""
        for name, disease in kg.DISEASES.items():
            assert "icd10" in disease
            assert disease["icd10"] is not None
            assert len(disease["icd10"]) > 0
    
    def test_icd10_format(self, kg):
        """Test ICD-10 codes follow standard format"""
        for name, disease in kg.DISEASES.items():
            icd10 = disease["icd10"]
            # ICD-10 codes typically start with a letter
            assert icd10[0].isalpha()


class TestGetKnowledgeGraph:
    """Tests for get_knowledge_graph factory function"""
    
    def test_returns_instance(self):
        """Test factory returns MockKnowledgeGraph instance"""
        kg = get_knowledge_graph()
        assert isinstance(kg, MockKnowledgeGraph)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
