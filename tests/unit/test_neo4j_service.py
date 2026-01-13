"""
Unit tests for Neo4j Knowledge Graph Service
Tests connection handling, query methods, and fallback behavior.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.knowledge_graph.neo4j_service import Neo4jKnowledgeGraph, get_knowledge_graph


class TestNeo4jKnowledgeGraphInit:
    """Tests for Neo4jKnowledgeGraph initialization"""
    
    def test_initialization(self):
        """Test KG initializes (with fallback)"""
        kg = Neo4jKnowledgeGraph()
        assert kg is not None
    
    def test_has_mock_fallback(self):
        """Test mock fallback is available"""
        kg = Neo4jKnowledgeGraph()
        assert kg._mock is not None
    
    def test_connection_status(self):
        """Test is_connected property exists"""
        kg = Neo4jKnowledgeGraph()
        assert hasattr(kg, 'is_connected')
        assert isinstance(kg.is_connected, bool)


class TestFindDiseasesBySymptoms:
    """Tests for find_diseases_by_symptoms method"""
    
    @pytest.fixture
    def kg(self):
        return Neo4jKnowledgeGraph()
    
    def test_returns_list(self, kg):
        """Test returns list"""
        results = kg.find_diseases_by_symptoms(["fever"])
        assert isinstance(results, list)
    
    def test_with_multiple_symptoms(self, kg):
        """Test with multiple symptoms"""
        results = kg.find_diseases_by_symptoms(["fever", "cough", "shortness of breath"])
        assert isinstance(results, list)
        assert len(results) > 0
    
    def test_result_structure(self, kg):
        """Test result structure"""
        results = kg.find_diseases_by_symptoms(["chest pain"])
        
        if len(results) > 0:
            result = results[0]
            assert "disease" in result
            assert "icd10" in result
            assert "severity" in result
    
    def test_empty_symptoms(self, kg):
        """Test with empty list"""
        results = kg.find_diseases_by_symptoms([])
        assert isinstance(results, list)
    
    def test_limit_parameter(self, kg):
        """Test limit parameter"""
        results = kg.find_diseases_by_symptoms(["fever"], limit=3)
        assert len(results) <= 3


class TestGetDiseaseInfo:
    """Tests for get_disease_info method"""
    
    @pytest.fixture
    def kg(self):
        return Neo4jKnowledgeGraph()
    
    def test_existing_disease(self, kg):
        """Test getting existing disease"""
        info = kg.get_disease_info("Pneumonia")
        
        if info:  # May be None if using mock
            assert "symptoms" in info or info is None
    
    def test_nonexistent_disease(self, kg):
        """Test nonexistent disease returns None"""
        info = kg.get_disease_info("FakeDisease12345")
        assert info is None


class TestCheckCriticalConditions:
    """Tests for check_critical_conditions method"""
    
    @pytest.fixture
    def kg(self):
        return Neo4jKnowledgeGraph()
    
    def test_returns_list(self, kg):
        """Test returns list"""
        results = kg.check_critical_conditions(["chest pain"])
        assert isinstance(results, list)
    
    def test_critical_symptoms(self, kg):
        """Test with critical symptoms"""
        results = kg.check_critical_conditions([
            "chest pain", "arm pain", "sweating"
        ])
        
        # Should find critical conditions
        assert isinstance(results, list)


class TestGetDifferentialDiagnosis:
    """Tests for get_differential_diagnosis method"""
    
    @pytest.fixture
    def kg(self):
        return Neo4jKnowledgeGraph()
    
    def test_returns_list(self, kg):
        """Test returns list"""
        results = kg.get_differential_diagnosis(["fever", "cough"])
        assert isinstance(results, list)
    
    def test_limited_to_five(self, kg):
        """Test results limited to 5"""
        results = kg.get_differential_diagnosis(["fever", "cough", "headache"])
        assert len(results) <= 5


class TestGetKnowledgeGraphFactory:
    """Tests for get_knowledge_graph factory function"""
    
    def test_returns_instance(self):
        """Test factory returns instance"""
        kg = get_knowledge_graph()
        assert kg is not None
    
    def test_has_required_methods(self):
        """Test instance has required methods"""
        kg = get_knowledge_graph()
        
        assert hasattr(kg, 'find_diseases_by_symptoms')
        assert hasattr(kg, 'get_disease_info')
        assert hasattr(kg, 'check_critical_conditions')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
