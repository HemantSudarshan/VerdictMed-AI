"""
Unit tests for Simple Diagnostic Agent
Tests the core diagnostic reasoning functionality.
"""
import pytest
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.reasoning.simple_agent import SimpleDiagnosticAgent


class TestSimpleDiagnosticAgentInit:
    """Tests for SimpleDiagnosticAgent initialization"""
    
    def test_initialization(self):
        """Test agent initializes"""
        agent = SimpleDiagnosticAgent()
        assert agent is not None
    
    def test_has_required_attributes(self):
        """Test agent has required attributes"""
        agent = SimpleDiagnosticAgent()
        
        assert hasattr(agent, 'run')
        assert hasattr(agent, 'kg')
        assert hasattr(agent, 'nlp')


class TestDiagnosticRun:
    """Tests for the run method"""
    
    @pytest.fixture
    def agent(self):
        return SimpleDiagnosticAgent()
    
    @pytest.mark.asyncio
    async def test_basic_run(self, agent):
        """Test basic diagnosis run"""
        result = await agent.run(
            symptoms="Patient has fever and cough",
            patient_id="TEST001"
        )
        
        assert result is not None
        assert isinstance(result, dict)
    
    @pytest.mark.asyncio
    async def test_result_has_diagnosis(self, agent):
        """Test result contains primary diagnosis"""
        result = await agent.run(
            symptoms="Chest pain and shortness of breath",
            patient_id="TEST002"
        )
        
        assert "primary_diagnosis" in result
    
    @pytest.mark.asyncio
    async def test_result_has_confidence(self, agent):
        """Test result contains confidence score"""
        result = await agent.run(
            symptoms="Fever and fatigue",
            patient_id="TEST003"
        )
        
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 1
    
    @pytest.mark.asyncio
    async def test_result_has_differentials(self, agent):
        """Test result contains differential diagnoses"""
        result = await agent.run(
            symptoms="Headache and nausea",
            patient_id="TEST004"
        )
        
        assert "differential_diagnoses" in result
        assert isinstance(result["differential_diagnoses"], list)
    
    @pytest.mark.asyncio
    async def test_result_has_safety_alerts(self, agent):
        """Test result contains safety alerts"""
        result = await agent.run(
            symptoms="Mild symptoms",
            patient_id="TEST005"
        )
        
        assert "safety_alerts" in result
        assert isinstance(result["safety_alerts"], list)
    
    @pytest.mark.asyncio
    async def test_result_has_requires_review(self, agent):
        """Test result contains requires_review flag"""
        result = await agent.run(
            symptoms="Some symptoms",
            patient_id="TEST006"
        )
        
        assert "requires_review" in result
        assert isinstance(result["requires_review"], bool)


class TestEdgeCases:
    """Tests for edge cases"""
    
    @pytest.fixture
    def agent(self):
        return SimpleDiagnosticAgent()
    
    @pytest.mark.asyncio
    async def test_empty_symptoms(self, agent):
        """Test with empty symptoms"""
        result = await agent.run(
            symptoms="",
            patient_id="EMPTY001"
        )
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_very_long_symptoms(self, agent):
        """Test with very long symptom text"""
        long_text = "Patient has fever. " * 50
        result = await agent.run(
            symptoms=long_text,
            patient_id="LONG001"
        )
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_special_characters(self, agent):
        """Test with special characters"""
        result = await agent.run(
            symptoms="Temp: 101°F, BP: 120/80 mmHg",
            patient_id="SPECIAL001"
        )
        
        assert result is not None


class TestCriticalSymptoms:
    """Tests for critical symptom detection"""
    
    @pytest.fixture
    def agent(self):
        return SimpleDiagnosticAgent()
    
    @pytest.mark.asyncio
    async def test_chest_pain_detection(self, agent):
        """Test chest pain triggers appropriate response"""
        result = await agent.run(
            symptoms="Severe chest pain radiating to arm with sweating",
            patient_id="CRITICAL001"
        )
        
        # Should have high requires_review or safety alerts
        assert result.get("requires_review") == True or len(result.get("safety_alerts", [])) > 0
    
    @pytest.mark.asyncio
    async def test_stroke_symptoms(self, agent):
        """Test stroke symptoms"""
        result = await agent.run(
            symptoms="Sudden facial droop, slurred speech, arm weakness",
            patient_id="CRITICAL002"
        )
        
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
