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
        result = await agent.run({
            "symptoms": "Severe chest pain radiating to arm with sweating",
            "patient_id": "CRITICAL001"
        })
        
        # Should have high requires_review or safety alerts
        assert result.get("needs_escalation") == True or len(result.get("safety_alerts", [])) > 0
    
    @pytest.mark.asyncio
    async def test_stroke_symptoms(self, agent):
        """Test stroke symptoms"""
        result = await agent.run({
            "symptoms": "Sudden facial droop, slurred speech, arm weakness",
            "patient_id": "CRITICAL002"
        })
        
        assert result is not None


class TestPRDRequirements:
    """
    Tests specific to PRD Stage 9.2 requirements.
    Validates core diagnostic capabilities per specification.
    """
    
    @pytest.fixture
    def agent(self):
        return SimpleDiagnosticAgent()
    
    @pytest.mark.asyncio
    async def test_pneumonia_detection(self, agent):
        """
        Test typical pneumonia case detection.
        PRD Requirement: Should identify pneumonia from classic presentation.
        """
        result = await agent.run({
            "symptoms": "Patient reports fever x3 days, productive cough, shortness of breath",
            "patient_id": "PRD_PNEUMONIA_001"
        })
        
        # Should have reasonable confidence
        assert result["confidence"] > 0.5, "Pneumonia case should have >50% confidence"
        
        # Should include pneumonia in differential
        differential = result.get("differential_diagnoses", [])
        pneumonia_found = any(
            "pneumonia" in d.get("disease", "").lower() or 
            "respiratory infection" in d.get("disease", "").lower()
            for d in differential
        )
        assert pneumonia_found, "Pneumonia should be in differential diagnoses"
        
        # Should extract key symptoms
        symptoms = result.get("extracted_symptoms", [])
        symptom_names = [s.get("symptom", "").lower() for s in symptoms]
        assert any("fever" in s for s in symptom_names), "Should extract fever"
        assert any("cough" in s for s in symptom_names), "Should extract cough"
    
    @pytest.mark.asyncio
    async def test_low_confidence_escalation(self, agent):
        """
        Test that low confidence triggers escalation.
        PRD Requirement: Vague symptoms should trigger human review.
        """
        result = await agent.run({
            "symptoms": "Patient feels unwell",  # Intentionally vague
            "patient_id": "PRD_VAGUE_001"
        })
        
        # Should escalate due to insufficient information OR have low confidence
        escalated = result.get("needs_escalation", False)
        low_confidence = result.get("confidence", 1.0) < 0.6
        
        assert escalated or low_confidence, \
            "Vague symptoms should trigger escalation or have confidence < 60%"
        
        # Should have safety alerts or explanation of uncertainty
        if not escalated:
            explanation = result.get("explanation", "")
            assert explanation, "Should provide explanation even with low confidence"
    
    @pytest.mark.asyncio
    async def test_critical_condition_alert(self, agent):
        """
        Test that critical conditions are flagged.
        PRD Requirement: MI symptoms should trigger critical alerts.
        """
        result = await agent.run({
            "symptoms": "Severe chest pain radiating to left arm, diaphoresis, nausea",
            "patient_id": "PRD_MI_001"
        })
        
        # Should flag as critical OR have high escalation
        has_critical_alert = "CRITICAL_CONDITION_DETECTED" in result.get("safety_alerts", [])
        needs_escalation = result.get("needs_escalation", False)
        
        # Should identify cardiac-related condition
        differential = result.get("differential_diagnoses", [])
        cardiac_found = any(
            any(term in d.get("disease", "").lower() 
                for term in ["myocardial", "heart", "cardiac", "chest pain"])
            for d in differential
        )
        
        assert has_critical_alert or needs_escalation or cardiac_found, \
            "MI symptoms should trigger critical response"
    
    @pytest.mark.asyncio
    async def test_negation_handling(self, agent):
        """
        Test that negated symptoms are handled correctly.
        PRD Requirement: Should not diagnose based on denied symptoms.
        """
        result = await agent.run({
            "symptoms": "Patient denies fever, no cough, no chest pain. Reports mild fatigue.",
            "patient_id": "PRD_NEGATION_001"
        })
        
        # Should not diagnose respiratory conditions
        symptoms = result.get("extracted_symptoms", [])
        
        # Check if fever is properly marked as negated
        fever_symptoms = [s for s in symptoms if "fever" in s.get("symptom", "").lower()]
        if fever_symptoms:
            # If fever was extracted, it should be marked as negated
            assert any(s.get("negated", False) for s in fever_symptoms), \
                "Denied fever should be marked as negated"
        
        # Should not have high confidence in fever-related conditions
        differential = result.get("differential_diagnoses", [])
        if differential:
            top_diagnosis = differential[0].get("disease", "").lower()
            # Shouldn't strongly suggest conditions requiring fever
            confidence = result.get("confidence", 0)
            if any(term in top_diagnosis for term in ["infection", "flu", "pneumonia"]):
                assert confidence < 0.75, \
                    "Should have lower confidence for infection without fever"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

