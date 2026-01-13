"""
Integration tests for full diagnostic pipeline
Tests end-to-end flow from symptoms to diagnosis.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.reasoning.simple_agent import SimpleDiagnosticAgent


class TestFullPipeline:
    """Tests for complete diagnostic pipeline"""
    
    @pytest.fixture
    def agent(self):
        """Create diagnostic agent"""
        return SimpleDiagnosticAgent()
    
    @pytest.mark.asyncio
    async def test_basic_diagnosis(self, agent):
        """Test basic diagnosis flow"""
        result = await agent.run(
            symptoms="Patient has fever and cough for 3 days",
            patient_id="TEST001"
        )
        
        assert result is not None
        assert "primary_diagnosis" in result
        assert "confidence" in result
    
    @pytest.mark.asyncio
    async def test_chest_pain_case(self, agent):
        """Test chest pain case"""
        result = await agent.run(
            symptoms="58 year old male with chest pain radiating to left arm, diaphoresis",
            patient_id="TEST002"
        )
        
        assert result is not None
        assert "safety_alerts" in result
    
    @pytest.mark.asyncio
    async def test_respiratory_case(self, agent):
        """Test respiratory symptoms case"""
        result = await agent.run(
            symptoms="Fever, productive cough, shortness of breath for 5 days",
            patient_id="TEST003"
        )
        
        assert result is not None
        assert "differential_diagnoses" in result
    
    @pytest.mark.asyncio
    async def test_empty_symptoms(self, agent):
        """Test handling of empty symptoms"""
        result = await agent.run(
            symptoms="",
            patient_id="TEST004"
        )
        
        # Should still return a result, possibly with low confidence
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_result_has_required_fields(self, agent):
        """Test result contains all required fields"""
        result = await agent.run(
            symptoms="Headache and nausea",
            patient_id="TEST005"
        )
        
        required_fields = [
            "primary_diagnosis",
            "confidence",
            "differential_diagnoses",
            "safety_alerts",
            "requires_review",
            "explanation"
        ]
        
        for field in required_fields:
            assert field in result, f"Missing field: {field}"
    
    @pytest.mark.asyncio
    async def test_low_confidence_triggers_review(self, agent):
        """Test that low confidence triggers physician review"""
        result = await agent.run(
            symptoms="vague symptoms unclear presentation",
            patient_id="TEST006"
        )
        
        # Low confidence should trigger review
        if result.get("confidence", 0) < 0.55:
            assert result.get("requires_review") == True


class TestSafetyIntegration:
    """Tests for safety layer integration"""
    
    @pytest.fixture
    def agent(self):
        return SimpleDiagnosticAgent()
    
    @pytest.mark.asyncio
    async def test_critical_symptoms_alert(self, agent):
        """Test critical symptoms generate alerts"""
        result = await agent.run(
            symptoms="severe chest pain, difficulty breathing, cold sweats",
            patient_id="TEST_CRITICAL"
        )
        
        assert result is not None
        # Should have safety alerts for critical symptoms
        assert "safety_alerts" in result


class TestDiagnosisQuality:
    """Tests for diagnosis quality"""
    
    @pytest.fixture
    def agent(self):
        return SimpleDiagnosticAgent()
    
    @pytest.mark.asyncio
    async def test_icd10_present(self, agent):
        """Test ICD-10 codes are included"""
        result = await agent.run(
            symptoms="fever and cough",
            patient_id="TEST_ICD"
        )
        
        primary = result.get("primary_diagnosis", {})
        # ICD-10 should be present
        if isinstance(primary, dict):
            assert "icd10" in primary or primary.get("icd10") is not None
    
    @pytest.mark.asyncio
    async def test_differential_diagnoses_ranked(self, agent):
        """Test differential diagnoses are ranked by confidence"""
        result = await agent.run(
            symptoms="chest pain and shortness of breath",
            patient_id="TEST_DIFF"
        )
        
        diffs = result.get("differential_diagnoses", [])
        if len(diffs) > 1:
            # Should be sorted by confidence (descending)
            confidences = [d.get("confidence", 0) for d in diffs]
            assert confidences == sorted(confidences, reverse=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
