"""
Tests for LangGraphDiagnosticAgent
Covers conditional branching, vector store integration, and visual explanation.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.reasoning.agent import (
    LangGraphDiagnosticAgent,
    should_escalate,
    extract_symptoms_node,
    generate_visual_explanation_node,
    retrieve_similar_cases_node,
    safety_validation_node,
    generate_standard_explanation_node,
    generate_escalation_explanation_node
)


class TestConditionalBranching:
    """Test the conditional escalation logic"""
    
    def test_should_escalate_returns_escalate_when_flag_true(self):
        """Test that escalation path is chosen when needs_escalation=True"""
        state = {"needs_escalation": True}
        result = should_escalate(state)
        assert result == "escalate"
    
    def test_should_escalate_returns_standard_when_flag_false(self):
        """Test that standard path is chosen when needs_escalation=False"""
        state = {"needs_escalation": False}
        result = should_escalate(state)
        assert result == "standard"
    
    def test_should_escalate_defaults_to_standard(self):
        """Test default behavior when flag is missing"""
        state = {}
        result = should_escalate(state)
        assert result == "standard"


class TestStandardExplanationNode:
    """Test the standard explanation output"""
    
    def test_standard_explanation_contains_diagnosis(self):
        """Standard explanation should contain the diagnosis"""
        state = {
            "primary_diagnosis": {"disease": "Pneumonia"},
            "confidence": 0.85,
            "extracted_symptoms": [{"symptom": "cough"}, {"symptom": "fever"}],
            "image_findings": {},
            "lab_findings": {},
            "similar_cases": [],
            "visual_explanation": None,
            "safety_alerts": []
        }
        result = generate_standard_explanation_node(state)
        
        assert "DIAGNOSIS: Pneumonia" in result["explanation"]
        assert "ESCALATION REQUIRED" not in result["explanation"]
        assert result["output_type"] == "standard"
    
    def test_standard_explanation_includes_visual_explanation(self):
        """Standard explanation should include visual interpretation if available"""
        state = {
            "primary_diagnosis": {"disease": "Pneumonia"},
            "confidence": 0.85,
            "extracted_symptoms": [],
            "image_findings": {"top_finding": "consolidation", "confidence": 0.8},
            "lab_findings": {},
            "similar_cases": [],
            "visual_explanation": {
                "available": True,
                "interpretation": "Model focused on bilateral infiltrates"
            },
            "safety_alerts": []
        }
        result = generate_standard_explanation_node(state)
        
        assert "bilateral infiltrates" in result["explanation"]


class TestEscalationExplanationNode:
    """Test the escalation explanation output"""
    
    def test_escalation_explanation_contains_warning(self):
        """Escalation explanation should contain ESCALATION REQUIRED"""
        state = {
            "primary_diagnosis": {"disease": "Unknown"},
            "confidence": 0.35,
            "safety_alerts": ["LOW_CONFIDENCE"],
            "conflicts_detected": [],
            "escalation_reason": "Confidence too low"
        }
        result = generate_escalation_explanation_node(state)
        
        assert "ESCALATION REQUIRED" in result["explanation"]
        assert "BELOW THRESHOLD" in result["explanation"]
        assert result["output_type"] == "escalation"
    
    def test_escalation_lists_conflicts(self):
        """Escalation should list detected conflicts"""
        state = {
            "primary_diagnosis": {"disease": "Pneumonia"},
            "confidence": 0.5,
            "safety_alerts": [],
            "conflicts_detected": [
                {"type": "MODALITY_DISAGREEMENT", "description": "Labs vs image conflict"}
            ],
            "escalation_reason": ""
        }
        result = generate_escalation_explanation_node(state)
        
        assert "conflict" in result["explanation"].lower()


class TestVisualExplanationNode:
    """Test the visual explanation generation"""
    
    def test_visual_explanation_with_image(self):
        """Should generate explanation when image findings present"""
        state = {
            "image_path": "/path/to/xray.png",
            "image_findings": {
                "top_finding": "pneumonia",
                "confidence": 0.85
            }
        }
        result = generate_visual_explanation_node(state)
        
        assert result["visual_explanation"] is not None
        assert result["visual_explanation"]["method"] == "attention_rollout"
        assert result["visual_explanation"]["available"] == True
    
    def test_visual_explanation_without_image(self):
        """Should return None when no image provided"""
        state = {
            "image_path": None,
            "image_findings": {}
        }
        result = generate_visual_explanation_node(state)
        
        assert result["visual_explanation"] is None
    
    def test_visual_explanation_with_image_error(self):
        """Should return None when image analysis failed"""
        state = {
            "image_path": "/path/to/xray.png",
            "image_findings": {"error": "Failed to load image"}
        }
        result = generate_visual_explanation_node(state)
        
        assert result["visual_explanation"] is None


class TestVectorStoreIntegration:
    """Test vector store retrieval with mocking"""
    
    @patch('src.reasoning.agent.weaviate')
    @patch('src.reasoning.agent.SentenceTransformer')
    def test_similar_cases_uses_near_vector(self, mock_transformer, mock_weaviate):
        """Verify that near_vector search is called (not near_text)"""
        # Setup mocks
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [0.1] * 384  # Fake embedding
        mock_transformer.return_value = mock_embedder
        
        mock_client = MagicMock()
        mock_query = MagicMock()
        mock_query.get.return_value.with_near_vector.return_value.with_limit.return_value.do.return_value = {
            "data": {"Get": {"PatientCase": [{"case_id": "123", "diagnosis": "Pneumonia"}]}}
        }
        mock_client.query = mock_query
        mock_weaviate.Client.return_value = mock_client
        
        state = {
            "extracted_symptoms": [
                {"symptom": "fever", "negated": False},
                {"symptom": "cough", "negated": False}
            ]
        }
        
        with patch('src.reasoning.agent.get_settings') as mock_settings:
            mock_settings.return_value.weaviate_url = "http://localhost:8080"
            result = retrieve_similar_cases_node(state)
        
        # Verify near_vector was called
        mock_query.get.return_value.with_near_vector.assert_called()
        assert len(result["similar_cases"]) == 1
    
    @patch('src.reasoning.agent.weaviate')
    @patch('src.reasoning.agent.SentenceTransformer')
    def test_similar_cases_handles_failure_gracefully(self, mock_transformer, mock_weaviate):
        """Verify workflow continues when vector store fails"""
        mock_weaviate.Client.side_effect = Exception("Connection refused")
        
        state = {
            "extracted_symptoms": [{"symptom": "fever", "negated": False}]
        }
        
        result = retrieve_similar_cases_node(state)
        
        # Should return empty list, not crash
        assert result["similar_cases"] == []
    
    def test_similar_cases_empty_with_no_symptoms(self):
        """Should return empty when no symptoms provided"""
        state = {"extracted_symptoms": []}
        result = retrieve_similar_cases_node(state)
        assert result["similar_cases"] == []


class TestSafetyValidationNode:
    """Test safety validation logic"""
    
    @patch('src.reasoning.agent.SafetyValidator')
    def test_low_confidence_triggers_escalation(self, mock_validator):
        """Low confidence should trigger escalation"""
        mock_validator.return_value.validate_diagnosis.return_value = {
            "alerts": [],
            "requires_escalation": False,
            "escalation_reasons": []
        }
        
        state = {"confidence": 0.4, "conflicts_detected": []}
        result = safety_validation_node(state)
        
        assert result["needs_escalation"] == True
    
    @patch('src.reasoning.agent.SafetyValidator')
    def test_conflicts_trigger_escalation(self, mock_validator):
        """Detected conflicts should trigger escalation"""
        mock_validator.return_value.validate_diagnosis.return_value = {
            "alerts": [],
            "requires_escalation": False,
            "escalation_reasons": []
        }
        
        state = {
            "confidence": 0.8,
            "conflicts_detected": [{"type": "MODALITY_DISAGREEMENT"}]
        }
        result = safety_validation_node(state)
        
        assert result["needs_escalation"] == True


class TestExtractSymptomsNode:
    """Test symptom extraction node"""
    
    def test_empty_text_returns_empty_list(self):
        """Empty text should return empty symptoms"""
        state = {"symptoms_text": ""}
        result = extract_symptoms_node(state)
        
        assert result["extracted_symptoms"] == []
        assert result["extracted_vitals"] == {}
    
    @patch('src.reasoning.agent.ClinicalNLPPipeline')
    def test_nlp_failure_returns_empty(self, mock_nlp):
        """NLP failure should return empty, not crash"""
        mock_nlp.side_effect = Exception("Model not loaded")
        
        state = {"symptoms_text": "Patient has fever"}
        result = extract_symptoms_node(state)
        
        assert result["extracted_symptoms"] == []


@pytest.mark.asyncio
class TestLangGraphAgentIntegration:
    """Integration tests for the full agent"""
    
    @patch('src.reasoning.agent.ClinicalNLPPipeline')
    async def test_agent_runs_without_error(self, mock_nlp):
        """Agent should complete without error on basic input"""
        mock_nlp.return_value.analyze_clinical_note.return_value = {
            "symptoms": [{"symptom": "fever", "negated": False}],
            "vitals": {}
        }
        
        agent = LangGraphDiagnosticAgent()
        result = await agent.run({
            "patient_id": "TEST001",
            "symptoms": "Patient has fever"
        })
        
        assert result is not None
        assert "processing_time_ms" in result
        assert "explanation" in result
    
    @patch('src.reasoning.agent.ClinicalNLPPipeline')
    async def test_agent_includes_output_type(self, mock_nlp):
        """Agent result should include output_type field"""
        mock_nlp.return_value.analyze_clinical_note.return_value = {
            "symptoms": [],
            "vitals": {}
        }
        
        agent = LangGraphDiagnosticAgent()
        result = await agent.run({"symptoms": ""})
        
        assert "output_type" in result
        assert result["output_type"] in ["standard", "escalation", "error"]
