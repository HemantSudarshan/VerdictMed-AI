"""
Unit tests for SHAP Explainer Module
Tests explanation generation, feature contributions, and reasoning chains.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.explainability.shap_explainer import SHAPExplainer, get_shap_explainer


class TestSHAPExplainerInit:
    """Tests for SHAPExplainer initialization"""
    
    def test_initialization(self):
        """Test explainer initializes"""
        explainer = SHAPExplainer()
        assert explainer is not None
    
    def test_get_shap_explainer(self):
        """Test factory function"""
        explainer = get_shap_explainer()
        assert explainer is not None
        assert isinstance(explainer, SHAPExplainer)


class TestExplainDiagnosis:
    """Tests for explain_diagnosis method"""
    
    @pytest.fixture
    def explainer(self):
        return SHAPExplainer()
    
    def test_returns_dict(self, explainer):
        """Test returns dictionary"""
        result = explainer.explain_diagnosis(
            symptoms=["fever", "cough"],
            diagnosis="Pneumonia",
            confidence=0.85
        )
        assert isinstance(result, dict)
    
    def test_has_required_keys(self, explainer):
        """Test result has all required keys"""
        result = explainer.explain_diagnosis(
            symptoms=["chest pain"],
            diagnosis="Test",
            confidence=0.7
        )
        
        required_keys = [
            "feature_contributions",
            "explanation_text",
            "visual_data",
            "reasoning_chain"
        ]
        for key in required_keys:
            assert key in result
    
    def test_feature_contributions_structure(self, explainer):
        """Test feature contributions structure"""
        result = explainer.explain_diagnosis(
            symptoms=["fever", "cough", "headache"],
            diagnosis="Flu",
            confidence=0.75
        )
        
        contributions = result["feature_contributions"]
        assert isinstance(contributions, list)
        assert len(contributions) == 3
        
        for c in contributions:
            assert "feature" in c
            assert "contribution" in c
            assert "importance" in c
    
    def test_contributions_sorted_by_importance(self, explainer):
        """Test contributions are sorted by contribution value"""
        result = explainer.explain_diagnosis(
            symptoms=["mild symptom", "chest pain", "fever"],
            diagnosis="Test",
            confidence=0.8
        )
        
        contributions = result["feature_contributions"]
        values = [c["contribution"] for c in contributions]
        assert values == sorted(values, reverse=True)
    
    def test_reasoning_chain_structure(self, explainer):
        """Test reasoning chain is a list of steps"""
        result = explainer.explain_diagnosis(
            symptoms=["fever"],
            diagnosis="Infection",
            confidence=0.6
        )
        
        chain = result["reasoning_chain"]
        assert isinstance(chain, list)
        assert len(chain) > 0
        
        for step in chain:
            assert isinstance(step, str)
            assert "Step" in step
    
    def test_visual_data_structure(self, explainer):
        """Test visual data for charts"""
        result = explainer.explain_diagnosis(
            symptoms=["symptom1", "symptom2"],
            diagnosis="Test",
            confidence=0.5
        )
        
        visual = result["visual_data"]
        assert "chart_type" in visual
        assert "labels" in visual
        assert "values" in visual
        assert "colors" in visual
    
    def test_explanation_text_format(self, explainer):
        """Test explanation text is human-readable"""
        result = explainer.explain_diagnosis(
            symptoms=["chest pain", "shortness of breath"],
            diagnosis="Myocardial Infarction",
            confidence=0.82
        )
        
        text = result["explanation_text"]
        assert isinstance(text, str)
        assert "Myocardial Infarction" in text
        assert "82%" in text or "Confidence" in text


class TestEmptyAndEdgeCases:
    """Tests for edge cases"""
    
    @pytest.fixture
    def explainer(self):
        return SHAPExplainer()
    
    def test_empty_symptoms(self, explainer):
        """Test with empty symptoms list"""
        result = explainer.explain_diagnosis(
            symptoms=[],
            diagnosis="Unknown",
            confidence=0.3
        )
        
        assert isinstance(result, dict)
        assert result["feature_contributions"] == []
    
    def test_single_symptom(self, explainer):
        """Test with single symptom"""
        result = explainer.explain_diagnosis(
            symptoms=["fever"],
            diagnosis="Infection",
            confidence=0.7
        )
        
        assert len(result["feature_contributions"]) == 1
    
    def test_many_symptoms(self, explainer):
        """Test with many symptoms"""
        symptoms = [f"symptom_{i}" for i in range(10)]
        result = explainer.explain_diagnosis(
            symptoms=symptoms,
            diagnosis="Complex Case",
            confidence=0.5
        )
        
        assert len(result["feature_contributions"]) == 10
    
    def test_low_confidence(self, explainer):
        """Test with very low confidence"""
        result = explainer.explain_diagnosis(
            symptoms=["vague symptom"],
            diagnosis="Uncertain",
            confidence=0.1
        )
        
        assert isinstance(result, dict)
    
    def test_high_confidence(self, explainer):
        """Test with high confidence"""
        result = explainer.explain_diagnosis(
            symptoms=["classic symptom"],
            diagnosis="Clear Diagnosis",
            confidence=0.99
        )
        
        assert isinstance(result, dict)


class TestEvidenceSources:
    """Tests for evidence source organization"""
    
    @pytest.fixture
    def explainer(self):
        return SHAPExplainer()
    
    def test_with_findings(self, explainer):
        """Test with findings dict"""
        result = explainer.explain_diagnosis(
            symptoms=["fever"],
            diagnosis="Test",
            confidence=0.7,
            findings={
                "symptoms": ["fever", "cough"],
                "vitals": {"temp": "101F", "hr": "90"}
            }
        )
        
        sources = result["evidence_sources"]
        assert "clinical_history" in sources
        assert "physical_exam" in sources
    
    def test_without_findings(self, explainer):
        """Test without findings dict"""
        result = explainer.explain_diagnosis(
            symptoms=["fever"],
            diagnosis="Test",
            confidence=0.7
        )
        
        sources = result["evidence_sources"]
        assert isinstance(sources, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
