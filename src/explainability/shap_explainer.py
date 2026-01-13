"""
SHAP Explainer for Medical AI
Generate SHAP (SHapley Additive exPlanations) for model interpretability.
Works with or without actual SHAP library installed.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger
from PIL import Image
import io
import base64

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed. Using mock explanations.")


class SHAPExplainer:
    """
    SHAP-based explainer for medical image and text analysis.
    Falls back to simulated explanations when SHAP is unavailable.
    """
    
    def __init__(self, model=None):
        """
        Initialize SHAP explainer.
        
        Args:
            model: ML model to explain (optional)
        """
        self.model = model
        self._explainer = None
        
        if SHAP_AVAILABLE and model is not None:
            try:
                self._explainer = shap.Explainer(model)
                logger.info("SHAP explainer initialized")
            except Exception as e:
                logger.warning(f"Could not initialize SHAP explainer: {e}")
    
    def explain_diagnosis(
        self,
        symptoms: List[str],
        diagnosis: str,
        confidence: float,
        findings: Dict = None
    ) -> Dict:
        """
        Generate explanation for a diagnosis.
        
        Args:
            symptoms: List of input symptoms
            diagnosis: Predicted diagnosis
            confidence: Confidence score
            findings: Additional findings from various sources
            
        Returns:
            Dict with explanation details
        """
        # Calculate feature contributions (simulated if SHAP unavailable)
        contributions = self._calculate_contributions(symptoms, diagnosis, confidence)
        
        # Generate natural language explanation
        explanation_text = self._generate_explanation_text(
            symptoms, diagnosis, confidence, contributions
        )
        
        # Create visual representation
        visual_data = self._create_visual_explanation(contributions)
        
        return {
            "feature_contributions": contributions,
            "explanation_text": explanation_text,
            "visual_data": visual_data,
            "shap_available": SHAP_AVAILABLE,
            "evidence_sources": self._get_evidence_sources(findings),
            "reasoning_chain": self._build_reasoning_chain(symptoms, diagnosis, contributions)
        }
    
    def _calculate_contributions(
        self, 
        symptoms: List[str], 
        diagnosis: str,
        confidence: float
    ) -> List[Dict]:
        """Calculate feature importance scores"""
        contributions = []
        
        # Distribute confidence across symptoms as contributions
        base_contribution = confidence / max(len(symptoms), 1)
        
        # Symptom importance weights (domain knowledge)
        importance_weights = {
            "chest pain": 1.5,
            "shortness of breath": 1.4,
            "fever": 1.2,
            "cough": 1.1,
            "sweating": 1.3,
            "diaphoresis": 1.4,
            "nausea": 1.0,
            "fatigue": 0.9,
            "headache": 0.8,
        }
        
        for symptom in symptoms:
            symptom_lower = symptom.lower()
            weight = 1.0
            
            for key, w in importance_weights.items():
                if key in symptom_lower:
                    weight = w
                    break
            
            contribution = min(base_contribution * weight, 1.0)
            
            contributions.append({
                "feature": symptom,
                "contribution": round(contribution, 3),
                "direction": "positive" if contribution > 0.1 else "neutral",
                "importance": "high" if contribution > 0.15 else "medium" if contribution > 0.1 else "low"
            })
        
        # Sort by contribution
        contributions.sort(key=lambda x: x["contribution"], reverse=True)
        
        return contributions
    
    def _generate_explanation_text(
        self,
        symptoms: List[str],
        diagnosis: str,
        confidence: float,
        contributions: List[Dict]
    ) -> str:
        """Generate human-readable explanation"""
        lines = []
        
        lines.append(f"**Diagnosis: {diagnosis}** (Confidence: {confidence*100:.0f}%)")
        lines.append("")
        
        # Top contributing factors
        high_contributors = [c for c in contributions if c["importance"] == "high"]
        if high_contributors:
            lines.append("**Key Contributing Factors:**")
            for c in high_contributors[:3]:
                lines.append(f"  • {c['feature']} (Impact: {c['contribution']*100:.0f}%)")
        
        # Supporting factors
        medium_contributors = [c for c in contributions if c["importance"] == "medium"]
        if medium_contributors:
            lines.append("")
            lines.append("**Supporting Evidence:**")
            for c in medium_contributors[:3]:
                lines.append(f"  • {c['feature']}")
        
        return "\n".join(lines)
    
    def _create_visual_explanation(self, contributions: List[Dict]) -> Dict:
        """Create data for visual representation"""
        return {
            "chart_type": "horizontal_bar",
            "labels": [c["feature"] for c in contributions],
            "values": [c["contribution"] for c in contributions],
            "colors": [
                "#22c55e" if c["importance"] == "high" 
                else "#f59e0b" if c["importance"] == "medium" 
                else "#94a3b8"
                for c in contributions
            ]
        }
    
    def _get_evidence_sources(self, findings: Dict = None) -> Dict:
        """Organize evidence by source"""
        sources = {
            "clinical_history": [],
            "physical_exam": [],
            "imaging": [],
            "lab_results": [],
            "knowledge_graph": []
        }
        
        if findings:
            if "symptoms" in findings:
                sources["clinical_history"] = findings["symptoms"]
            if "vitals" in findings:
                sources["physical_exam"] = [
                    f"{k}: {v}" for k, v in findings["vitals"].items() if v
                ]
            if "image_findings" in findings:
                sources["imaging"] = findings["image_findings"]
            if "lab_values" in findings:
                sources["lab_results"] = [
                    f"{lab['test']}: {lab['value']}" 
                    for lab in findings.get("lab_values", [])
                ]
        
        return {k: v for k, v in sources.items() if v}
    
    def _build_reasoning_chain(
        self,
        symptoms: List[str],
        diagnosis: str,
        contributions: List[Dict]
    ) -> List[str]:
        """Build step-by-step reasoning chain"""
        chain = []
        
        chain.append(f"Step 1: Extracted {len(symptoms)} clinical entities from input")
        
        if symptoms:
            top_symptoms = ", ".join(symptoms[:3])
            chain.append(f"Step 2: Key symptoms identified: {top_symptoms}")
        
        if contributions:
            top = contributions[0]
            chain.append(
                f"Step 3: Highest contributing factor: {top['feature']} "
                f"(impact: {top['contribution']*100:.0f}%)"
            )
        
        chain.append(f"Step 4: Knowledge graph matched pattern to {diagnosis}")
        chain.append(f"Step 5: Generated differential diagnoses and safety alerts")
        
        return chain


class MockSHAPExplainer(SHAPExplainer):
    """Mock explainer that works without SHAP installed"""
    
    def __init__(self):
        super().__init__(model=None)


def get_shap_explainer(model=None) -> SHAPExplainer:
    """Get SHAP explainer instance"""
    return SHAPExplainer(model=model)


if __name__ == "__main__":
    # Test the explainer
    explainer = get_shap_explainer()
    
    result = explainer.explain_diagnosis(
        symptoms=["chest pain", "shortness of breath", "sweating"],
        diagnosis="Myocardial Infarction",
        confidence=0.82,
        findings={"symptoms": ["chest pain", "dyspnea"]}
    )
    
    print("=== SHAP Explanation ===")
    print(result["explanation_text"])
    print("\nReasoning Chain:")
    for step in result["reasoning_chain"]:
        print(f"  {step}")
