"""
Safety Validator
Comprehensive safety validation for medical diagnoses.
Implements confidence checks, critical condition detection, and escalation logic.
"""

from typing import Dict, List, Optional
from loguru import logger


class SafetyValidator:
    """Comprehensive safety validation for medical diagnoses"""
    
    def __init__(self, config=None):
        """
        Initialize safety validator.
        
        Args:
            config: Configuration object with thresholds
        """
        self.config = config
        
        # Default thresholds
        self.min_confidence_threshold = getattr(config, 'min_confidence_threshold', 0.55)
        self.escalation_threshold = getattr(config, 'escalation_threshold', 0.70)
        
        # High-risk conditions requiring immediate attention
        self.high_risk_conditions = [
            "acute myocardial infarction",
            "pulmonary embolism",
            "stroke",
            "sepsis",
            "anaphylaxis",
            "meningitis",
            "aortic dissection",
            "tension pneumothorax",
            "cardiac arrest",
            "respiratory failure"
        ]
        
        # Symptom-disease requirements for sanity checks
        self.required_symptoms = {
            "myocardial infarction": ["chest pain", "chest_pain"],
            "pneumonia": ["cough", "fever"],
            "appendicitis": ["abdominal pain", "abdominal_pain"],
            "meningitis": ["headache", "fever", "neck stiffness"],
            "pulmonary embolism": ["dyspnea", "chest pain", "shortness of breath"]
        }
        
        logger.info("SafetyValidator initialized")
    
    def validate_diagnosis(self, state: Dict) -> Dict:
        """
        Run all safety validations.
        
        Args:
            state: Diagnostic state with predictions and findings
            
        Returns:
            Dict with validation results, alerts, and escalation decision
        """
        validations = {
            "confidence_check": self._check_confidence(state),
            "critical_condition_check": self._check_critical_conditions(state),
            "data_quality_check": self._check_data_quality(state),
            "sanity_check": self._perform_sanity_check(state),
            "conflict_check": self._check_signal_conflicts(state)
        }
        
        # Aggregate results
        all_passed = all(v["passed"] for v in validations.values())
        alerts = []
        for name, result in validations.items():
            if not result["passed"]:
                alerts.extend(result.get("alerts", []))
            elif result.get("alerts"):  # Some checks always pass but add alerts
                alerts.extend(result.get("alerts", []))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_alerts = []
        for alert in alerts:
            if alert not in seen:
                seen.add(alert)
                unique_alerts.append(alert)
        
        return {
            "all_passed": all_passed,
            "validations": validations,
            "alerts": unique_alerts,
            "requires_escalation": not all_passed,
            "escalation_reasons": [a for a in unique_alerts if "CRITICAL" in a or "LOW_CONFIDENCE" in a]
        }
    
    def _check_confidence(self, state: Dict) -> Dict:
        """Check if confidence meets threshold"""
        confidence = state.get("confidence", 0)
        threshold = self.min_confidence_threshold
        
        passed = confidence >= threshold
        
        alerts = []
        if not passed:
            alerts.append(f"LOW_CONFIDENCE: {confidence:.1%} below threshold {threshold:.1%}")
        elif confidence < self.escalation_threshold:
            alerts.append(f"MODERATE_CONFIDENCE: {confidence:.1%}")
        
        return {
            "passed": passed,
            "confidence": confidence,
            "threshold": threshold,
            "alerts": alerts
        }
    
    def _check_critical_conditions(self, state: Dict) -> Dict:
        """Flag critical conditions requiring immediate attention"""
        primary = state.get("primary_diagnosis", {})
        disease = primary.get("disease", "").lower() if isinstance(primary, dict) else ""
        
        is_critical = any(cond in disease for cond in self.high_risk_conditions)
        
        alerts = []
        if is_critical:
            alerts.append(f"⚠️ CRITICAL: {disease} requires immediate specialist attention")
        
        # Check severity field
        severity = primary.get("severity", "") if isinstance(primary, dict) else ""
        if severity == "critical":
            is_critical = True
            if not alerts:
                alerts.append(f"⚠️ CRITICAL: High severity condition detected")
        
        return {
            "passed": True,  # Always pass, but alert
            "is_critical": is_critical,
            "alerts": alerts
        }
    
    def _check_data_quality(self, state: Dict) -> Dict:
        """Check if input data quality is sufficient"""
        issues = []
        
        # Check symptoms
        symptoms = state.get("extracted_symptoms", [])
        positive_symptoms = [s for s in symptoms if not s.get("negated", False)]
        
        if len(positive_symptoms) < 2:
            issues.append("Insufficient positive symptoms extracted")
        
        # Check image quality
        image = state.get("image_findings", {})
        if image.get("needs_review"):
            issues.append(f"Image quality issue: {image.get('review_reason', 'unspecified')}")
        
        # Check if essential data is missing
        if not state.get("symptoms_text") and not positive_symptoms:
            issues.append("No symptom data provided")
        
        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "alerts": issues
        }
    
    def _perform_sanity_check(self, state: Dict) -> Dict:
        """Verify diagnosis makes logical sense"""
        primary = state.get("primary_diagnosis", {})
        disease = primary.get("disease", "").lower() if isinstance(primary, dict) else ""
        
        symptoms = []
        for s in state.get("extracted_symptoms", []):
            if not s.get("negated", False):
                symptoms.append(s.get("symptom", "").lower())
        
        # Check if required symptoms are present
        for condition, required in self.required_symptoms.items():
            if condition in disease:
                has_required = any(
                    any(req in s for s in symptoms) 
                    for req in required
                )
                if not has_required:
                    return {
                        "passed": False,
                        "alerts": [f"SANITY_CHECK_FAILED: Diagnosis '{disease}' but missing typical symptom(s): {required}"]
                    }
        
        return {"passed": True, "alerts": []}
    
    def _check_signal_conflicts(self, state: Dict) -> Dict:
        """Check for conflicting signals between modalities"""
        conflicts = state.get("conflicts_detected", [])
        
        # Check for image vs symptom conflicts
        image_finding = state.get("image_findings", {}).get("top_finding", "")
        primary_disease = state.get("primary_diagnosis", {})
        disease_name = primary_disease.get("disease", "") if isinstance(primary_disease, dict) else ""
        
        if image_finding and disease_name:
            if not self._signals_align(image_finding, disease_name):
                conflicts.append(f"Image suggests '{image_finding}' but symptoms suggest '{disease_name}'")
        
        return {
            "passed": len(conflicts) == 0,
            "conflicts": conflicts,
            "alerts": [f"SIGNAL_CONFLICT: {c}" for c in conflicts]
        }
    
    def _signals_align(self, image_finding: str, disease: str) -> bool:
        """Check if image and symptom findings align"""
        disease_lower = disease.lower()
        finding_lower = image_finding.lower()
        
        # If image shows "normal", check if disease should show on image
        if "normal" in finding_lower:
            imaging_visible_conditions = ["pneumonia", "cardiomegaly", "effusion", "mass", "nodule"]
            if any(cond in disease_lower for cond in imaging_visible_conditions):
                return False  # Conflict: normal image but imaging-visible disease
        
        alignment_map = {
            "pneumonia": ["pneumonia", "consolidation", "infiltrate", "opacity"],
            "tuberculosis": ["tuberculosis", "cavity", "infiltrate", "tb"],
            "heart": ["cardiomegaly", "enlarged heart", "cardiac"],
            "pleural": ["effusion", "pleural"],
            "cancer": ["mass", "nodule", "tumor"]
        }
        
        for key, values in alignment_map.items():
            if key in disease_lower:
                if any(v in finding_lower for v in values):
                    return True
        
        return True  # Default to aligned if unknown relationship


def generate_disclaimer() -> str:
    """Generate standard medical AI disclaimer"""
    return """
⚠️ IMPORTANT DISCLAIMER ⚠️

This Clinical Decision Support System (CDSS) is designed as a SECOND OPINION tool only.
It is NOT a replacement for professional medical judgment.

This system:
- Can make mistakes (no system is 100% accurate)
- Should NOT be the sole basis for diagnosis
- Should NOT replace specialist consultation
- Requires physician validation before any clinical decision

The treating physician is ALWAYS responsible for final diagnosis and treatment decisions.

By using this system, you acknowledge:
1. You understand its limitations
2. You will always verify output with clinical judgment
3. You accept full responsibility for patient care decisions
4. You will not rely solely on system recommendations
"""
