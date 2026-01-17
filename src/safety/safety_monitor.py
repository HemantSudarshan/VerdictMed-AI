"""
Safety Monitor
Wrapper for diagnosis with fallback mechanisms, blocking logic, and audit logging.
Implements the safety net pattern from CDSS_Challenges_Safety_Guide.md.
"""

from typing import Dict, Optional, List
from datetime import datetime
from loguru import logger
from enum import Enum


class DiagnosisStatus(str, Enum):
    """Status of diagnosis attempt"""
    SUCCESS = "success"
    BLOCKED = "blocked"
    FALLBACK = "fallback"
    ERROR = "error"


class SafetyMonitor:
    """
    Wrapper for diagnosis with comprehensive safety checks and fallback mechanisms.
    
    Features:
    - Primary LLM diagnosis with fallback to rule-based
    - Automatic blocking for low confidence
    - Sanity checks on diagnosis output
    - Model drift detection
    - Complete audit trail
    """
    
    def __init__(self, config=None):
        """
        Initialize safety monitor.
        
        Args:
            config: Configuration with thresholds
        """
        self.config = config
        self.min_confidence = 0.55  # Minimum confidence to proceed
        self.conflict_threshold = 0.5  # Max difference between modalities
        self.daily_metrics: Dict = {}
        self.alerts: List[Dict] = []
        self.audit_log: List[Dict] = []
        
        # Rule-based fallback patterns
        self.symptom_disease_rules = {
            frozenset(["fever", "cough", "shortness of breath"]): {
                "disease": "Respiratory Infection",
                "confidence": 0.65,
                "severity": "moderate"
            },
            frozenset(["chest pain", "shortness of breath", "sweating"]): {
                "disease": "Possible Cardiac Event",
                "confidence": 0.70,
                "severity": "critical",
                "escalation": True
            },
            frozenset(["headache", "fever", "neck stiffness"]): {
                "disease": "Possible Meningitis",
                "confidence": 0.60,
                "severity": "critical",
                "escalation": True
            },
            frozenset(["abdominal pain", "nausea", "vomiting"]): {
                "disease": "Gastrointestinal Issue",
                "confidence": 0.60,
                "severity": "moderate"
            },
            frozenset(["fever", "fatigue", "body aches"]): {
                "disease": "Viral Syndrome",
                "confidence": 0.55,
                "severity": "mild"
            },
        }
        
        # Required symptoms for sanity check
        self.required_symptoms = {
            "myocardial infarction": ["chest pain", "shortness of breath"],
            "pneumonia": ["cough", "fever"],
            "stroke": ["weakness", "numbness", "confusion"],
            "sepsis": ["fever", "tachycardia"],
            "appendicitis": ["abdominal pain"],
        }
    
    async def diagnose_with_safety(
        self, 
        patient_data: Dict,
        primary_agent=None
    ) -> Dict:
        """
        Run diagnosis with full safety net.
        
        Args:
            patient_data: Patient symptoms, labs, image_path
            primary_agent: Primary diagnostic agent (optional, lazy loaded)
            
        Returns:
            Dict with diagnosis, status, and safety metadata
        """
        start_time = datetime.now()
        request_id = patient_data.get("request_id", str(hash(str(patient_data))))
        
        result = {
            "request_id": request_id,
            "status": DiagnosisStatus.BLOCKED,
            "diagnosis": None,
            "confidence": 0.0,
            "safety_alerts": [],
            "needs_escalation": False,
            "explanation": "",
            "_metadata": {
                "primary_used": False,
                "fallback_used": False,
                "blocked": False,
                "processing_time_ms": 0
            }
        }
        
        try:
            # Step 1: Try primary LLM-based diagnosis
            if primary_agent is None:
                from src.reasoning.simple_agent import SimpleDiagnosticAgent
                primary_agent = SimpleDiagnosticAgent()
            
            diagnosis = await primary_agent.run(patient_data)
            result["_metadata"]["primary_used"] = True
            
        except Exception as e:
            # Fallback to rule-based diagnosis
            logger.warning(f"Primary agent failed: {e}. Using fallback.")
            self._log_alert("PRIMARY_FAILED", str(e))
            
            diagnosis = self._rule_based_diagnose(patient_data)
            diagnosis["_fallback"] = True
            diagnosis["confidence"] *= 0.7  # Reduce confidence for fallback
            result["_metadata"]["fallback_used"] = True
        
        # Step 2: Safety checks
        confidence = diagnosis.get("confidence", 0.0)
        
        # Check 2a: Minimum confidence threshold
        if confidence < self.min_confidence:
            self._log_alert("LOW_CONFIDENCE", f"Confidence {confidence:.2f} below threshold {self.min_confidence}")
            result["status"] = DiagnosisStatus.BLOCKED
            result["safety_alerts"].append("LOW_CONFIDENCE_BLOCKED")
            result["needs_escalation"] = True
            result["explanation"] = "Diagnosis blocked due to low confidence. Specialist review required."
            result["_metadata"]["blocked"] = True
            
            self._audit_log(request_id, "BLOCKED", patient_data, result)
            return result
        
        # Check 2b: Sanity check - does diagnosis match symptoms?
        if not self._sanity_check(diagnosis, patient_data):
            self._log_alert("SANITY_CHECK_FAILED", "Diagnosis doesn't match symptoms")
            result["status"] = DiagnosisStatus.BLOCKED
            result["safety_alerts"].append("SANITY_CHECK_FAILED")
            result["needs_escalation"] = True
            result["explanation"] = "Diagnosis doesn't align with provided symptoms. Manual review required."
            result["_metadata"]["blocked"] = True
            
            self._audit_log(request_id, "BLOCKED", patient_data, result)
            return result
        
        # Check 2c: Model drift detection
        if self._check_model_drift():
            diagnosis["safety_alerts"] = diagnosis.get("safety_alerts", [])
            diagnosis["safety_alerts"].append("MODEL_DRIFT_WARNING")
            logger.warning("Model drift detected - interpret with caution")
        
        # Step 3: Build successful result
        result["status"] = DiagnosisStatus.FALLBACK if diagnosis.get("_fallback") else DiagnosisStatus.SUCCESS
        result["diagnosis"] = diagnosis.get("primary_diagnosis", {})
        result["differential_diagnoses"] = diagnosis.get("differential_diagnoses", [])
        result["confidence"] = confidence
        result["safety_alerts"] = diagnosis.get("safety_alerts", [])
        result["needs_escalation"] = diagnosis.get("needs_escalation", False)
        result["explanation"] = diagnosis.get("explanation", "")
        result["_metadata"]["processing_time_ms"] = int((datetime.now() - start_time).total_seconds() * 1000)
        
        self._audit_log(request_id, result["status"].value, patient_data, result)
        
        return result
    
    def _rule_based_diagnose(self, patient_data: Dict) -> Dict:
        """
        Fallback rule-based diagnosis using symptom patterns.
        
        Used when primary LLM fails.
        """
        symptoms_text = patient_data.get("symptoms", "").lower()
        extracted = set()
        
        # Simple keyword extraction
        symptom_keywords = [
            "fever", "cough", "shortness of breath", "chest pain",
            "headache", "nausea", "vomiting", "fatigue", "weakness",
            "abdominal pain", "diarrhea", "sweating", "neck stiffness",
            "body aches", "sore throat", "runny nose"
        ]
        
        for keyword in symptom_keywords:
            if keyword in symptoms_text:
                extracted.add(keyword)
        
        # Match against rules
        best_match = None
        best_match_count = 0
        
        for symptom_set, diagnosis in self.symptom_disease_rules.items():
            match_count = len(symptom_set & extracted)
            if match_count >= 2 and match_count > best_match_count:
                best_match = diagnosis
                best_match_count = match_count
        
        if best_match:
            return {
                "primary_diagnosis": {
                    "disease": best_match["disease"],
                    "confidence": best_match["confidence"],
                    "severity": best_match["severity"]
                },
                "differential_diagnoses": [],
                "confidence": best_match["confidence"],
                "needs_escalation": best_match.get("escalation", False),
                "safety_alerts": ["RULE_BASED_FALLBACK"],
                "explanation": f"Rule-based diagnosis based on symptoms: {', '.join(extracted)}"
            }
        
        # No match - require human review
        return {
            "primary_diagnosis": {},
            "differential_diagnoses": [],
            "confidence": 0.0,
            "needs_escalation": True,
            "safety_alerts": ["NO_MATCH_FOUND"],
            "explanation": "Unable to determine diagnosis from available symptoms. Human review required."
        }
    
    def _sanity_check(self, diagnosis: Dict, patient_data: Dict) -> bool:
        """
        Verify diagnosis makes logical sense given patient data.
        
        Returns:
            True if diagnosis passes sanity check
        """
        primary = diagnosis.get("primary_diagnosis", {})
        disease_name = primary.get("disease", "").lower()
        symptoms_text = patient_data.get("symptoms", "").lower()
        
        # Check if any required symptoms are present
        for condition, required in self.required_symptoms.items():
            if condition in disease_name:
                # At least one required symptom must be present
                if not any(req in symptoms_text for req in required):
                    logger.warning(f"Sanity check failed: {disease_name} requires {required}")
                    return False
        
        return True
    
    def _check_model_drift(self) -> bool:
        """
        Check if model accuracy has degraded significantly.
        
        Returns:
            True if drift detected
        """
        baseline_acc = self.daily_metrics.get("baseline_accuracy", 0.90)
        current_acc = self.daily_metrics.get("current_accuracy", 0.90)
        
        if baseline_acc - current_acc > 0.05:  # >5% drop
            return True
        
        return False
    
    def _log_alert(self, alert_type: str, message: str):
        """Log safety alert"""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "type": alert_type,
            "message": message
        }
        self.alerts.append(alert)
        logger.warning(f"Safety Alert [{alert_type}]: {message}")
    
    def _audit_log(self, request_id: str, action: str, patient_data: Dict, result: Dict):
        """Log for audit trail"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "action": action,
            "patient_id": patient_data.get("patient_id", "unknown"),
            "diagnosis": result.get("diagnosis", {}).get("disease", "none"),
            "confidence": result.get("confidence", 0.0),
            "blocked": result.get("_metadata", {}).get("blocked", False),
            "fallback_used": result.get("_metadata", {}).get("fallback_used", False)
        }
        self.audit_log.append(entry)
        logger.info(f"Audit: {action} for {request_id}")
    
    def update_metrics(self, accuracy: float):
        """Update current accuracy metrics for drift detection"""
        if "baseline_accuracy" not in self.daily_metrics:
            self.daily_metrics["baseline_accuracy"] = accuracy
        self.daily_metrics["current_accuracy"] = accuracy
    
    def get_alerts(self) -> List[Dict]:
        """Get recent alerts"""
        return self.alerts[-100:]  # Last 100 alerts
    
    def get_audit_log(self) -> List[Dict]:
        """Get audit log entries"""
        return self.audit_log[-1000:]  # Last 1000 entries


# Singleton instance
_safety_monitor = None


def get_safety_monitor() -> SafetyMonitor:
    """Get singleton safety monitor instance"""
    global _safety_monitor
    if _safety_monitor is None:
        _safety_monitor = SafetyMonitor()
    return _safety_monitor
