"""
LangGraph State Definitions for Diagnostic Workflow

Defines TypedDict state for diagnostic reasoning graph.
Compatible with LangGraph for future migration from SimpleDiagnosticAgent.

NOTE: The current SimpleDiagnosticAgent uses standard dicts for state management
to avoid LangGraph dependency. These type definitions provide type safety and
serve as a migration path to full LangGraph implementation when needed.
"""

from typing import TypedDict, List, Dict, Optional, Tuple
from enum import Enum


class RiskLevel(str, Enum):
    """Patient risk classification levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DiagnosticState(TypedDict, total=False):
    """
    State object for diagnostic reasoning workflow.
    
    Used in multi-step diagnostic agent to track and pass data
    between reasoning nodes (NLP → Vision → KG → Safety → Explanation).
    
    Compatible with LangGraph's StateGraph when full workflow is implemented.
    """
    
    # Input data
    patient_id: str
    symptoms_text: str
    lab_results: Optional[Dict]
    image_path: Optional[str]
    patient_history: Optional[Dict]
    
    # Extracted data from NLP
    extracted_symptoms: List[Dict]  # [{symptom: str, negated: bool, confidence: float}]
    extracted_vitals: Dict  # {temperature: float, bp_systolic: int, ...}
    
    # Vision analysis results
    image_findings: Dict  # {top_finding: str, confidence: float, abnormalities: [...]}
    
    # Knowledge graph reasoning results
    kg_diseases: List[Dict]  # [{disease: str, icd10: str, match_ratio: float, ...}]
    similar_cases: List[Dict]  # Similar historical cases from vector DB
    
    # Diagnosis output
    differential_diagnoses: List[Dict]  # Ranked list of possible diagnoses
    primary_diagnosis: Dict  # Top diagnosis with confidence and evidence
    confidence: float  # Overall confidence in primary diagnosis [0.0, 1.0]
    confidence_interval: Tuple[float, float]  # Bayesian confidence bounds
    
    # Safety validation
    conflicts_detected: List[str]  # Signal conflicts between modalities
    safety_alerts: List[str]  # Safety warnings triggered
    needs_escalation: bool  # Whether case requires human review
    escalation_reason: Optional[str]  # Why escalation is needed
    risk_level: RiskLevel  # Patient risk classification
    
    # Explainability
    explanation: str  # Human-readable diagnostic explanation
    feature_importances: Dict  # Which features drove the decision
    shap_values: Optional[Dict]  # SHAP explainability values
    
    # Metadata
    processing_time_ms: int  # Total processing time
    model_versions: Dict  # Versions of models used
    cache_hit: bool  # Whether result came from cache
    timestamp: str  # ISO timestamp of diagnosis


# Type aliases for cleaner type hints
SymptomDict = TypedDict('SymptomDict', {
    'symptom': str,
    'negated': bool,
    'confidence': float,
    'severity': Optional[str]
})

DiseaseDict = TypedDict('DiseaseDict', {
    'disease': str,
    'icd10': str,
    'confidence': float,
    'severity': str,
    'category': str,
    'matched_symptoms': List[str],
    'supporting_evidence': List[str],
    'contradicting_evidence': List[str]
})


# Graph node return types
class NodeUpdate(TypedDict, total=False):
    """
    Partial state update returned by graph nodes.
    LangGraph merges this with existing state.
    """
    extracted_symptoms: List[SymptomDict]
    extracted_vitals: Dict
    image_findings: Dict
    kg_diseases: List[DiseaseDict]
    differential_diagnoses: List[DiseaseDict]
    primary_diagnosis: DiseaseDict
    confidence: float
    safety_alerts: List[str]
    needs_escalation: bool
    explanation: str


# Example usage with SimpleDiagnosticAgent
def example_state_creation() -> DiagnosticState:
    """
    Example of creating initial state for diagnostic workflow.
    
    SimpleDiagnosticAgent currently creates this as a regular dict,
    but using this TypedDict provides type checking.
    """
    initial_state: DiagnosticState = {
        "patient_id": "P12345",
        "symptoms_text": "Patient has fever and cough for 3 days",
        "lab_results": {"wbc": 12.5, "crp": 45},
        "image_path": "/data/xrays/p12345.png",
        "patient_history": {"age": 45, "sex": "M"},
        
        # Initialize empty collections
        "extracted_symptoms": [],
        "extracted_vitals": {},
        "image_findings": {},
        "kg_diseases": [],
        "similar_cases": [],
        "differential_diagnoses": [],
        "primary_diagnosis": {},
        "confidence": 0.0,
        "confidence_interval": (0.0, 0.0),
        
        # Safety defaults
        "conflicts_detected": [],
        "safety_alerts": [],
        "needs_escalation": False,
        "escalation_reason": None,
        "risk_level": RiskLevel.LOW,
        
        # Explanation defaults
        "explanation": "",
        "feature_importances": {},
        "shap_values": None,
        
        # Metadata
        "processing_time_ms": 0,
        "model_versions": {},
        "cache_hit": False,
        "timestamp": ""
    }
    
    return initial_state


__all__ = [
    "DiagnosticState",
    "RiskLevel",
    "SymptomDict",
    "DiseaseDict",
    "NodeUpdate",
    "example_state_creation"
]
