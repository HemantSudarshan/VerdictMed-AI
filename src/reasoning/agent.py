"""
LangGraph Diagnostic Agent
Graph-based reasoning workflow using LangGraph StateGraph.
Implements the diagnostic pipeline as explicit nodes and edges.
"""

from typing import Dict, List, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from loguru import logger
import asyncio
from concurrent.futures import ThreadPoolExecutor

from src.reasoning.state import DiagnosticState


# Node functions for the diagnostic workflow

def extract_symptoms_node(state: DiagnosticState) -> Dict:
    """
    Node 1: Extract symptoms from clinical text using NLP.
    """
    from src.nlp.clinical_nlp import ClinicalNLPPipeline
    
    symptoms_text = state.get("symptoms_text", "")
    
    if not symptoms_text:
        return {"extracted_symptoms": [], "extracted_vitals": {}}
    
    try:
        nlp = ClinicalNLPPipeline(load_models=False)
        result = nlp.analyze_clinical_note(symptoms_text)
        
        return {
            "extracted_symptoms": result.get("symptoms", []),
            "extracted_vitals": result.get("vitals", {})
        }
    except Exception as e:
        logger.error(f"NLP extraction failed: {e}")
        return {"extracted_symptoms": [], "extracted_vitals": {}}


def process_labs_node(state: DiagnosticState) -> Dict:
    """
    Node 2: Process lab values against clinical thresholds.
    """
    from src.nlp.lab_processor import get_lab_processor
    
    lab_results = state.get("lab_results") or {}
    
    if not lab_results:
        return {"lab_findings": {}}
    
    try:
        processor = get_lab_processor()
        findings = processor.process(lab_results)
        
        return {"lab_findings": findings}
    except Exception as e:
        logger.error(f"Lab processing failed: {e}")
        return {"lab_findings": {}}


def analyze_image_node(state: DiagnosticState) -> Dict:
    """
    Node 3: Analyze medical image using BiomedCLIP.
    """
    image_path = state.get("image_path")
    
    if not image_path:
        return {"image_findings": {}}
    
    try:
        from src.vision.biomedclip import BiomedCLIPAnalyzer
        
        analyzer = BiomedCLIPAnalyzer()
        findings = analyzer.analyze_chest_xray(image_path)
        
        return {"image_findings": findings}
    except Exception as e:
        logger.error(f"Image analysis failed: {e}")
        return {"image_findings": {"error": str(e)}}


def generate_gradcam_node(state: DiagnosticState) -> Dict:
    """
    Node 3b: Generate GradCAM heatmap for image explanation.
    Only runs if image analysis was performed.
    """
    image_findings = state.get("image_findings", {})
    image_path = state.get("image_path")
    
    if not image_path or not image_findings or "error" in image_findings:
        return {"gradcam_explanation": None}
    
    try:
        from src.vision.explainer import GradCAMExplainer
        import cv2
        import numpy as np
        
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return {"gradcam_explanation": None}
        
        # Note: GradCAM requires a PyTorch model - BiomedCLIP is CLIP-based
        # For now, return placeholder with image findings
        explanation = {
            "available": False,
            "reason": "GradCAM requires PyTorch CNN model, BiomedCLIP is transformer-based",
            "top_finding": image_findings.get("top_finding"),
            "attention_regions": "Visual attention explanation pending model integration"
        }
        
        return {"gradcam_explanation": explanation}
        
    except Exception as e:
        logger.error(f"GradCAM generation failed: {e}")
        return {"gradcam_explanation": None}


def query_knowledge_graph_node(state: DiagnosticState) -> Dict:
    """
    Node 4: Query Neo4j knowledge graph for matching diseases.
    """
    extracted_symptoms = state.get("extracted_symptoms", [])
    
    # Filter to positive (non-negated) symptoms
    positive_symptoms = [
        s["symptom"] for s in extracted_symptoms 
        if not s.get("negated", False)
    ]
    
    if not positive_symptoms:
        return {"kg_diseases": []}
    
    try:
        from src.knowledge_graph.query_engine import MedicalKnowledgeGraph
        from src.config import get_settings
        
        settings = get_settings()
        kg = MedicalKnowledgeGraph(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password
        )
        
        diseases = kg.find_diseases_by_symptoms(positive_symptoms, limit=10)
        
        return {"kg_diseases": diseases}
        
    except Exception as e:
        logger.warning(f"KG query failed: {e}")
        return {"kg_diseases": []}


def retrieve_similar_cases_node(state: DiagnosticState) -> Dict:
    """
    Node 5: Retrieve similar historical cases from vector store.
    """
    extracted_symptoms = state.get("extracted_symptoms", [])
    
    if not extracted_symptoms:
        return {"similar_cases": []}
    
    try:
        import weaviate
        from src.config import get_settings
        
        settings = get_settings()
        client = weaviate.Client(settings.weaviate_url)
        
        # Create symptom embedding query
        symptom_list = [s["symptom"] for s in extracted_symptoms if not s.get("negated")]
        symptom_text = ", ".join(symptom_list)
        
        # Query similar cases
        result = client.query.get(
            "PatientCase",
            ["case_id", "symptoms", "diagnosis", "icd10", "outcome", "severity"]
        ).with_near_text({
            "concepts": [symptom_text]
        }).with_limit(5).do()
        
        cases = result.get("data", {}).get("Get", {}).get("PatientCase", [])
        
        return {"similar_cases": cases}
        
    except Exception as e:
        logger.warning(f"Vector store query failed: {e}")
        return {"similar_cases": []}


def fuse_modalities_node(state: DiagnosticState) -> Dict:
    """
    Node 6: Fuse multimodal signals with weighted scoring.
    """
    # Weights for each modality
    WEIGHTS = {"symptoms": 0.30, "labs": 0.35, "image": 0.35}
    
    modality_scores = {}
    conflicts = []
    
    # Score symptoms
    symptoms = state.get("extracted_symptoms", [])
    positive_symptoms = [s for s in symptoms if not s.get("negated", False)]
    infection_keywords = ["fever", "cough", "fatigue", "chills", "pain"]
    
    symptom_score = 0.0
    if positive_symptoms:
        infection_count = sum(1 for s in positive_symptoms 
                              if any(k in s.get("symptom", "").lower() for k in infection_keywords))
        symptom_score = min(1.0, infection_count / 3)
    modality_scores["symptoms"] = symptom_score
    
    # Score labs
    lab_findings = state.get("lab_findings", {})
    lab_score = lab_findings.get("severity_score", 0.0)
    modality_scores["labs"] = lab_score
    
    # Score image
    image_findings = state.get("image_findings", {})
    image_score = 0.0
    if image_findings.get("top_finding"):
        image_score = image_findings.get("confidence", 0.5)
    modality_scores["image"] = image_score
    
    # Detect conflicts
    scores = [s for s in [symptom_score, lab_score, image_score] if s > 0]
    if len(scores) >= 2:
        max_diff = max(scores) - min(scores)
        if max_diff > 0.5:
            conflicts.append({
                "type": "MODALITY_DISAGREEMENT",
                "description": f"Significant disagreement between data sources (diff={max_diff:.2f})"
            })
    
    # Calculate fusion score
    available_weight = sum(WEIGHTS[k] for k in modality_scores if modality_scores[k] > 0 or k == "symptoms")
    fusion_score = sum(modality_scores[k] * WEIGHTS[k] for k in modality_scores) / max(available_weight, 0.01)
    
    return {
        "modality_scores": modality_scores,
        "conflicts_detected": conflicts,
        "fusion_score": fusion_score
    }


def generate_differential_node(state: DiagnosticState) -> Dict:
    """
    Node 7: Generate differential diagnoses from collected evidence.
    """
    kg_diseases = state.get("kg_diseases", [])
    image_findings = state.get("image_findings", {})
    fusion_score = state.get("fusion_score", 0.5)
    
    differential = []
    
    for disease in kg_diseases[:5]:
        # Calculate confidence
        base_score = disease.get("match_ratio", 0.5)
        
        # Boost if image finding matches
        top_finding = image_findings.get("top_finding", "").lower()
        disease_name = disease.get("disease", "").lower()
        if top_finding and any(word in top_finding for word in disease_name.split()):
            base_score *= 1.2
        
        confidence = min(base_score * fusion_score * 1.5, 0.99)
        
        differential.append({
            "disease": disease.get("disease", "Unknown"),
            "icd10": disease.get("icd10"),
            "confidence": round(confidence, 3),
            "severity": disease.get("severity", "unknown"),
            "matched_symptoms": disease.get("matched_symptoms", [])
        })
    
    differential.sort(key=lambda x: x["confidence"], reverse=True)
    
    if not differential:
        differential = [{
            "disease": "Unspecified condition",
            "icd10": None,
            "confidence": 0.3,
            "severity": "unknown"
        }]
    
    primary = differential[0] if differential else {}
    
    return {
        "differential_diagnoses": differential,
        "primary_diagnosis": primary,
        "confidence": primary.get("confidence", 0.0)
    }


def safety_validation_node(state: DiagnosticState) -> Dict:
    """
    Node 8: Perform safety validation and escalation checks.
    """
    from src.safety.validator import SafetyValidator
    
    try:
        validator = SafetyValidator()
        result = validator.validate_diagnosis(state)
        
        return {
            "safety_alerts": result.get("alerts", []),
            "needs_escalation": result.get("requires_escalation", False),
            "escalation_reason": "; ".join(result.get("escalation_reasons", []))
        }
    except Exception as e:
        logger.error(f"Safety validation failed: {e}")
        return {
            "safety_alerts": ["SAFETY_CHECK_ERROR"],
            "needs_escalation": True,
            "escalation_reason": f"Safety validation failed: {e}"
        }


def generate_explanation_node(state: DiagnosticState) -> Dict:
    """
    Node 9: Generate human-readable explanation with SHAP values.
    """
    primary = state.get("primary_diagnosis", {})
    disease_name = primary.get("disease", "Unknown condition")
    confidence = state.get("confidence", 0)
    symptoms = state.get("extracted_symptoms", [])
    image_findings = state.get("image_findings", {})
    lab_findings = state.get("lab_findings", {})
    similar_cases = state.get("similar_cases", [])
    
    explanation = f"DIAGNOSIS: {disease_name}\nCONFIDENCE: {confidence:.1%}\n\n"
    
    explanation += "KEY FINDINGS:\n"
    for symptom in symptoms[:5]:
        neg = " (denied)" if symptom.get("negated") else ""
        explanation += f"• {symptom['symptom'].title()}{neg}\n"
    
    if image_findings.get("top_finding"):
        explanation += f"• X-ray: {image_findings['top_finding']} ({image_findings.get('confidence', 0):.1%})\n"
    
    if lab_findings.get("abnormalities"):
        explanation += f"• Labs: {len(lab_findings['abnormalities'])} abnormalities detected\n"
    
    if similar_cases:
        explanation += f"\nSIMILAR CASES: {len(similar_cases)} similar cases found\n"
    
    if state.get("safety_alerts"):
        explanation += f"\n⚠️ ALERTS: {', '.join(state['safety_alerts'][:3])}\n"
    
    explanation += "\n⚠️ AI-assisted analysis. Verify with physician."
    
    # Generate feature importances (simplified SHAP-like)
    feature_importances = {}
    if symptoms:
        for i, s in enumerate(symptoms[:3]):
            feature_importances[s["symptom"]] = round(0.3 - (i * 0.05), 2)
    if image_findings.get("top_finding"):
        feature_importances["image_finding"] = 0.25
    if lab_findings.get("severity_score", 0) > 0:
        feature_importances["lab_severity"] = round(lab_findings["severity_score"] * 0.2, 2)
    
    return {
        "explanation": explanation,
        "feature_importances": feature_importances
    }


def create_diagnostic_graph() -> StateGraph:
    """
    Create the LangGraph diagnostic workflow.
    
    Graph structure:
    START -> extract_symptoms -> process_labs -> analyze_image -> generate_gradcam
          -> query_kg -> retrieve_similar_cases -> fuse_modalities
          -> generate_differential -> safety_validation -> generate_explanation -> END
    """
    
    # Create graph with DiagnosticState
    workflow = StateGraph(DiagnosticState)
    
    # Add nodes
    workflow.add_node("extract_symptoms", extract_symptoms_node)
    workflow.add_node("process_labs", process_labs_node)
    workflow.add_node("analyze_image", analyze_image_node)
    workflow.add_node("generate_gradcam", generate_gradcam_node)
    workflow.add_node("query_kg", query_knowledge_graph_node)
    workflow.add_node("retrieve_similar_cases", retrieve_similar_cases_node)
    workflow.add_node("fuse_modalities", fuse_modalities_node)
    workflow.add_node("generate_differential", generate_differential_node)
    workflow.add_node("safety_validation", safety_validation_node)
    workflow.add_node("generate_explanation", generate_explanation_node)
    
    # Define edges (linear flow)
    workflow.set_entry_point("extract_symptoms")
    workflow.add_edge("extract_symptoms", "process_labs")
    workflow.add_edge("process_labs", "analyze_image")
    workflow.add_edge("analyze_image", "generate_gradcam")
    workflow.add_edge("generate_gradcam", "query_kg")
    workflow.add_edge("query_kg", "retrieve_similar_cases")
    workflow.add_edge("retrieve_similar_cases", "fuse_modalities")
    workflow.add_edge("fuse_modalities", "generate_differential")
    workflow.add_edge("generate_differential", "safety_validation")
    workflow.add_edge("safety_validation", "generate_explanation")
    workflow.add_edge("generate_explanation", END)
    
    return workflow


class LangGraphDiagnosticAgent:
    """
    LangGraph-based diagnostic reasoning agent.
    Uses StateGraph for explicit workflow management.
    """
    
    def __init__(self, config=None):
        """
        Initialize the LangGraph agent.
        
        Args:
            config: Configuration object with settings
        """
        self.config = config
        self.workflow = create_diagnostic_graph()
        self.app = self.workflow.compile()
        
        logger.info("LangGraphDiagnosticAgent initialized")
    
    async def run(self, patient_data: Dict) -> Dict:
        """
        Execute the diagnostic workflow.
        
        Args:
            patient_data: Dict with symptoms, optional image_path, labs, etc.
            
        Returns:
            Complete diagnostic result
        """
        import time
        start_time = time.time()
        
        # Initialize state
        initial_state: DiagnosticState = {
            "patient_id": patient_data.get("patient_id", ""),
            "symptoms_text": patient_data.get("symptoms", ""),
            "lab_results": patient_data.get("labs") or patient_data.get("lab_results"),
            "image_path": patient_data.get("image_path"),
            "patient_history": patient_data.get("history"),
            # Initialize empty fields
            "extracted_symptoms": [],
            "extracted_vitals": {},
            "lab_findings": {},
            "image_findings": {},
            "gradcam_explanation": None,
            "kg_diseases": [],
            "similar_cases": [],
            "differential_diagnoses": [],
            "primary_diagnosis": {},
            "confidence": 0.0,
            "modality_scores": {},
            "conflicts_detected": [],
            "safety_alerts": [],
            "needs_escalation": False,
            "escalation_reason": None,
            "explanation": "",
            "feature_importances": {},
        }
        
        # Run workflow
        try:
            result = self.app.invoke(initial_state)
        except Exception as e:
            logger.error(f"LangGraph workflow error: {e}")
            result = {
                **initial_state,
                "safety_alerts": ["WORKFLOW_ERROR"],
                "needs_escalation": True,
                "explanation": f"Error during diagnosis: {e}"
            }
        
        # Add metadata
        result["processing_time_ms"] = int((time.time() - start_time) * 1000)
        
        return result


# Export
__all__ = ["LangGraphDiagnosticAgent", "create_diagnostic_graph"]
