# CDSS PRD Part 2: Reasoning, Safety & Deployment
## AI-Powered Clinical Decision Support System - LLM Execution Guide

**Continues from Part 1** | See CDSS_PRD_Part1.md for Stages 1-4

---

# STAGE 5: KNOWLEDGE GRAPH REASONING (Week 5)

## Goal
Query the medical knowledge graph for symptom-to-disease reasoning.

## Step 5.1: Knowledge Graph Query Engine
```python
# src/knowledge_graph/query_engine.py
from py2neo import Graph
from typing import List, Dict
from loguru import logger

class MedicalKnowledgeGraph:
    """Query medical knowledge graph for diagnostic reasoning"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.graph = Graph(uri, auth=(user, password))
    
    def find_diseases_by_symptoms(self, symptoms: List[str], limit: int = 10) -> List[Dict]:
        """
        Find diseases that match given symptoms.
        
        Returns ranked list of diseases with match scores.
        """
        query = """
        UNWIND $symptoms AS symptom_name
        MATCH (d:Disease)-[r:PRESENTS_WITH]->(s:Symptom)
        WHERE toLower(s.name) = toLower(symptom_name)
        WITH d, COUNT(DISTINCT s) as symptom_matches, COLLECT(s.name) as matched_symptoms
        MATCH (d)-[:PRESENTS_WITH]->(all_symptoms:Symptom)
        WITH d, symptom_matches, matched_symptoms, COUNT(all_symptoms) as total_symptoms
        RETURN d.name as disease,
               d.icd10 as icd10,
               d.severity as severity,
               symptom_matches,
               total_symptoms,
               toFloat(symptom_matches) / toFloat(total_symptoms) as match_ratio,
               matched_symptoms
        ORDER BY symptom_matches DESC, match_ratio DESC
        LIMIT $limit
        """
        
        results = self.graph.run(query, symptoms=symptoms, limit=limit).data()
        return results
    
    def get_disease_details(self, icd10: str) -> Dict:
        """Get full disease information including tests and treatments"""
        query = """
        MATCH (d:Disease {icd10: $icd10})
        OPTIONAL MATCH (d)-[:PRESENTS_WITH]->(s:Symptom)
        OPTIONAL MATCH (d)-[:DIAGNOSED_BY]->(t:Test)
        OPTIONAL MATCH (d)-[:TREATED_BY]->(m:Medication)
        RETURN d.name as name,
               d.icd10 as icd10,
               d.category as category,
               d.severity as severity,
               COLLECT(DISTINCT s.name) as symptoms,
               COLLECT(DISTINCT t.name) as diagnostic_tests,
               COLLECT(DISTINCT m.name) as treatments
        """
        result = self.graph.run(query, icd10=icd10).data()
        return result[0] if result else None
    
    def find_differential_diagnoses(self, symptoms: List[str], exclude: List[str] = None) -> List[Dict]:
        """Find alternative diagnoses for differential"""
        exclude = exclude or []
        
        query = """
        UNWIND $symptoms AS symptom_name
        MATCH (d:Disease)-[:PRESENTS_WITH]->(s:Symptom)
        WHERE toLower(s.name) = toLower(symptom_name)
        AND NOT d.icd10 IN $exclude
        WITH d, COUNT(DISTINCT s) as matches
        RETURN d.name as disease,
               d.icd10 as icd10,
               d.severity as severity,
               matches
        ORDER BY matches DESC
        LIMIT 5
        """
        return self.graph.run(query, symptoms=symptoms, exclude=exclude).data()
    
    def check_contraindications(self, medications: List[str], patient_conditions: List[str]) -> List[Dict]:
        """Check for drug contraindications"""
        query = """
        UNWIND $medications AS med_name
        MATCH (m:Medication {name: med_name})-[:CONTRAINDICATED_FOR]->(c:Condition)
        WHERE c.name IN $conditions
        RETURN m.name as medication,
               c.name as contraindicated_condition,
               "DO NOT USE" as warning
        """
        return self.graph.run(query, medications=medications, conditions=patient_conditions).data()
```

---

# STAGE 6: LANGGRAPH DIAGNOSTIC AGENT (Weeks 6-7)

## Goal
Build multi-step reasoning agent that orchestrates all modules.

## Step 6.1: Agent State Definition
```python
# src/reasoning/state.py
from typing import TypedDict, List, Dict, Optional
from enum import Enum

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class DiagnosticState(TypedDict):
    # Input data
    patient_id: str
    symptoms_text: str
    lab_results: Optional[Dict]
    image_path: Optional[str]
    patient_history: Optional[Dict]
    
    # Extracted data
    extracted_symptoms: List[Dict]
    extracted_vitals: Dict
    image_findings: Dict
    
    # Reasoning results
    kg_diseases: List[Dict]
    similar_cases: List[Dict]
    
    # Diagnosis output
    differential_diagnoses: List[Dict]
    primary_diagnosis: Dict
    confidence: float
    confidence_interval: tuple
    
    # Safety
    conflicts_detected: List[str]
    safety_alerts: List[str]
    needs_escalation: bool
    escalation_reason: Optional[str]
    
    # Explainability
    explanation: str
    feature_importances: Dict
    
    # Metadata
    processing_time_ms: int
    model_versions: Dict
```

## Step 6.2: LangGraph Workflow
```python
# src/reasoning/agent.py
from langgraph.graph import StateGraph, END
from typing import Literal
from loguru import logger
import time

from .state import DiagnosticState, RiskLevel
from src.vision.biomedclip import BiomedCLIPAnalyzer
from src.nlp.clinical_nlp import ClinicalNLPPipeline
from src.knowledge_graph.query_engine import MedicalKnowledgeGraph
from src.safety.validator import SafetyValidator

class DiagnosticAgent:
    """Multi-step diagnostic reasoning agent"""
    
    def __init__(self, config):
        self.vision = BiomedCLIPAnalyzer()
        self.nlp = ClinicalNLPPipeline()
        self.kg = MedicalKnowledgeGraph(
            config.neo4j_uri, config.neo4j_user, config.neo4j_password
        )
        self.safety = SafetyValidator(config)
        self.config = config
        
        # Build workflow
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(DiagnosticState)
        
        # Add nodes
        workflow.add_node("extract_symptoms", self.extract_symptoms)
        workflow.add_node("analyze_image", self.analyze_image)
        workflow.add_node("query_knowledge_graph", self.query_knowledge_graph)
        workflow.add_node("retrieve_similar_cases", self.retrieve_similar_cases)
        workflow.add_node("generate_differential", self.generate_differential)
        workflow.add_node("safety_check", self.safety_check)
        workflow.add_node("generate_explanation", self.generate_explanation)
        workflow.add_node("escalate", self.escalate_to_human)
        
        # Add edges
        workflow.add_edge("extract_symptoms", "analyze_image")
        workflow.add_edge("analyze_image", "query_knowledge_graph")
        workflow.add_edge("query_knowledge_graph", "retrieve_similar_cases")
        workflow.add_edge("retrieve_similar_cases", "generate_differential")
        workflow.add_edge("generate_differential", "safety_check")
        
        # Conditional edge after safety check
        workflow.add_conditional_edges(
            "safety_check",
            self.should_escalate,
            {
                "escalate": "escalate",
                "continue": "generate_explanation"
            }
        )
        
        workflow.add_edge("escalate", END)
        workflow.add_edge("generate_explanation", END)
        
        workflow.set_entry_point("extract_symptoms")
        
        return workflow.compile()
    
    def extract_symptoms(self, state: DiagnosticState) -> dict:
        """Extract symptoms from clinical text"""
        symptoms = self.nlp.extract_symptoms(state["symptoms_text"])
        vitals = self.nlp.extract_vitals(state["symptoms_text"])
        
        # Filter out negated symptoms
        positive_symptoms = [s for s in symptoms if not s["negated"]]
        
        return {
            "extracted_symptoms": positive_symptoms,
            "extracted_vitals": vitals
        }
    
    def analyze_image(self, state: DiagnosticState) -> dict:
        """Analyze medical image if provided"""
        if not state.get("image_path"):
            return {"image_findings": {"status": "no_image_provided"}}
        
        findings = self.vision.analyze_chest_xray(state["image_path"])
        return {"image_findings": findings}
    
    def query_knowledge_graph(self, state: DiagnosticState) -> dict:
        """Query KG for possible diseases"""
        symptom_names = [s["symptom"] for s in state["extracted_symptoms"]]
        diseases = self.kg.find_diseases_by_symptoms(symptom_names)
        return {"kg_diseases": diseases}
    
    def retrieve_similar_cases(self, state: DiagnosticState) -> dict:
        """Find similar past cases from vector store"""
        # Implementation uses Weaviate similarity search
        # Simplified for brevity
        return {"similar_cases": []}
    
    def generate_differential(self, state: DiagnosticState) -> dict:
        """Generate differential diagnosis"""
        kg_diseases = state["kg_diseases"]
        image_findings = state["image_findings"]
        
        # Combine signals
        differential = []
        
        for disease in kg_diseases[:5]:
            confidence = self._calculate_confidence(disease, image_findings, state)
            
            differential.append({
                "disease": disease["disease"],
                "icd10": disease["icd10"],
                "confidence": confidence,
                "severity": disease.get("severity", "unknown"),
                "matched_symptoms": disease.get("matched_symptoms", []),
                "supporting_evidence": [],
                "contradicting_evidence": []
            })
        
        # Sort by confidence
        differential.sort(key=lambda x: x["confidence"], reverse=True)
        
        primary = differential[0] if differential else None
        
        return {
            "differential_diagnoses": differential,
            "primary_diagnosis": primary,
            "confidence": primary["confidence"] if primary else 0.0
        }
    
    def _calculate_confidence(self, disease: Dict, image_findings: Dict, state: Dict) -> float:
        """Calculate confidence combining multiple signals"""
        base_score = disease.get("match_ratio", 0.5)
        
        # Boost if image supports
        if image_findings.get("top_finding"):
            if disease["disease"].lower() in image_findings["top_finding"].lower():
                base_score *= 1.2
        
        # Adjust for data completeness
        data_completeness = self._calculate_data_completeness(state)
        confidence = base_score * data_completeness
        
        return min(confidence, 0.99)  # Cap at 99%
    
    def _calculate_data_completeness(self, state: Dict) -> float:
        """Calculate how complete the input data is"""
        score = 0.0
        if state.get("extracted_symptoms"):
            score += 0.4
        if state.get("image_findings", {}).get("status") != "no_image_provided":
            score += 0.35
        if state.get("lab_results"):
            score += 0.25
        return max(score, 0.4)  # Minimum 40%
    
    def safety_check(self, state: DiagnosticState) -> dict:
        """Run safety validations"""
        alerts = []
        conflicts = []
        needs_escalation = False
        escalation_reason = None
        
        # Check confidence threshold
        if state["confidence"] < self.config.min_confidence_threshold:
            needs_escalation = True
            escalation_reason = f"Low confidence: {state['confidence']:.1%}"
            alerts.append("LOW_CONFIDENCE")
        
        # Check for critical conditions
        primary = state.get("primary_diagnosis", {})
        if primary.get("severity") == "critical":
            alerts.append("CRITICAL_CONDITION_DETECTED")
        
        # Check for signal conflicts
        image_finding = state.get("image_findings", {}).get("top_finding", "")
        primary_disease = primary.get("disease", "")
        
        if image_finding and primary_disease:
            if not self._signals_align(image_finding, primary_disease):
                conflicts.append(f"Image suggests '{image_finding}' but symptoms suggest '{primary_disease}'")
                alerts.append("SIGNAL_CONFLICT")
        
        return {
            "safety_alerts": alerts,
            "conflicts_detected": conflicts,
            "needs_escalation": needs_escalation,
            "escalation_reason": escalation_reason
        }
    
    def _signals_align(self, image_finding: str, disease: str) -> bool:
        """Check if image and symptom findings align"""
        # Simplified alignment check
        disease_lower = disease.lower()
        finding_lower = image_finding.lower()
        
        alignment_map = {
            "pneumonia": ["pneumonia", "consolidation", "infiltrate"],
            "tuberculosis": ["tuberculosis", "cavity", "infiltrate"],
            "heart": ["cardiomegaly", "enlarged heart"]
        }
        
        for key, values in alignment_map.items():
            if key in disease_lower:
                if any(v in finding_lower for v in values):
                    return True
                if "normal" in finding_lower:
                    return False
        
        return True  # Default to aligned if unknown
    
    def should_escalate(self, state: DiagnosticState) -> Literal["escalate", "continue"]:
        """Decide if case needs human escalation"""
        if state.get("needs_escalation"):
            return "escalate"
        return "continue"
    
    def escalate_to_human(self, state: DiagnosticState) -> dict:
        """Prepare escalation to human doctor"""
        return {
            "explanation": f"""
⚠️ ESCALATION REQUIRED

Reason: {state.get('escalation_reason', 'Safety check triggered')}

Available Data:
- Symptoms: {[s['symptom'] for s in state.get('extracted_symptoms', [])]}
- Image findings: {state.get('image_findings', {}).get('top_finding', 'N/A')}
- Possible diagnoses: {[d['disease'] for d in state.get('differential_diagnoses', [])[:3]]}

This case requires specialist review before proceeding.
"""
        }
    
    def generate_explanation(self, state: DiagnosticState) -> dict:
        """Generate human-readable explanation"""
        primary = state.get("primary_diagnosis", {})
        differential = state.get("differential_diagnoses", [])
        
        explanation = f"""
DIAGNOSIS: {primary.get('disease', 'Unknown')} 
CONFIDENCE: {state.get('confidence', 0):.1%}

KEY FINDINGS:
"""
        # Add symptom evidence
        for symptom in state.get("extracted_symptoms", [])[:5]:
            explanation += f"• {symptom['symptom'].title()}\n"
        
        # Add image evidence
        img = state.get("image_findings", {})
        if img.get("top_finding"):
            explanation += f"• X-ray: {img['top_finding']} (confidence: {img.get('confidence', 0):.1%})\n"
        
        # Add differential
        explanation += "\nDIFFERENTIAL DIAGNOSES:\n"
        for i, dx in enumerate(differential[:3], 1):
            explanation += f"{i}. {dx['disease']} ({dx['confidence']:.1%})\n"
        
        # Add safety notes
        if state.get("safety_alerts"):
            explanation += f"\n⚠️ ALERTS: {', '.join(state['safety_alerts'])}\n"
        
        explanation += "\n⚠️ This is AI-assisted analysis. Final diagnosis requires physician review."
        
        return {"explanation": explanation}
    
    def run(self, patient_data: dict) -> DiagnosticState:
        """Execute the diagnostic workflow"""
        start_time = time.time()
        
        initial_state = DiagnosticState(
            patient_id=patient_data.get("patient_id", ""),
            symptoms_text=patient_data.get("symptoms", ""),
            lab_results=patient_data.get("labs"),
            image_path=patient_data.get("image_path"),
            patient_history=patient_data.get("history"),
            extracted_symptoms=[],
            extracted_vitals={},
            image_findings={},
            kg_diseases=[],
            similar_cases=[],
            differential_diagnoses=[],
            primary_diagnosis={},
            confidence=0.0,
            confidence_interval=(0.0, 0.0),
            conflicts_detected=[],
            safety_alerts=[],
            needs_escalation=False,
            escalation_reason=None,
            explanation="",
            feature_importances={},
            processing_time_ms=0,
            model_versions={}
        )
        
        result = self.workflow.invoke(initial_state)
        result["processing_time_ms"] = int((time.time() - start_time) * 1000)
        
        return result
```

---

# STAGE 7: SAFETY LAYER (Week 8)

## Step 7.1: Safety Validator
```python
# src/safety/validator.py
from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger

class SafetyValidator:
    """Comprehensive safety validation for medical diagnoses"""
    
    def __init__(self, config):
        self.config = config
        self.high_risk_conditions = [
            "acute myocardial infarction",
            "pulmonary embolism", 
            "stroke",
            "sepsis",
            "anaphylaxis",
            "meningitis"
        ]
    
    def validate_diagnosis(self, state: Dict) -> Dict:
        """Run all safety validations"""
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
        
        return {
            "all_passed": all_passed,
            "validations": validations,
            "alerts": alerts,
            "requires_escalation": not all_passed
        }
    
    def _check_confidence(self, state: Dict) -> Dict:
        """Check if confidence meets threshold"""
        confidence = state.get("confidence", 0)
        threshold = self.config.min_confidence_threshold
        
        passed = confidence >= threshold
        
        return {
            "passed": passed,
            "confidence": confidence,
            "threshold": threshold,
            "alerts": [] if passed else [f"Confidence {confidence:.1%} below threshold {threshold:.1%}"]
        }
    
    def _check_critical_conditions(self, state: Dict) -> Dict:
        """Flag critical conditions requiring immediate attention"""
        primary = state.get("primary_diagnosis", {})
        disease = primary.get("disease", "").lower()
        
        is_critical = any(cond in disease for cond in self.high_risk_conditions)
        
        return {
            "passed": True,  # Always pass, but alert
            "is_critical": is_critical,
            "alerts": [f"⚠️ CRITICAL: {disease} requires immediate specialist attention"] if is_critical else []
        }
    
    def _check_data_quality(self, state: Dict) -> Dict:
        """Check if input data quality is sufficient"""
        issues = []
        
        # Check symptoms
        symptoms = state.get("extracted_symptoms", [])
        if len(symptoms) < 2:
            issues.append("Insufficient symptoms extracted")
        
        # Check image quality
        image = state.get("image_findings", {})
        if image.get("needs_review"):
            issues.append(f"Image quality issue: {image.get('review_reason')}")
        
        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "alerts": issues
        }
    
    def _perform_sanity_check(self, state: Dict) -> Dict:
        """Verify diagnosis makes logical sense"""
        primary = state.get("primary_diagnosis", {})
        symptoms = [s["symptom"] for s in state.get("extracted_symptoms", [])]
        
        # Disease-symptom requirements
        requirements = {
            "myocardial infarction": ["chest pain"],
            "pneumonia": ["cough", "fever"],
            "appendicitis": ["abdominal pain"]
        }
        
        disease = primary.get("disease", "").lower()
        
        for condition, required in requirements.items():
            if condition in disease:
                if not any(req in " ".join(symptoms).lower() for req in required):
                    return {
                        "passed": False,
                        "alerts": [f"Diagnosis '{disease}' but missing typical symptom(s): {required}"]
                    }
        
        return {"passed": True, "alerts": []}
    
    def _check_signal_conflicts(self, state: Dict) -> Dict:
        """Check for conflicting signals between modalities"""
        conflicts = state.get("conflicts_detected", [])
        
        return {
            "passed": len(conflicts) == 0,
            "conflicts": conflicts,
            "alerts": [f"Signal conflict: {c}" for c in conflicts]
        }
```

---

# STAGE 8: API & DEPLOYMENT (Weeks 9-10)

## Step 8.1: FastAPI Application
```python
# src/api/main.py
from fastapi import FastAPI, HTTPException, Depends, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uuid

from src.config import get_settings, Settings
from src.reasoning.agent import DiagnosticAgent

app = FastAPI(
    title="CDSS API",
    description="Clinical Decision Support System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class DiagnosisRequest(BaseModel):
    patient_id: Optional[str] = None
    symptoms: str
    age: Optional[int] = None
    sex: Optional[str] = None
    lab_results: Optional[dict] = None

class DiagnosisResponse(BaseModel):
    request_id: str
    primary_diagnosis: dict
    differential_diagnoses: list
    confidence: float
    explanation: str
    safety_alerts: list
    requires_review: bool
    processing_time_ms: int

# Dependency
def get_agent(settings: Settings = Depends(get_settings)):
    return DiagnosticAgent(settings)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

@app.post("/api/v1/diagnose", response_model=DiagnosisResponse)
async def diagnose(
    request: DiagnosisRequest,
    agent: DiagnosticAgent = Depends(get_agent)
):
    """Generate diagnosis from patient data"""
    
    request_id = str(uuid.uuid4())
    
    try:
        result = agent.run({
            "patient_id": request.patient_id or request_id,
            "symptoms": request.symptoms,
            "labs": request.lab_results
        })
        
        return DiagnosisResponse(
            request_id=request_id,
            primary_diagnosis=result.get("primary_diagnosis", {}),
            differential_diagnoses=result.get("differential_diagnoses", []),
            confidence=result.get("confidence", 0.0),
            explanation=result.get("explanation", ""),
            safety_alerts=result.get("safety_alerts", []),
            requires_review=result.get("needs_escalation", False),
            processing_time_ms=result.get("processing_time_ms", 0)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/diagnose-with-image")
async def diagnose_with_image(
    symptoms: str,
    image: UploadFile = File(...),
    agent: DiagnosticAgent = Depends(get_agent)
):
    """Generate diagnosis including medical image analysis"""
    
    # Save uploaded image
    image_path = f"/tmp/{uuid.uuid4()}_{image.filename}"
    with open(image_path, "wb") as f:
        f.write(await image.read())
    
    result = agent.run({
        "symptoms": symptoms,
        "image_path": image_path
    })
    
    return result
```

## Step 8.2: Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spacy model
RUN python -m spacy download en_core_web_sm

# Copy application
COPY src/ src/
COPY configs/ configs/

# Run
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

# STAGE 9: MONITORING & VALIDATION (Weeks 11-12)

## Step 9.1: Prometheus Metrics
```python
# src/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Counters
diagnosis_requests = Counter(
    'cdss_diagnosis_requests_total',
    'Total diagnosis requests',
    ['status']  # success, error, escalated
)

# Histograms
diagnosis_latency = Histogram(
    'cdss_diagnosis_latency_seconds',
    'Diagnosis latency',
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0]
)

confidence_distribution = Histogram(
    'cdss_confidence_score',
    'Distribution of confidence scores',
    buckets=[0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
)

# Gauges
model_accuracy = Gauge(
    'cdss_model_accuracy',
    'Current model accuracy (7-day rolling)',
    ['disease_category']
)
```

## Step 9.2: Test Suite
```python
# tests/test_diagnostic_agent.py
import pytest
from src.reasoning.agent import DiagnosticAgent
from src.config import get_settings

@pytest.fixture
def agent():
    return DiagnosticAgent(get_settings())

class TestDiagnosticAgent:
    
    def test_pneumonia_detection(self, agent):
        """Test typical pneumonia case"""
        result = agent.run({
            "symptoms": "Patient reports fever x3 days, productive cough, shortness of breath"
        })
        
        assert result["confidence"] > 0.5
        assert any("pneumonia" in d["disease"].lower() 
                   for d in result["differential_diagnoses"])
    
    def test_low_confidence_escalation(self, agent):
        """Test that low confidence triggers escalation"""
        result = agent.run({
            "symptoms": "Patient feels unwell"  # Vague symptoms
        })
        
        # Should escalate due to insufficient information
        assert result["needs_escalation"] or result["confidence"] < 0.6
    
    def test_critical_condition_alert(self, agent):
        """Test that critical conditions are flagged"""
        result = agent.run({
            "symptoms": "Severe chest pain radiating to left arm, diaphoresis, nausea"
        })
        
        assert "CRITICAL_CONDITION_DETECTED" in result.get("safety_alerts", []) or \
               any("myocardial" in d["disease"].lower() 
                   for d in result["differential_diagnoses"])
    
    def test_negation_handling(self, agent):
        """Test that negated symptoms are handled correctly"""
        result = agent.run({
            "symptoms": "Patient denies fever, no cough, no chest pain. Reports mild fatigue."
        })
        
        # Should not diagnose respiratory conditions
        symptoms = result.get("extracted_symptoms", [])
        fever_symptoms = [s for s in symptoms if "fever" in s["symptom"] and not s["negated"]]
        assert len(fever_symptoms) == 0
```

---

# VALIDATION CHECKLIST

## Before Deployment

- [ ] All tests pass: `pytest tests/ -v`
- [ ] Docker builds: `docker build -t cdss:latest .`
- [ ] API responds: `curl http://localhost:8000/health`
- [ ] Diagnosis endpoint works with sample data
- [ ] Image analysis returns plausible results
- [ ] Safety alerts trigger correctly
- [ ] Low confidence cases escalate
- [ ] Prometheus metrics exposed
- [ ] Logs capture all decisions

## Accuracy Targets

| Metric | Target | Validation Method |
|--------|--------|------------------|
| Overall Accuracy | > 85% | Test on 500 cases |
| Critical Condition Detection | > 95% | Must not miss emergencies |
| False Negative Rate | < 8% | Especially for serious conditions |
| Escalation Rate | 15-25% | Too low = overconfident |
| Latency (p95) | < 5 sec | Load testing |

---

# QUICK START COMMANDS

```bash
# 1. Setup
docker-compose up -d
pip install -r requirements.txt
python scripts/download_data.py

# 2. Initialize databases
python scripts/init_databases.py

# 3. Run tests
pytest tests/ -v

# 4. Start API
uvicorn src.api.main:app --reload

# 5. Test diagnosis
curl -X POST http://localhost:8000/api/v1/diagnose \
  -H "Content-Type: application/json" \
  -d '{"symptoms": "fever, cough, shortness of breath"}'
```

---

**END OF PRD**

This document provides complete, executable instructions for building the CDSS. Execute stages sequentially, validating each before proceeding.

---

# STAGE 10: INFERENCE OPTIMIZATION (Week 13)

## Goal
Optimize model inference for <2 second latency targets.

## Step 10.1: Redis Caching Layer
```python
# src/cache/inference_cache.py
import redis
import hashlib
import json
from typing import Optional, Dict
from loguru import logger

class InferenceCache:
    """Cache diagnosis results for repeated queries"""
    
    def __init__(self, redis_url: str, ttl_seconds: int = 3600):
        self.redis_client = redis.from_url(redis_url)
        self.ttl = ttl_seconds
    
    def _generate_key(self, symptoms: str, image_hash: Optional[str] = None) -> str:
        """Generate cache key from input data"""
        content = f"{symptoms.lower().strip()}:{image_hash or 'no_image'}"
        return f"cdss:diagnosis:{hashlib.md5(content.encode()).hexdigest()}"
    
    def get_cached(self, symptoms: str, image_hash: Optional[str] = None) -> Optional[Dict]:
        """Retrieve cached diagnosis if available"""
        key = self._generate_key(symptoms, image_hash)
        cached = self.redis_client.get(key)
        
        if cached:
            logger.info(f"Cache HIT: {key[:20]}...")
            return json.loads(cached)
        
        logger.info(f"Cache MISS: {key[:20]}...")
        return None
    
    def cache_result(self, symptoms: str, result: Dict, image_hash: Optional[str] = None):
        """Cache diagnosis result"""
        key = self._generate_key(symptoms, image_hash)
        self.redis_client.setex(key, self.ttl, json.dumps(result))
        logger.info(f"Cached: {key[:20]}...")
    
    def invalidate(self, pattern: str = "*"):
        """Invalidate cache entries"""
        keys = self.redis_client.keys(f"cdss:diagnosis:{pattern}")
        if keys:
            self.redis_client.delete(*keys)
            logger.info(f"Invalidated {len(keys)} cache entries")
```

## Step 10.2: Model Quantization
```python
# scripts/quantize_models.py
"""
Quantize models to 4-bit for faster CPU inference.
"""
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch

def quantize_biobert():
    """Quantize BioBERT to 4-bit"""
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        "emilyalsentzer/Bio_ClinicalBERT",
        quantization_config=quantization_config,
        device_map="auto"
    )
    
    # Save quantized model
    model.save_pretrained("models/biobert_4bit")
    print("BioBERT quantized and saved!")

def export_to_onnx():
    """Export model to ONNX for optimized inference"""
    from optimum.onnxruntime import ORTModelForSequenceClassification
    
    model = ORTModelForSequenceClassification.from_pretrained(
        "emilyalsentzer/Bio_ClinicalBERT",
        export=True
    )
    model.save_pretrained("models/biobert_onnx")
    print("Exported to ONNX!")

if __name__ == "__main__":
    quantize_biobert()
    export_to_onnx()
```

## Step 10.3: Batch Processing
```python
# src/api/batch_processor.py
from typing import List, Dict
import asyncio
from concurrent.futures import ThreadPoolExecutor

class BatchProcessor:
    """Process multiple diagnoses in parallel"""
    
    def __init__(self, agent, max_workers: int = 4):
        self.agent = agent
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def process_batch(self, requests: List[Dict]) -> List[Dict]:
        """Process batch of diagnosis requests"""
        loop = asyncio.get_event_loop()
        
        # Process in parallel
        tasks = [
            loop.run_in_executor(self.executor, self.agent.run, req)
            for req in requests
        ]
        
        results = await asyncio.gather(*tasks)
        return results
```

## Latency Targets

| Component | Target | Optimization |
|-----------|--------|-------------|
| Image preprocessing | < 50ms | OpenCV with SIMD |
| BiomedCLIP inference | < 200ms | Quantization + caching |
| NLP extraction | < 100ms | ONNX runtime |
| KG query | < 50ms | Neo4j indexes |
| Total e2e | < 500ms | Aggressive caching |

---

# STAGE 11: CI/CD PIPELINE (Week 14)

## Goal
Automated testing, deployment, and rollback procedures.

## Step 11.1: GitHub Actions Workflow
```yaml
# .github/workflows/ci-cd.yml
name: CDSS CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run unit tests
        run: pytest tests/unit -v --cov=src --cov-report=xml
      
      - name: Run integration tests
        run: pytest tests/integration -v
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Build Docker image
        run: docker build -t cdss:${{ github.sha }} .
      
      - name: Push to registry
        run: |
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
          docker tag cdss:${{ github.sha }} cdss:latest
          docker push cdss:${{ github.sha }}
          docker push cdss:latest

  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    steps:
      - name: Deploy to staging
        run: |
          # Deploy 10% traffic to new version
          kubectl set image deployment/cdss-staging cdss=cdss:${{ github.sha }}
          kubectl rollout status deployment/cdss-staging

  deploy-production:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to production (canary)
        run: |
          # Canary deployment: 5% traffic first
          kubectl set image deployment/cdss-canary cdss=cdss:${{ github.sha }}
          sleep 300  # Wait 5 minutes
          
          # Check metrics
          ACCURACY=$(curl -s http://prometheus:9090/api/v1/query?query=cdss_accuracy | jq '.data.result[0].value[1]')
          if (( $(echo "$ACCURACY < 0.85" | bc -l) )); then
            echo "Accuracy dropped below 85%, rolling back"
            kubectl rollout undo deployment/cdss-canary
            exit 1
          fi
          
          # Full rollout
          kubectl set image deployment/cdss-prod cdss=cdss:${{ github.sha }}
          kubectl rollout status deployment/cdss-prod
```

## Step 11.2: A/B Testing Framework
```python
# src/ab_testing/router.py
import random
from typing import Dict

class ABTestRouter:
    """Route requests between model versions for A/B testing"""
    
    def __init__(self, config: Dict):
        self.experiments = config.get("experiments", {})
    
    def get_model_version(self, user_id: str, experiment_name: str) -> str:
        """Determine which model version to use"""
        experiment = self.experiments.get(experiment_name)
        if not experiment:
            return "control"
        
        # Consistent hashing for user
        user_hash = hash(user_id) % 100
        
        cumulative = 0
        for variant, percentage in experiment["variants"].items():
            cumulative += percentage
            if user_hash < cumulative:
                return variant
        
        return "control"

# Config example
AB_CONFIG = {
    "experiments": {
        "new_vision_model": {
            "variants": {
                "control": 90,  # Current model
                "treatment": 10  # New model
            },
            "metrics": ["accuracy", "latency"]
        }
    }
}
```

## Step 11.3: Rollback Procedure
```bash
#!/bin/bash
# scripts/rollback.sh

# Get previous deployment
PREVIOUS_REVISION=$(kubectl rollout history deployment/cdss-prod | tail -2 | head -1 | awk '{print $1}')

echo "Rolling back to revision $PREVIOUS_REVISION..."
kubectl rollout undo deployment/cdss-prod --to-revision=$PREVIOUS_REVISION

# Verify
kubectl rollout status deployment/cdss-prod

echo "Rollback complete. Verifying health..."
curl -f http://cdss-prod/health || echo "HEALTH CHECK FAILED"
```

---

# STAGE 12: DOCTOR UX DASHBOARD (Week 15)

## Goal
Build intuitive interface for doctor workflow integration.

## Step 12.1: Streamlit Dashboard
```python
# app/doctor_dashboard.py
import streamlit as st
from src.reasoning.agent import DiagnosticAgent
from src.config import get_settings

st.set_page_config(
    page_title="CDSS - Clinical Decision Support",
    page_icon="🏥",
    layout="wide"
)

def main():
    st.title("🏥 Clinical Decision Support System")
    st.markdown("AI-assisted diagnostic support for healthcare providers")
    
    # Sidebar: Patient Info
    with st.sidebar:
        st.header("📋 Patient Information")
        patient_id = st.text_input("Patient ID", placeholder="P12345")
        age = st.number_input("Age", 0, 120, 45)
        sex = st.selectbox("Sex", ["Male", "Female"])
        
    # Main area: Two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Clinical Input")
        
        # Symptoms
        symptoms = st.text_area(
            "Chief Complaint & History",
            placeholder="Patient presents with fever x3 days, productive cough, shortness of breath...",
            height=150
        )
        
        # Image upload
        image_file = st.file_uploader(
            "Upload X-ray (optional)",
            type=["png", "jpg", "jpeg", "dcm"]
        )
        
        # Lab results
        with st.expander("Lab Results (optional)"):
            wbc = st.number_input("WBC (x10³/μL)", 0.0, 50.0, 0.0)
            crp = st.number_input("CRP (mg/L)", 0.0, 500.0, 0.0)
            procalcitonin = st.number_input("Procalcitonin (ng/mL)", 0.0, 100.0, 0.0)
        
        # Analyze button
        if st.button("🔍 Analyze", type="primary", use_container_width=True):
            if symptoms:
                with st.spinner("Analyzing..."):
                    result = run_diagnosis(symptoms, age, sex)
                    st.session_state.result = result
            else:
                st.warning("Please enter symptoms")
    
    with col2:
        st.subheader("🩺 AI Analysis")
        
        if "result" in st.session_state:
            result = st.session_state.result
            
            # Primary diagnosis
            primary = result.get("primary_diagnosis", {})
            confidence = result.get("confidence", 0)
            
            # Color code by confidence
            if confidence > 0.8:
                st.success(f"**{primary.get('disease', 'Unknown')}** (Confidence: {confidence:.0%})")
            elif confidence > 0.6:
                st.warning(f"**{primary.get('disease', 'Unknown')}** (Confidence: {confidence:.0%})")
            else:
                st.error(f"**{primary.get('disease', 'Unknown')}** (Confidence: {confidence:.0%})")
            
            # Safety alerts
            if result.get("safety_alerts"):
                for alert in result["safety_alerts"]:
                    st.error(f"⚠️ {alert}")
            
            # Differential diagnoses
            st.markdown("### Differential Diagnoses")
            for i, dx in enumerate(result.get("differential_diagnoses", [])[:5], 1):
                st.markdown(f"{i}. **{dx['disease']}** ({dx['confidence']:.0%})")
            
            # Explanation
            with st.expander("📖 Reasoning"):
                st.markdown(result.get("explanation", "No explanation available"))
            
            # Doctor action
            st.markdown("---")
            st.markdown("### Doctor Decision")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("✅ Confirm Diagnosis", use_container_width=True):
                    save_confirmation(result, confirmed=True)
                    st.success("Diagnosis confirmed!")
            with col_b:
                if st.button("🔄 Override", use_container_width=True):
                    st.session_state.show_override = True
            
            if st.session_state.get("show_override"):
                correct_diagnosis = st.text_input("Correct diagnosis:")
                notes = st.text_area("Notes for AI learning:")
                if st.button("Submit Override"):
                    save_confirmation(result, confirmed=False, 
                                    override=correct_diagnosis, notes=notes)
                    st.success("Override saved for model improvement!")

@st.cache_resource
def get_agent():
    return DiagnosticAgent(get_settings())

def run_diagnosis(symptoms, age, sex):
    agent = get_agent()
    return agent.run({
        "symptoms": symptoms,
        "patient_info": {"age": age, "sex": sex}
    })

def save_confirmation(result, confirmed, override=None, notes=None):
    # Save to database for model retraining
    pass

if __name__ == "__main__":
    main()
```

## Doctor Workflow Integration

```
┌─────────────────────────────────────────────────────────────┐
│                    DOCTOR WORKFLOW                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. PATIENT INTAKE                                          │
│     └─ Nurse enters vitals, chief complaint                 │
│                                                             │
│  2. AI ANALYSIS (Background)                                │
│     └─ CDSS analyzes while doctor reviews patient           │
│                                                             │
│  3. DOCTOR SEES PATIENT                                     │
│     └─ Examines patient, takes history                      │
│                                                             │
│  4. AI SUGGESTION (Pop-up)                                  │
│     └─ "Based on symptoms + X-ray: Pneumonia (87%)"         │
│     └─ Show differential, key findings, suggested tests     │
│                                                             │
│  5. DOCTOR DECISION                                         │
│     ├─ CONFIRM: "Yes, agree with AI"                        │
│     └─ OVERRIDE: "No, I think it's TB" → Logged for ML      │
│                                                             │
│  6. ORDER TESTS / TREATMENT                                 │
│     └─ AI suggests: "Consider sputum culture"               │
│                                                             │
│  7. FEEDBACK LOOP                                           │
│     └─ Lab results confirm/deny → Model retraining          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

# STAGE 13: DETAILED MONITORING (Week 16)

## Goal
Production-grade monitoring with alert thresholds and incident response.

## Step 13.1: Prometheus Metrics Definitions
```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

rule_files:
  - /etc/prometheus/alerts/*.yml

scrape_configs:
  - job_name: 'cdss-api'
    static_configs:
      - targets: ['cdss-api:8000']
    metrics_path: /metrics
```

## Step 13.2: Alert Rules
```yaml
# monitoring/alerts/cdss_alerts.yml
groups:
  - name: CDSS Critical Alerts
    rules:
      - alert: AccuracyDropped
        expr: cdss_model_accuracy < 0.85
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Model accuracy below 85%"
          description: "CDSS accuracy is {{ $value }}%. Immediate investigation required."
          runbook: "https://docs.cdss.io/runbooks/accuracy-drop"
      
      - alert: HighLatency
        expr: histogram_quantile(0.95, cdss_diagnosis_latency_seconds_bucket) > 5
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "P95 latency exceeds 5 seconds"
          description: "P95 latency is {{ $value }}s. Check for resource constraints."
      
      - alert: EscalationRateHigh
        expr: rate(cdss_escalations_total[1h]) / rate(cdss_diagnosis_requests_total[1h]) > 0.3
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Escalation rate above 30%"
          description: "{{ $value | humanizePercentage }} of cases escalating. Model may be underconfident."
      
      - alert: FalseNegativeSpike
        expr: increase(cdss_false_negatives_total[1h]) > 5
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Multiple false negatives detected"
          description: "{{ $value }} false negatives in last hour. URGENT: Review missed cases."
      
      - alert: ServiceDown
        expr: up{job="cdss-api"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "CDSS API is down"
          description: "No scrapes successful for 1 minute."

  - name: CDSS Business Alerts
    rules:
      - alert: LowDailyVolume
        expr: increase(cdss_diagnosis_requests_total[24h]) < 100
        for: 1h
        labels:
          severity: info
        annotations:
          summary: "Low daily diagnosis volume"
          description: "Only {{ $value }} diagnoses in 24h. Check for service issues."
```

## Step 13.3: Grafana Dashboard JSON
```json
{
  "title": "CDSS Operations Dashboard",
  "panels": [
    {
      "title": "Diagnosis Requests/min",
      "type": "graph",
      "targets": [
        {"expr": "rate(cdss_diagnosis_requests_total[1m])"}
      ]
    },
    {
      "title": "Model Accuracy (7-day rolling)",
      "type": "gauge",
      "targets": [
        {"expr": "cdss_model_accuracy"}
      ],
      "thresholds": [
        {"value": 0, "color": "red"},
        {"value": 0.85, "color": "yellow"},
        {"value": 0.92, "color": "green"}
      ]
    },
    {
      "title": "Latency Distribution",
      "type": "heatmap",
      "targets": [
        {"expr": "histogram_quantile(0.50, cdss_diagnosis_latency_seconds_bucket)"},
        {"expr": "histogram_quantile(0.95, cdss_diagnosis_latency_seconds_bucket)"},
        {"expr": "histogram_quantile(0.99, cdss_diagnosis_latency_seconds_bucket)"}
      ]
    },
    {
      "title": "Escalation Rate",
      "type": "stat",
      "targets": [
        {"expr": "sum(rate(cdss_escalations_total[1h])) / sum(rate(cdss_diagnosis_requests_total[1h]))"}
      ]
    },
    {
      "title": "Safety Alerts (24h)",
      "type": "table",
      "targets": [
        {"expr": "increase(cdss_safety_alerts_total[24h])"}
      ]
    }
  ]
}
```

## Step 13.4: Incident Response Procedures
```markdown
# CDSS Incident Response Playbook

## Severity Levels

| Level | Description | Response Time | Escalation |
|-------|-------------|---------------|------------|
| P1 | Service down, false negatives | < 15 min | On-call + manager |
| P2 | Accuracy < 85%, high latency | < 1 hour | On-call |
| P3 | Elevated escalations, drift | < 4 hours | Next business day |

## Runbooks

### Accuracy Drop (P1/P2)
1. Check recent deployments: `kubectl rollout history deployment/cdss`
2. Compare model versions in A/B test
3. Check data distribution shift
4. If new deployment: rollback `./scripts/rollback.sh`
5. Page ML team if persistent

### High Latency (P2)
1. Check pod resource usage: `kubectl top pods`
2. Check database latency: Neo4j, Weaviate
3. Check cache hit rate: Redis
4. Scale up if needed: `kubectl scale deployment/cdss --replicas=5`

### False Negative Spike (P1)
1. IMMEDIATELY review missed cases
2. Check if specific disease category affected
3. Check recent data or model changes
4. Consider taking system offline if patient safety at risk
5. Page ML team and medical advisory board
```

## Validation
- [ ] Prometheus scraping CDSS metrics
- [ ] Grafana dashboard showing all panels
- [ ] Alerts firing correctly in test environment
- [ ] Runbooks reviewed by on-call team

---

# FINAL VALIDATION CHECKLIST

## Before Go-Live

### Technical Validation
- [ ] All tests pass: `pytest tests/ -v`
- [ ] Docker builds: `docker build -t cdss:latest .`
- [ ] API responds: `curl http://localhost:8000/health`
- [ ] Diagnosis endpoint works with sample data
- [ ] Image analysis returns plausible results
- [ ] Safety alerts trigger correctly
- [ ] Low confidence cases escalate
- [ ] Prometheus metrics exposed
- [ ] Logs capture all decisions
- [ ] Caching reduces latency
- [ ] CI/CD pipeline runs successfully

### Safety Validation
- [ ] 500+ test cases evaluated
- [ ] Accuracy > 85%
- [ ] Critical condition detection > 95%
- [ ] False negative rate < 8%
- [ ] Escalation rate 15-25%
- [ ] Doctor feedback mechanism works

### Compliance Validation
- [ ] Audit logs complete
- [ ] Data encryption verified
- [ ] Access controls tested
- [ ] Disclaimer displayed

---

**COMPLETE PRD v2.0**

All gaps from PRD_Evaluation.md have been addressed:
- ✅ Gap #1: Data Pipeline (Stage 2.5 in Part 1)
- ✅ Gap #2: Inference Optimization (Stage 10)
- ✅ Gap #3: CI/CD Pipeline (Stage 11)
- ✅ Gap #4: Doctor Adoption UX (Stage 12)
- ✅ Gap #5: Monitoring Details (Stage 13)
