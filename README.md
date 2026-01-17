# 🏥 VerdictMed AI - Clinical Decision Support System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.0-008CC1?style=for-the-badge&logo=neo4j&logoColor=white)](https://neo4j.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research%20Prototype-blue?style=for-the-badge)]()

---

### *"AI-assisted clinical decision support for medical education and research"*

</div>

---

## 🎯 What is VerdictMed AI?

VerdictMed AI is a **research prototype** for AI-assisted clinical diagnosis. It demonstrates how multimodal medical data (symptoms, lab values, X-rays) can be processed and interpreted using modern AI techniques.

**Key Capabilities:**
- Analyze clinical symptoms using NLP (Natural Language Processing)
- Interpret lab values against clinical thresholds
- Analyze chest X-rays using vision AI (BiomedCLIP)
- Query a medical knowledge graph for symptom-disease relationships
- Detect conflicts between different data sources
- Enforce safety validations requiring physician review

---

## 📊 Current Implementation Status

### ✅ Completed Components

| Component | Status | Description |
|-----------|--------|-------------|
| **NLP Pipeline** | ✅ Done | Symptom extraction, negation detection, abbreviation expansion using SciSpacy |
| **Lab Processor** | ✅ Done | 25+ lab tests with clinical thresholds, severity scoring, pattern detection |
| **Vision Module** | ✅ Done | Chest X-ray analysis using BiomedCLIP (zero-shot classification) |
| **Knowledge Graph** | ✅ Done | Neo4j integration for disease-symptom relationships (with mock fallback) |
| **Safety Validator** | ✅ Done | Confidence thresholds, critical condition detection, conflict flagging |
| **Safety Monitor** | ✅ Done | Fallback mechanisms, low-confidence blocking, audit logging |
| **Multimodal Fusion** | ✅ Done | Weighted scoring (symptoms 30%, labs 35%, image 35%), conflict detection |
| **API Layer** | ✅ Done | FastAPI with `/diagnose` and `/diagnose-with-image` endpoints |
| **Explainability** | ✅ Done | SHAP feature importance, reasoning chain generation |
| **Monitoring** | ✅ Done | Prometheus alerts, incident response runbooks |
| **CI/CD Pipeline** | ✅ Done | GitHub Actions with canary deployment strategy |
| **Tests** | ✅ Done | 95+ automated tests across all modules |

### 🚧 Not Yet Implemented

| Feature | Status |
|---------|--------|
| Full UMLS ontology integration | Planned |
| FHIR interoperability for EHR systems | Planned |
| GradCAM heatmaps for X-ray visualization | Partial (module exists, UI pending) |
| 500+ case validation study | Pending |
| Mobile application | Planned |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    API LAYER (FastAPI)                   │
│         /diagnose  •  /diagnose-with-image              │
├─────────────────────────────────────────────────────────┤
│                   SAFETY MONITOR                         │
│    Low confidence blocking • Fallback • Audit logging    │
├─────────────────────────────────────────────────────────┤
│                 MULTIMODAL FUSION                        │
│   Symptoms (30%) + Labs (35%) + Image (35%) → Score     │
│              Conflict Detection & Explanation            │
├─────────────────────────────────────────────────────────┤
│                  REASONING LAYER                         │
│    SimpleDiagnosticAgent • Knowledge Graph Query         │
├────────────────┬──────────────────┬─────────────────────┤
│   NLP MODULE   │   LAB PROCESSOR  │   VISION MODULE     │
│   (SciSpacy)   │   (Thresholds)   │   (BiomedCLIP)      │
├────────────────┴──────────────────┴─────────────────────┤
│                  DATA LAYER                              │
│   Neo4j (Knowledge Graph) • Redis (Cache) • PostgreSQL   │
└─────────────────────────────────────────────────────────┘
```

---

## 📚 For MBBS Students & Medical Researchers

### What This Project Demonstrates

1. **How AI processes clinical text**: The NLP module extracts symptoms from free-text notes, handles negation ("patient denies fever"), and expands medical abbreviations (SOB → shortness of breath).

2. **How lab values are interpreted**: The lab processor compares values against clinical reference ranges and detects patterns like sepsis (elevated WBC + lactate) or cardiac damage (elevated troponin).

3. **How medical images are analyzed**: BiomedCLIP uses vision AI to classify chest X-ray findings (consolidation, effusion, etc.) without requiring disease-specific training.

4. **How knowledge graphs work**: Neo4j stores disease-symptom relationships from medical ontologies, enabling reasoning about which diseases match a patient's symptoms.

5. **Why safety layers matter**: The system demonstrates how AI must handle uncertainty, flag conflicts, and require human oversight for clinical decisions.

### Research Applications

| Research Area | How VerdictMed Helps |
|---------------|----------------------|
| **Medical AI Safety** | Study how systems handle conflicting signals, low confidence, and edge cases |
| **Clinical NLP** | Evaluate symptom extraction accuracy on different note styles |
| **Multimodal Fusion** | Compare weighting schemes for combining text, labs, and imaging |
| **Explainability** | Study how SHAP values and reasoning chains improve physician trust |
| **Knowledge Graphs** | Explore graph-based medical reasoning vs. pure neural approaches |

### Limitations to Understand

- **NOT for clinical use**: This is a research prototype, not FDA-cleared software
- **Dataset bias**: Models trained on public datasets (NIH ChestX-ray14) have known demographic biases
- **No real validation**: Accuracy claims are based on test data, not clinical trials
- **Mock components**: Some features (like full UMLS integration) use simplified mock implementations

### Setting Up for Research

```bash
# Clone the repository
git clone https://github.com/HemantSudarshan/VerdictMed-AI.git
cd VerdictMed-AI

# Option 1: Docker (easiest)
docker-compose up --build

# Option 2: Local Python
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn src.api.main:app --reload
```

**Access:**
- API Docs: http://localhost:8000/docs
- Neo4j Browser: http://localhost:7474 (neo4j/secure_password)
- Grafana: http://localhost:3000 (admin/admin)

### Example API Usage

```python
import requests

# Text-based diagnosis
response = requests.post(
    "http://localhost:8000/api/v1/diagnose",
    json={
        "symptoms": "Patient has fever for 3 days, productive cough, and shortness of breath",
        "lab_results": {
            "wbc": 15.2,
            "crp": 45.0,
            "procalcitonin": 0.8
        }
    },
    headers={"X-API-Key": "your-api-key"}
)

result = response.json()
print(f"Diagnosis: {result['primary_diagnosis']['disease']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Lab Flags: {result.get('lab_findings', {}).get('flags', [])}")
```

### Key Files to Study

| File | Purpose |
|------|---------|
| `src/nlp/clinical_nlp.py` | Symptom extraction and negation detection |
| `src/nlp/lab_processor.py` | Lab value interpretation with clinical thresholds |
| `src/vision/biomedclip.py` | Chest X-ray analysis using vision AI |
| `src/reasoning/simple_agent.py` | Main diagnostic workflow and multimodal fusion |
| `src/safety/validator.py` | Safety checks and conflict detection |
| `src/safety/safety_monitor.py` | Fallback mechanisms and blocking logic |
| `src/knowledge_graph/query_engine.py` | Neo4j graph queries |

---

## 🧪 Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific module tests
python -m pytest tests/unit/test_lab_processor.py -v
python -m pytest tests/unit/test_safety_monitor.py -v
python -m pytest tests/unit/test_reasoning_agent.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

**Current Test Coverage:** ~75% across 95+ test cases

---

## 📈 Monitoring & Alerts

Six production-grade Prometheus alerts defined in `monitoring/alerts/cdss_alerts.yml`:

| Alert | Trigger | Severity |
|-------|---------|----------|
| AccuracyDropped | Accuracy < 85% for 5min | Critical |
| FalseNegativeSpike | >5 false negatives/hour | Critical |
| ServiceDown | API unavailable 1min | Critical |
| HighLatency | P95 > 5 seconds | Warning |
| EscalationRateHigh | >30% escalation rate | Warning |
| LowDailyVolume | <100 diagnoses/day | Info |

---

## ⚖️ Medical & Legal Disclaimer

**⚠️ RESEARCH PROTOTYPE ONLY**

- This system is **NOT FDA-cleared** for clinical use
- **NOT a replacement** for physician judgment
- All outputs require verification by qualified healthcare providers
- Designed for **educational and research purposes only**
- No real patient data should be processed without proper IRB approval

---

## 📄 License

MIT License © 2026 VerdictMed AI

---

<div align="center">

**Last Updated:** January 17, 2026 | **Status:** Research Prototype

</div>
