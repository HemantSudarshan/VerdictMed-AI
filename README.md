# 🏥 VerdictMed AI - Clinical Decision Support System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18.0-61DAFB?style=for-the-badge&logo=react&logoColor=white)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.0-008CC1?style=for-the-badge&logo=neo4j&logoColor=white)](https://neo4j.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-83%20Passing-success?style=for-the-badge)](tests/)
[![Status](https://img.shields.io/badge/Status-MVP%20Ready-brightgreen?style=for-the-badge)]()

---

### 💬 *"Bridging the gap between raw clinical data and actionable life-saving insights."*

</div>

---

## 🎯 What is VerdictMed AI?

**VerdictMed AI** is a Clinical Decision Support System (CDSS) prototype that:
- ✅ Analyzes multimodal clinical data (X-rays, text, labs)
- ✅ Provides AI-assisted diagnostic suggestions with explanations
- ✅ Enforces safety validations with human-in-the-loop design
- ✅ Uses Knowledge Graphs to reduce AI hallucinations

### 🧠 Neuro-Symbolic Architecture

VerdictMed combines:
- **Neural Networks** for pattern recognition (BiomedCLIP for images, SciSpacy for text)
- **Knowledge Graphs** for structured medical reasoning (Neo4j)
- **GraphRAG** approach to validate diagnoses against medical ontologies

---

## 📊 Project Status

<table>
<tr>
<td width="50%">

### ✅ Implemented
- 🏗️ 5-Layer Architecture
- 🖼️ Vision Analysis (BiomedCLIP)
- 📝 NLP Pipeline (SciSpacy)
- 🧠 Knowledge Graph (Neo4j with fallback)
- 🎯 Safety Validator
- 🔍 SHAP Explainability
- ⚡ FastAPI Backend
- ⚛️ React Dashboard
- 🐳 Docker Compose
- 📊 Prometheus Monitoring
- 🚨 Alert Rules (6 defined)
- 🔄 CI/CD Pipeline
- 📋 83+ Automated Tests

</td>
<td width="50%">

### 🚧 In Development
- 🏥 Full UMLS Integration
- 🔗 FHIR Interoperability
- 🎨 GradCAM Heatmaps (module exists)
- 📈 500+ Case Validation
- 🗣️ Voice Dictation
- 🌍 Multi-language Support
- 📱 Mobile Interface

</td>
</tr>
</table>

> **Current Version**: v1.0 MVP - Demonstrates core CDSS capabilities

---

## 🏛️ System Architecture

```
┌────────────────────────────────────────────────┐
│          🖥️  PRESENTATION LAYER                │
│            React Clinical Dashboard             │
├────────────────────────────────────────────────┤
│             🔌 API GATEWAY LAYER                │
│         FastAPI • Async Processing             │
├────────────────────────────────────────────────┤
│            🧠 REASONING LAYER                   │
│   Diagnostic Agent • Safety • Explainability   │
├────────────────────────────────────────────────┤
│            👁️  PERCEPTION LAYER                 │
│      Vision (BiomedCLIP) • NLP (SciSpacy)      │
├────────────────────────────────────────────────┤
│           📚 DATA & KNOWLEDGE LAYER             │
│     Neo4j • Redis • PostgreSQL • Weaviate      │
└────────────────────────────────────────────────┘
```

---

## 💡 Key Features

### 1. 🧠 Multimodal Analysis
| Input Type | Processing | Output |
|-----------|-----------|--------|
| 🖼️ **X-ray/DICOM** | BiomedCLIP embeddings | Abnormality detection |
| 📝 **Clinical Notes** | SciSpacy NER + negation | Symptom extraction |
| 🧪 **Lab Values** | Structured parsing | Biomarker analysis |

### 2. 🔍 Explainability
Every diagnosis includes:
- **SHAP Values**: Feature contribution scores
- **Reasoning Chains**: Step-by-step logical path
- **Confidence Scores**: Uncertainty quantification

### 3. 🛡️ Safety Validation
```
Confidence < 55%?        → Escalate to physician
Critical Condition?      → Immediate alert
Signal Conflicts?        → Flag for review
Missing Key Symptoms?    → Sanity check fail
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Frontend** | React 18 + Tailwind CSS | Clinical Dashboard |
| **Backend** | FastAPI (Python 3.10) | API Gateway |
| **Vision** | BiomedCLIP + PyTorch | Medical Image Analysis |
| **NLP** | SciSpacy | Clinical Text Processing |
| **Knowledge** | Neo4j 5.0 | Disease-Symptom Graph |
| **Caching** | Redis 7 | Performance Optimization |
| **Database** | PostgreSQL 15 | Data Persistence |
| **Explainability** | SHAP | Feature Importance |
| **Monitoring** | Prometheus + Grafana | Observability |
| **Infrastructure** | Docker Compose | Service Orchestration |

---

## ⚡ Quick Start

### 🐳 Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/HemantSudarshan/VerdictMed-AI.git
cd VerdictMed-AI

# Start all services
docker-compose up --build

# Wait ~30 seconds for services to initialize
```

**Access Points:**
- 🖥️ **Dashboard**: http://localhost:3000 (React UI)
- 🔌 **API Docs**: http://localhost:8000/docs (Swagger)
- 📊 **Grafana**: http://localhost:3000 (Monitoring - admin/admin)
- 🔍 **Neo4j Browser**: http://localhost:7474 (neo4j/secure_password)

---

### 💻 Local Development

<details>
<summary><strong>Backend Setup</strong></summary>

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start API (requires Docker services running)
uvicorn src.api.main:app --reload
```
</details>

<details>
<summary><strong>Frontend Setup</strong></summary>

```bash
cd frontend
npm install
npm run dev
```
</details>

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run PRD-specific diagnostic tests
pytest tests/unit/test_reasoning_agent.py::TestPRDRequirements -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Test Coverage

| Module | Tests | Coverage |
|--------|-------|----------|
| Vision | 12 | 85% |
| NLP | 15 | 80% |
| Reasoning | 20 | 75% |
| Safety | 18 | 88% |
| Knowledge Graph | 12 | 82% |
| API | 6 | 70% |

**Total**: 83+ tests, ~75% overall coverage

---

## 📊 Monitoring & Alerts

### Prometheus Alert Rules

6 production-grade alerts defined in [`monitoring/alerts/cdss_alerts.yml`](monitoring/alerts/cdss_alerts.yml):

| Alert | Condition | Severity |
|-------|-----------|----------|
| **AccuracyDropped** | Accuracy < 85% for 5m | Critical |
| **HighLatency** | P95 latency > 5s | Warning |
| **EscalationRateHigh** | > 30% escalation rate | Warning |
| **FalseNegativeSpike** | > 5 false negatives/hour | Critical |
| **ServiceDown** | API unavailable | Critical |
| **LowDailyVolume** | < 100 diagnoses/day | Info |

### Incident Response

Full runbooks in [`docs/runbooks/incident-response.md`](docs/runbooks/incident-response.md)

---

## 🚀 CI/CD Pipeline

Automated via GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)):

**Continuous Integration** (every commit):
- ✅ Linting (Ruff) + Type checking (MyPy)
- ✅ Unit & integration tests
- ✅ Security scanning (Trivy)
- ✅ Docker build

**Continuous Deployment**:
- **Staging** (on `develop`): Auto-deploy + smoke tests
- **Production** (on `main`): Canary deployment (5% → validate → 100%) with auto-rollback

---

## 📚 Documentation

- [Architecture Deep Dive](src/README.md)
- [Alert Rules](monitoring/alerts/cdss_alerts.yml)
- [Incident Response](docs/runbooks/incident-response.md)
- [Fusion Strategy](src/fusion/README.md)

---

## 🔮 Roadmap

### Short-term
- [ ] Complete 500+ case validation study
- [ ] Full UMLS ontology integration
- [ ] FHIR R4 support for EHR integration

### Long-term
- [ ] GradCAM heatmap UI integration
- [ ] Voice dictation support
- [ ] Multi-language clinical notes
- [ ] Mobile application

---

## ⚖️ Legal & Compliance

### 🏥 Medical Disclaimer

**VerdictMed AI is a research prototype for demonstration purposes.**

- ❌ **NOT FDA-cleared** for clinical use
- ❌ **NOT a replacement** for physician judgment
- ✅ **DESIGNED AS** a decision support tool requiring human oversight
- ✅ **REQUIRES** physician sign-off on all recommendations

### Data & Privacy
- Encryption: AES-256 for sensitive data
- Audit logging: Complete request/response trail
- No real patient data used in public demo

---

## 📄 License

**MIT License** © 2026 VerdictMed AI Team

Open source and free to use. See [LICENSE](LICENSE) for details.

---

<div align="center">

### 🎯 Built for Medical AI Research

**[⬆ Back to Top](#-verdictmed-ai---clinical-decision-support-system)**

---

**Last Updated:** January 17, 2026 | **Status:** MVP Prototype

</div>
