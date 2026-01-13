# VerdictMed AI 🏥

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18.0-61DAFB)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95-009688)](https://fastapi.tiangolo.com/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.0-008CC1)](https://neo4j.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-83%20Passing-success)](tests/)
[![MVP Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)]()

> **"Reducing diagnostic error and physician burnout through Explainable Multimodal AI."**

**VerdictMed AI** is a production-grade Clinical Decision Support System (CDSS) that integrates multimodal patient data—medical imaging (DICOM/X-ray), clinical notes, and lab values—into a unified diagnostic reasoning engine. Built with safety and explainability at its core, it leverages Knowledge Graphs and LLMs to provide evidence-based recommendations.

---

## 🏗️ System Architecture

VerdictMed AI follows a modular **5-Layer Architecture** designed for scalability, interpretability, and safety.

```mermaid
graph TD
    subgraph "Application Layer (Frontend)"
        UI[React Clinical Dashboard]
        Inputs[Multimodal Input Handler]
        Vis[Visualization Engine]
    end

    subgraph "API Layer"
        FastAPI[FastAPI Gateway]
        Auth[Security & Auth]
        Val[Input Validator]
    end

    subgraph "Reasoning Layer (The Brain)"
        Agent[Diagnostic Agent (LLM)]
        Fusion[Multimodal Fusion]
        Safety[Safety Validator]
        Explain[Explainability Engine (SHAP)]
    end

    subgraph "Perception Layer"
        Vision[Vision Module (BiomedCLIP)]
        NLP[NLP Module (SciSpacy)]
    end

    subgraph "Data & Knowledge Layer"
        KG[(Neo4j Knowledge Graph)]
        Redis[(Redis Cache)]
        Vector[(FAISS Vector Store)]
    end

    UI --> FastAPI
    FastAPI --> Val
    Val --> Fusion
    Fusion --> Vision
    Fusion --> NLP
    Vision --> Agent
    NLP --> Agent
    Agent <--> KG
    Agent --> Safety
    Safety --> Explain
    Explain --> Vis
```

---

## 🚀 Key Features

### 1. Multimodal Analysis 👁️🗣️🩸
- **Medical Imaging**: Analyzes Chest X-rays (DICOM/PNG) using **BiomedCLIP** to detect pathologies like Pneumonia, Cardiomegaly, and Pleural Effusion.
- **Clinical NLP**: Extracts symptoms, conditions, and entities from unstructured physician notes using **SciSpacy**.
- **Lab Data**: Contextualizes structured lab values (CBC, metabolic panels) against reference ranges.

### 2. Neuro-Symbolic Reasoning 🧠
- Combines the flexibility of **LLMs** with the factual precision of a **Neo4j Knowledge Graph**.
- Maps extracted symptoms to standard **ICD-10 codes**.
- Validates LLM hypotheses against grounded medical knowledge (Graph RAG).

### 3. Explainability & Trust 🔍
- **SHAP Integrations**: Visualizes feature importance contributions for every diagnosis.
- **Reasoning Chains**: Provides transparent, step-by-step logic traces (e.g., *Steps 1-5*).
- **Evidence Linking**: Ties recommendations directly to patient data sources.

### 4. Safety-First Design 🛡️
- **Human-in-the-Loop**: High-stakes decisions always flag for physician review.
- **Critical Alerts**: Real-time warnings for critical conditions (e.g., Sepsis, MI).
- **Confidence Thresholds**: Automatic referral for low-confidence predictions (<55%).

---

## 🛠️ Tech Stack

| Component | Technology | Role |
|-----------|------------|------|
| **Frontend** | React, Tailwind CSS | 3-Column Clinical Dashboard |
| **Backend** | FastAPI, Python 3.9 | High-performance API Gateway |
| **Vision** | BiomedCLIP, PyTorch | Zero-shot Medical Image Classification |
| **NLP** | SciSpacy, regular expressions | Entity Extraction & Negation |
| **Knowledge** | Neo4j (Cypher) | Disease-Symptom Knowledge Graph |
| **Explainability** | SHAP, GradCAM | Model Interpretability |
| **Deployment** | Docker, Docker Compose | Containerized Orchestration |

---

## ⚡ Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.9+ (for local dev)
- Node.js 16+ (for local dev)

### Option 1: Docker (Recommended)
Run the entire stack with one command:
```bash
docker-compose up --build
```
Access the dashboard at **http://localhost:3000** and API docs at **http://localhost:8000/docs**.

### Option 2: Local Development

**1. Backend Setup**
```bash
# Clone repository
git clone https://github.com/HemantSudarshan/VerdictMed-AI.git
cd VerdictMed-AI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run server
python -m uvicorn src.api.main:app --reload
```

**2. Frontend Setup**
```bash
cd frontend
npm install
npm run dev
```

---

## 🧪 Testing & Quality

VerdictMed AI maintains a high standard of code quality with **~75% test coverage**.

### Run the Test Suite
```bash
# Run all 83+ unit and integration tests
python -m pytest tests/

# Run specific modules
python -m pytest tests/unit/test_vision.py
python -m pytest tests/unit/test_neo4j_service.py
```

### Safety Validation
The system includes a dedicated `SafetyValidator` module that enforces:
- ✅ Confidence > 55%
- ✅ No critical flag overrides
- ✅ Data completeness checks

---

## 📊 Project Gallery

### Clinical Dashboard
*(Add your screenshots here)*
The unified interface organizes patient data, analysis, and AI reasoning into a scan-friendly layout.

![Dashboard Preview](docs/dashboard_preview.png)

---

## 🔮 Future Roadmap

- [ ] **MedSAM Integration**: Detailed graphical segmentation of X-ray anomalies.
- [ ] **FHIR Support**: Interoperability with standard EMR systems.
- [ ] **Voice Input**: Speech-to-text for dictating clinical notes.

---

## 👨‍⚕️ Disclaimer

**VerdictMed AI is a Clinical Decision Support System (CDSS) prototype.** 
It is intended to **assist** medical professionals, not replace them. All diagnoses must be verified by a qualified physician. This system is not FDA-cleared for autonomous clinical use.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
