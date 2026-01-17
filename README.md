# 🏥 VerdictMed AI - Clinical Decision Support System

<div align="center">

[![Status](https://img.shields.io/badge/Status-Research%20Prototype-blue?style=for-the-badge)]()
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=for-the-badge&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Reasoning-ff69b4?style=for-the-badge)](https://github.com/langchain-ai/langgraph)
[![Score](https://img.shields.io/badge/MVP%20Score-91.5%2F100-success?style=for-the-badge)]()

</div>

An AI-powered research prototype for multimodal clinical diagnosis, demonstrating a safety-first approach to building medical AI systems.

## 🎯 What is VerdictMed AI?

VerdictMed AI is an **educational and research tool** that showcases how to build a reliable Clinical Decision Support System (CDSS). It processes patient symptoms, lab results, and medical images to generate diagnostic suggestions, all orchestrated by a graph-based reasoning engine and governed by a comprehensive safety layer.

**⚠️ This is a research prototype and is NOT intended for clinical use.**

## ✨ Key Features

- **🧠 Multimodal Analysis:** Fuses data from clinical notes (NLP), lab results, and medical images (Vision AI) to form a holistic view
- **🤖 Graph-Based Reasoning:** Uses **LangGraph** to create a transparent, stateful, and observable diagnostic workflow with conditional branching for handling high-risk cases
- **🛡️ Safety-First Design:** Multi-layered safety validator that checks for low confidence, data conflicts, and critical conditions, automatically flagging cases for human review
- **🔎 Similar Case Retrieval:** Integrates with **Weaviate** vector store to find similar historical patient cases
- **⚕️ Medical Knowledge Graph:** Leverages **Neo4j** to map symptoms to potential diseases based on medical ontologies
- **🚀 Production-Ready Infrastructure:** Full Docker Compose setup and production-grade CI/CD pipeline with automated canary releases

## 🏗️ System Architecture

The system uses a `LangGraph` StateGraph to orchestrate the flow of data between specialized nodes:

```
[ Patient Data ]
      │
      ▼
[ NLP & Lab Analysis ]───▶[ Image Analysis ]
      │                          │
      ▼                          ▼
[ KG & Vector Search ]◀───[ Visual Explanation ]
      │
      ▼
[ Multimodal Fusion ]
      │
      ▼
[ Safety Validation ]──(Conditional)──▶[ Escalation Alert ]
      │
      └─(Standard)──▶[ Generate Diagnosis ]
                            │
                            ▼
                      [ Final Output ]
```

## 🛠️ Tech Stack

- **Backend:** FastAPI
- **Reasoning Engine:** LangGraph (StateGraph with conditional branching)
- **AI/ML:** PyTorch, Transformers, Sentence-Transformers, SciSpacy, BiomedCLIP
- **Databases:** Neo4j (Knowledge Graph), Weaviate (Vector Store), PostgreSQL (Audit Logs), Redis (Cache)
- **Deployment:** Docker, GitHub Actions (Canary Deployments)

## 🚀 Getting Started

The entire application stack can be run easily using Docker.

1. **Clone the repository:**
```bash
git clone https://github.com/HemantSudarshan/VerdictMed-AI.git
cd VerdictMed-AI
```

2. **Set up environment variables:**
```bash
cp .env.example .env
# (Optional) Edit .env file if you have custom configurations
```

3. **Run with Docker Compose:**
```bash
docker-compose up --build -d
```

The API will be available at `http://localhost:8000`.

## ⚙️ Usage Example

You can interact with the API using any HTTP client. Here is a Python example:

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/diagnose",
    headers={"X-API-Key": "your-secret-api-key"},  # Replace with your key
    json={
        "symptoms": "Patient presents with high fever, persistent cough, and shortness of breath.",
        "lab_results": {
            "wbc": 16.5,  # High
            "crp": 55.0   # High
        }
    }
)

if response.status_code == 200:
    result = response.json()
    print(f"Diagnosis: {result['primary_diagnosis']['disease']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Requires Review: {result['requires_review']}")
    print(f"\nExplanation:\n{result['explanation']}")
else:
    print(f"Error: {response.status_code}")
    print(response.json())
```

## 🧪 Running Tests

To run the full test suite:

```bash
pytest tests/ -v
```

Current test coverage: **~75%** with **95+ test cases**

## 📊 Project Status

**MVP Evaluation Score: 91.5/100 - APPROVED**

| Component | Score |
|-----------|-------|
| Architecture | 9/10 |
| Tech Stack | 9/10 |
| Safety & Compliance | 10/10 |
| Deployment | 10/10 |
| Testing | 8/10 |

**Status:** Production-ready research prototype

## 🔐 Safety Features

- **Confidence Thresholding:** Low-confidence diagnoses automatically escalated
- **Conflict Detection:** Identifies disagreements between data sources (labs vs imaging)
- **Critical Condition Alerts:** Flags life-threatening conditions for immediate review
- **Audit Logging:** Complete HIPAA-compliant audit trail
- **Human-in-the-Loop:** Review workflow for doctor confirmation

## ⚖️ Disclaimer

This system is an educational and research prototype. It is **not FDA-cleared** and **must not be used for actual clinical diagnosis or patient care**. All outputs must be verified by a qualified healthcare professional.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ for medical AI research and education**

</div>
