# VerdictMed AI - Clinical Decision Support System

[![CI/CD](https://github.com/yourusername/verdictmed-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/verdictmed-ai/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An AI-powered Clinical Decision Support System that provides diagnostic assistance by analyzing patient symptoms, medical images, and lab results.

## ⚠️ Important Disclaimer

> **This system is designed as a SECOND OPINION tool only. It is NOT a replacement for professional medical judgment. All outputs require verification by a qualified healthcare provider.**

---

## 🚀 Features

- **Multimodal Analysis**: Combines symptoms, X-ray images, and lab results
- **Clinical NLP**: Extracts symptoms with negation detection and abbreviation expansion
- **Knowledge Graph**: Neo4j-based medical knowledge for differential diagnosis
- **Safety Layer**: 5-point validation including critical condition detection
- **API Authentication**: Secure endpoints with API key protection
- **Audit Logging**: HIPAA-compliant request logging
- **Redis Caching**: Optimized performance with intelligent caching
- **Doctor Dashboard**: Streamlit UI for clinical interaction

---

## 📦 Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- 8GB+ RAM recommended

### 1. Clone & Setup
```bash
git clone https://github.com/yourusername/verdictmed-ai.git
cd verdictmed-ai

# Create environment file
cp .env.example .env
# Edit .env with your settings
```

### 2. Start Infrastructure
```bash
docker-compose up -d
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Initialize Databases
```bash
python scripts/init_databases.py
```

### 5. Run the API
```bash
uvicorn src.api.main:app --reload
```

### 6. Run the Dashboard (optional)
```bash
streamlit run app/doctor_dashboard.py
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Doctor Dashboard (Streamlit)             │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    FastAPI + Auth + Audit                   │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐ │
│  │  Vision  │  │   NLP    │  │Knowledge │  │   Safety    │ │
│  │ BiomedCLIP│  │ SciSpacy │  │  Graph   │  │  Validator  │ │
│  └──────────┘  └──────────┘  └──────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐ │
│  │PostgreSQL│  │  Neo4j   │  │ Weaviate │  │    Redis    │ │
│  └──────────┘  └──────────┘  └──────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## 📡 API Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | ❌ | Health check |
| `/api/v1/diagnose` | POST | ✅ | Symptom-based diagnosis |
| `/api/v1/diagnose-with-image` | POST | ✅ | Diagnosis with X-ray |
| `/api/v1/symptoms` | GET | ❌ | Common symptoms list |
| `/api/v1/disclaimer` | GET | ❌ | Medical disclaimer |
| `/metrics` | GET | ❌ | Prometheus metrics |

### Example Request
```bash
curl -X POST http://localhost:8000/api/v1/diagnose \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"symptoms": "45yo male with fever, cough, SOB x3 days"}'
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 📊 Monitoring

- **Prometheus**: `http://localhost:9090`
- **Grafana**: `http://localhost:3000` (admin/admin)

Import the dashboard from `monitoring/grafana/dashboards/cdss-dashboard.json`.

---

## 🔒 Security Features

- **API Key Authentication**: All diagnosis endpoints protected
- **PII Encryption**: Patient data encrypted at rest (Fernet)
- **Audit Logging**: All requests logged for compliance
- **Non-root Docker**: Containers run as unprivileged user

---

## 📁 Project Structure

```
VerdictMed AI/
├── src/
│   ├── api/           # FastAPI endpoints, auth, middleware
│   ├── vision/        # BiomedCLIP image analysis
│   ├── nlp/           # Clinical NLP pipeline
│   ├── reasoning/     # Diagnostic agent
│   ├── safety/        # Safety validator
│   ├── knowledge_graph/  # Neo4j queries
│   ├── cache/         # Redis caching
│   ├── security/      # Encryption service
│   └── monitoring/    # Prometheus metrics
├── app/               # Streamlit dashboard
├── tests/             # Unit tests
├── scripts/           # Database init, data streaming
├── monitoring/        # Prometheus & Grafana configs
├── .github/workflows/ # CI/CD pipeline
└── docker-compose.yml # Infrastructure services
```

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `pytest tests/`
4. Submit a Pull Request

---

**Built with ❤️ for safer healthcare AI**
