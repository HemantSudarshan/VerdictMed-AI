# CDSS PRD Part 1: Foundation & Architecture
## AI-Powered Clinical Decision Support System - LLM Execution Guide

**Version**: 1.0 | **Date**: January 2026 | **Status**: Ready for AI Agent Execution

---

## DOCUMENT OVERVIEW

This PRD is designed for an LLM/AI agent to execute step-by-step. Follow each stage sequentially. Each stage has:
- **Goal**: What to achieve
- **Inputs**: What you need before starting
- **Steps**: Exact commands and code to write
- **Outputs**: What to produce
- **Validation**: How to verify success

**Total Timeline**: 16 weeks (320 hours)

---

# STAGE 1: PROJECT INITIALIZATION (Week 1, Days 1-2)

## Goal
Set up project structure, dependencies, and development environment.

## Steps

### Step 1.1: Create Project Directory Structure
```bash
mkdir -p cdss-project/{src/{vision,nlp,fusion,reasoning,explainability,safety,api},data/{raw,processed,models},tests,configs,scripts,docs,monitoring}
cd cdss-project
```

### Step 1.2: Create requirements.txt
```text
# Core ML
torch>=2.0.0
transformers>=4.35.0
sentence-transformers>=2.2.0
open-clip-torch>=2.20.0

# Medical NLP
scispacy>=0.5.0
https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_md-0.5.1.tar.gz

# Vision
opencv-python>=4.8.0
Pillow>=10.0.0

# Knowledge Graph
neo4j>=5.0.0
py2neo>=2021.1

# Vector Database
weaviate-client>=3.25.0

# LLM & Agents
langchain>=0.1.0
langgraph>=0.0.20
ollama>=0.1.0

# API
fastapi>=0.104.0
uvicorn>=0.24.0
python-multipart>=0.0.6

# Database
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0
redis>=5.0.0

# Monitoring
prometheus-client>=0.19.0

# Explainability
shap>=0.43.0

# Utilities
pydantic>=2.5.0
python-dotenv>=1.0.0
loguru>=0.7.0
httpx>=0.25.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
```

### Step 1.3: Create .env.example
```env
# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=cdss
POSTGRES_USER=cdss_user
POSTGRES_PASSWORD=secure_password

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=secure_password

# Weaviate
WEAVIATE_URL=http://localhost:8080

# Redis
REDIS_URL=redis://localhost:6379

# LLM
OLLAMA_HOST=http://localhost:11434
LLM_MODEL=llama3.1:70b

# Security
SECRET_KEY=your-secret-key-here
ENCRYPTION_KEY=your-encryption-key

# Monitoring
PROMETHEUS_PORT=9090
```

### Step 1.4: Create docker-compose.yml
```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: cdss
      POSTGRES_USER: cdss_user
      POSTGRES_PASSWORD: secure_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  neo4j:
    image: neo4j:5-community
    environment:
      NEO4J_AUTH: neo4j/secure_password
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j_data:/data

  weaviate:
    image: semitechnologies/weaviate:latest
    ports:
      - "8080:8080"
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
    volumes:
      - weaviate_data:/var/lib/weaviate

  redis:
    image: redis:7
    ports:
      - "6379:6379"

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    depends_on:
      - prometheus

volumes:
  postgres_data:
  neo4j_data:
  weaviate_data:
```

### Step 1.5: Create src/__init__.py and base config
```python
# src/config.py
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # Database
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "cdss"
    postgres_user: str = "cdss_user"
    postgres_password: str = "secure_password"
    
    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "secure_password"
    
    # Weaviate
    weaviate_url: str = "http://localhost:8080"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # LLM
    ollama_host: str = "http://localhost:11434"
    llm_model: str = "llama3.1:70b"
    
    # Safety thresholds
    min_confidence_threshold: float = 0.55
    escalation_threshold: float = 0.70
    max_uncertainty: float = 0.15
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
```

## Validation
- [ ] All directories created
- [ ] requirements.txt exists
- [ ] docker-compose.yml starts all services: `docker-compose up -d`
- [ ] All containers healthy: `docker-compose ps`

---

# STAGE 2: DATA LAYER SETUP (Week 1, Days 3-5)

## Goal
Set up databases, download medical datasets, create data pipelines.

## Step 2.1: Database Models
```python
# src/database/models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Patient(Base):
    __tablename__ = "patients"
    
    id = Column(String, primary_key=True)
    age = Column(Integer)
    sex = Column(String(1))
    created_at = Column(DateTime, default=datetime.utcnow)
    # Encrypted fields
    encrypted_name = Column(Text)
    encrypted_contact = Column(Text)
    
    diagnoses = relationship("Diagnosis", back_populates="patient")
    
class Diagnosis(Base):
    __tablename__ = "diagnoses"
    
    id = Column(String, primary_key=True)
    patient_id = Column(String, ForeignKey("patients.id"))
    
    # Input data
    symptoms = Column(JSON)
    lab_results = Column(JSON)
    image_path = Column(String)
    
    # AI Output
    predicted_diagnosis = Column(String)
    confidence = Column(Float)
    confidence_interval_low = Column(Float)
    confidence_interval_high = Column(Float)
    differential_diagnoses = Column(JSON)
    explanation = Column(Text)
    
    # Safety flags
    conflict_detected = Column(Boolean, default=False)
    escalated_to_human = Column(Boolean, default=False)
    safety_alerts = Column(JSON)
    
    # Doctor verification
    doctor_id = Column(String)
    doctor_confirmed = Column(Boolean)
    actual_diagnosis = Column(String)
    doctor_notes = Column(Text)
    
    # Audit
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
    
    patient = relationship("Patient", back_populates="diagnoses")

class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user_id = Column(String)
    action = Column(String)  # READ, WRITE, DIAGNOSE, ESCALATE
    resource_type = Column(String)  # patient, diagnosis
    resource_id = Column(String)
    details = Column(JSON)
    ip_address = Column(String)
```

## Step 2.2: Neo4j Knowledge Graph Schema
```python
# src/knowledge_graph/schema.py
from py2neo import Graph, Node, Relationship

def create_medical_schema(graph: Graph):
    """Create medical knowledge graph schema"""
    
    # Clear existing (for dev only)
    # graph.delete_all()
    
    # Create constraints
    constraints = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Disease) REQUIRE d.icd10 IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Symptom) REQUIRE s.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Test) REQUIRE t.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Medication) REQUIRE m.name IS UNIQUE",
    ]
    
    for constraint in constraints:
        graph.run(constraint)
    
    # Create indexes
    indexes = [
        "CREATE INDEX IF NOT EXISTS FOR (d:Disease) ON (d.name)",
        "CREATE INDEX IF NOT EXISTS FOR (s:Symptom) ON (s.category)",
    ]
    
    for index in indexes:
        graph.run(index)

def populate_sample_diseases(graph: Graph):
    """Populate with sample disease data"""
    
    diseases = [
        {
            "icd10": "J18.9",
            "name": "Pneumonia, unspecified",
            "category": "respiratory",
            "symptoms": ["fever", "cough", "dyspnea", "chest_pain"],
            "tests": ["chest_xray", "cbc", "sputum_culture"],
            "severity": "moderate"
        },
        {
            "icd10": "I21.9",
            "name": "Acute myocardial infarction",
            "category": "cardiovascular",
            "symptoms": ["chest_pain", "dyspnea", "diaphoresis", "nausea"],
            "tests": ["ecg", "troponin", "cbc"],
            "severity": "critical"
        },
        {
            "icd10": "A15.0",
            "name": "Tuberculosis of lung",
            "category": "infectious",
            "symptoms": ["cough", "fever", "night_sweats", "weight_loss", "hemoptysis"],
            "tests": ["chest_xray", "sputum_afb", "genexpert"],
            "severity": "high"
        }
    ]
    
    for disease_data in diseases:
        # Create disease node
        disease = Node("Disease", 
                      icd10=disease_data["icd10"],
                      name=disease_data["name"],
                      category=disease_data["category"],
                      severity=disease_data["severity"])
        graph.merge(disease, "Disease", "icd10")
        
        # Create symptom relationships
        for symptom_name in disease_data["symptoms"]:
            symptom = Node("Symptom", name=symptom_name)
            graph.merge(symptom, "Symptom", "name")
            
            rel = Relationship(disease, "PRESENTS_WITH", symptom)
            graph.merge(rel)
        
        # Create test relationships
        for test_name in disease_data["tests"]:
            test = Node("Test", name=test_name)
            graph.merge(test, "Test", "name")
            
            rel = Relationship(disease, "DIAGNOSED_BY", test)
            graph.merge(rel)
```

## Step 2.3: Weaviate Vector Schema
```python
# src/vector_store/schema.py
import weaviate

def create_weaviate_schema(client: weaviate.Client):
    """Create Weaviate schema for patient case embeddings"""
    
    schema = {
        "classes": [
            {
                "class": "PatientCase",
                "description": "Embedded patient case for similarity search",
                "vectorizer": "none",  # We provide our own embeddings
                "properties": [
                    {"name": "case_id", "dataType": ["string"]},
                    {"name": "symptoms", "dataType": ["string[]"]},
                    {"name": "lab_summary", "dataType": ["text"]},
                    {"name": "image_findings", "dataType": ["string[]"]},
                    {"name": "diagnosis", "dataType": ["string"]},
                    {"name": "icd10", "dataType": ["string"]},
                    {"name": "outcome", "dataType": ["string"]},
                    {"name": "age_group", "dataType": ["string"]},
                    {"name": "sex", "dataType": ["string"]},
                ]
            },
            {
                "class": "MedicalDocument",
                "description": "Medical literature and guidelines",
                "vectorizer": "none",
                "properties": [
                    {"name": "doc_id", "dataType": ["string"]},
                    {"name": "title", "dataType": ["string"]},
                    {"name": "content", "dataType": ["text"]},
                    {"name": "source", "dataType": ["string"]},
                    {"name": "category", "dataType": ["string"]},
                ]
            }
        ]
    }
    
    # Delete existing if present
    for class_obj in schema["classes"]:
        try:
            client.schema.delete_class(class_obj["class"])
        except:
            pass
    
    # Create schema
    client.schema.create(schema)
```

## Step 2.4: Data Download Script
```python
# scripts/download_data.py
"""
Download public medical datasets.
Run: python scripts/download_data.py
"""

import os
import urllib.request
from pathlib import Path

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "nih_chestxray_sample": {
        "url": "https://nihcc.app.box.com/v/ChestXray-NIHCC/file/220660789610",
        "description": "NIH Chest X-ray sample (full dataset requires PhysioNet access)"
    },
    "synthea_sample": {
        "url": "https://synthetichealth.github.io/synthea-sample-data/downloads/synthea_sample_data_csv_apr2020.zip",
        "filename": "synthea_sample.zip"
    }
}

def download_file(url: str, filename: str):
    filepath = DATA_DIR / filename
    if not filepath.exists():
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"Saved to {filepath}")
    else:
        print(f"{filename} already exists")

if __name__ == "__main__":
    print("=== CDSS Data Download ===")
    print("\nNOTE: For MIMIC-CXR and NIH ChestX-ray14 full datasets:")
    print("1. Create account at https://physionet.org/")
    print("2. Complete required training")
    print("3. Request access to datasets")
    print("4. Download using physionet-build tools\n")
    
    # Download Synthea sample
    download_file(
        DATASETS["synthea_sample"]["url"],
        DATASETS["synthea_sample"]["filename"]
    )
```

## Validation
- [ ] PostgreSQL tables created: `docker exec -it cdss-project_postgres_1 psql -U cdss_user -d cdss -c "\dt"`
- [ ] Neo4j schema exists: Access http://localhost:7474
- [ ] Weaviate schema created: `curl http://localhost:8080/v1/schema`
- [ ] Sample data populated

---

# STAGE 2.5: DATA PIPELINE IMPLEMENTATION (Week 2)

## Goal
Download, process, and validate medical datasets with production-grade pipelines.

> **⚠️ NO STORAGE? Use API Approach Instead!**
> If you don't have 100GB+ storage, use the API streaming approach below.
> See: `CDSS_Datasets_API_Guide.md` for full details.

## OPTION A: API Streaming (No Download, < 5 GB)

### Step 2.5.0: HuggingFace Streaming Setup
```python
# scripts/stream_medical_data.py
"""
Stream medical datasets without downloading.
Requires: pip install datasets
"""
from datasets import load_dataset
from typing import Iterator, Dict
import numpy as np

class MedicalDataStreamer:
    """Stream medical datasets from HuggingFace Hub"""
    
    def __init__(self):
        self.xray_dataset = None
        self.text_dataset = None
    
    def stream_xrays(self, num_samples: int = 1000) -> Iterator[Dict]:
        """Stream chest X-rays without downloading full dataset"""
        dataset = load_dataset(
            "alkzar90/NIH-Chest-X-ray-dataset",
            split="train",
            streaming=True  # KEY: Uses ~0 storage
        )
        
        for i, sample in enumerate(dataset):
            if i >= num_samples:
                break
            
            yield {
                "image": np.array(sample["image"]),
                "labels": sample["labels"],
                "index": i
            }
    
    def stream_clinical_notes(self, num_samples: int = 1000) -> Iterator[Dict]:
        """Stream clinical text data"""
        # MTSamples is small enough to download
        dataset = load_dataset(
            "medical_dialog",
            split="train",
            streaming=True
        )
        
        for i, sample in enumerate(dataset):
            if i >= num_samples:
                break
            yield sample
    
    def get_sample_batch(self, batch_size: int = 32):
        """Get a batch of samples for training"""
        samples = list(self.stream_xrays(batch_size))
        images = np.stack([s["image"] for s in samples])
        labels = [s["labels"] for s in samples]
        return images, labels

# Usage
if __name__ == "__main__":
    streamer = MedicalDataStreamer()
    
    # Test streaming
    for i, sample in enumerate(streamer.stream_xrays(10)):
        print(f"Sample {i}: shape={sample['image'].shape}, labels={sample['labels']}")
```

### Small Downloadable Datasets (< 500 MB total)
```bash
# MTSamples: Medical transcriptions (20 MB)
pip install kaggle
kaggle datasets download -d tboyle10/medicaltranscriptions

# Synthea: Generate synthetic patients (100 MB)
git clone https://github.com/synthetichealth/synthea.git
cd synthea && ./gradlew build
./run_synthea -p 500 --exporter.csv.export true
```

---

## OPTION B: Full Download (92+ GB Storage Required)

### Step 2.5.1: PhysioNet Data Access Setup
```bash
# 1. Create PhysioNet account at https://physionet.org/
# 2. Complete CITI training (required for MIMIC access)
# 3. Sign data use agreement for:
#    - MIMIC-CXR: https://physionet.org/content/mimic-cxr/2.0.0/
#    - NIH ChestX-ray14: https://nihcc.app.box.com/v/ChestXray-NIHCC

# Install PhysioNet download tool
pip install wget

# For MIMIC-CXR (after access granted):
# wget -r -N -c -np --user YOUR_USERNAME --ask-password https://physionet.org/files/mimic-cxr/2.0.0/
```

## Step 2.5.2: NIH ChestX-ray14 Download Script
```python
# scripts/download_nih_chestxray.py
"""
Download NIH ChestX-ray14 dataset.
Total: 112,120 frontal X-ray images, ~45GB
"""
import os
import subprocess
from pathlib import Path
from loguru import logger

DATA_DIR = Path("data/raw/nih_chestxray")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# NIH dataset is hosted on Box - download links
NIH_LINKS = [
    "https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz",  # images_001
    "https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pez8x0fhxn.gz",  # images_002
    # ... Add all 12 image archive links
]

def download_nih_dataset():
    logger.info("Downloading NIH ChestX-ray14 dataset...")
    
    # Download metadata first
    metadata_url = "https://nihcc.box.com/shared/static/y67e47azzy8gbseu291uswez7d0y3t9d.csv"
    subprocess.run(["wget", "-O", str(DATA_DIR / "Data_Entry_2017_v2020.csv"), metadata_url])
    
    # Download images (12 archives, ~4GB each)
    for i, link in enumerate(NIH_LINKS, 1):
        output_file = DATA_DIR / f"images_{i:03d}.tar.gz"
        if not output_file.exists():
            logger.info(f"Downloading archive {i}/12...")
            subprocess.run(["wget", "-O", str(output_file), link])
            subprocess.run(["tar", "-xzf", str(output_file), "-C", str(DATA_DIR)])
        else:
            logger.info(f"Archive {i}/12 already exists, skipping")
    
    logger.info("NIH ChestX-ray14 download complete!")

if __name__ == "__main__":
    download_nih_dataset()
```

## Step 2.5.3: Data Processing Pipeline
```python
# scripts/process_xray_data.py
"""
Process raw X-ray images and metadata into training-ready format.
"""
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import json

class XRayDataProcessor:
    def __init__(self, raw_dir: str, processed_dir: str):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Disease label mapping
        self.disease_labels = [
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
            "Mass", "Nodule", "Pneumonia", "Pneumothorax",
            "Consolidation", "Edema", "Emphysema", "Fibrosis",
            "Pleural_Thickening", "Hernia"
        ]
    
    def load_metadata(self) -> pd.DataFrame:
        """Load and validate NIH metadata"""
        metadata_path = self.raw_dir / "Data_Entry_2017_v2020.csv"
        df = pd.read_csv(metadata_path)
        
        logger.info(f"Loaded {len(df)} records from metadata")
        
        # Validate required columns
        required = ["Image Index", "Finding Labels", "Patient Age", "Patient Gender"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        return df
    
    def preprocess_image(self, image_path: Path) -> np.ndarray:
        """Preprocess single X-ray image"""
        # Load grayscale
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError(f"Failed to load: {image_path}")
        
        # Resize to standard size
        img = cv2.resize(img, (224, 224))
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        
        return img
    
    def parse_labels(self, label_string: str) -> list:
        """Parse multi-label string to list"""
        if label_string == "No Finding":
            return []
        return [l.strip() for l in label_string.split("|")]
    
    def process_dataset(self, sample_size: int = None):
        """Process full dataset into train/val/test splits"""
        df = self.load_metadata()
        
        if sample_size:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)
            logger.info(f"Sampled {len(df)} records for processing")
        
        # Process each image
        processed_records = []
        failed = 0
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
            image_name = row["Image Index"]
            image_path = self.raw_dir / "images" / image_name
            
            if not image_path.exists():
                failed += 1
                continue
            
            try:
                # Preprocess image
                img = self.preprocess_image(image_path)
                
                # Save processed image
                output_path = self.processed_dir / "images" / image_name
                output_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(output_path.with_suffix(".npy"), img)
                
                # Record metadata
                processed_records.append({
                    "image_id": image_name,
                    "image_path": str(output_path.with_suffix(".npy")),
                    "labels": self.parse_labels(row["Finding Labels"]),
                    "age": int(row["Patient Age"]),
                    "sex": row["Patient Gender"],
                    "view_position": row.get("View Position", "PA")
                })
                
            except Exception as e:
                logger.warning(f"Failed to process {image_name}: {e}")
                failed += 1
        
        logger.info(f"Processed: {len(processed_records)}, Failed: {failed}")
        
        # Save metadata
        with open(self.processed_dir / "metadata.json", "w") as f:
            json.dump(processed_records, f, indent=2)
        
        # Create train/val/test splits (70/15/15)
        self._create_splits(processed_records)
    
    def _create_splits(self, records: list):
        """Create stratified train/val/test splits"""
        np.random.seed(42)
        np.random.shuffle(records)
        
        n = len(records)
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)
        
        splits = {
            "train": records[:train_end],
            "val": records[train_end:val_end],
            "test": records[val_end:]
        }
        
        for split_name, split_data in splits.items():
            path = self.processed_dir / f"{split_name}.json"
            with open(path, "w") as f:
                json.dump(split_data, f, indent=2)
            logger.info(f"{split_name}: {len(split_data)} samples")

if __name__ == "__main__":
    processor = XRayDataProcessor(
        raw_dir="data/raw/nih_chestxray",
        processed_dir="data/processed/nih_chestxray"
    )
    # Process sample for development (full dataset: remove sample_size)
    processor.process_dataset(sample_size=5000)
```

## Step 2.5.4: Data Validation
```python
# scripts/validate_data.py
"""
Validate processed data quality before training.
"""
import json
import numpy as np
from pathlib import Path
from loguru import logger

class DataValidator:
    def __init__(self, processed_dir: str):
        self.processed_dir = Path(processed_dir)
    
    def validate(self) -> dict:
        """Run all validation checks"""
        results = {
            "metadata_check": self._validate_metadata(),
            "image_check": self._validate_images(),
            "label_distribution": self._check_label_distribution(),
            "data_quality": self._check_data_quality()
        }
        
        all_passed = all(r["passed"] for r in results.values())
        results["overall_passed"] = all_passed
        
        return results
    
    def _validate_metadata(self) -> dict:
        """Check metadata file exists and is valid"""
        metadata_path = self.processed_dir / "metadata.json"
        
        if not metadata_path.exists():
            return {"passed": False, "error": "metadata.json not found"}
        
        with open(metadata_path) as f:
            data = json.load(f)
        
        required_fields = ["image_id", "image_path", "labels", "age", "sex"]
        for record in data[:10]:  # Check first 10
            missing = [f for f in required_fields if f not in record]
            if missing:
                return {"passed": False, "error": f"Missing fields: {missing}"}
        
        return {"passed": True, "record_count": len(data)}
    
    def _validate_images(self) -> dict:
        """Check image files exist and have correct shape"""
        with open(self.processed_dir / "metadata.json") as f:
            data = json.load(f)
        
        sample = data[:100]  # Check 100 random
        missing = 0
        wrong_shape = 0
        
        for record in sample:
            img_path = Path(record["image_path"])
            if not img_path.exists():
                missing += 1
                continue
            
            img = np.load(img_path)
            if img.shape != (224, 224):
                wrong_shape += 1
        
        passed = missing == 0 and wrong_shape == 0
        return {
            "passed": passed,
            "missing_images": missing,
            "wrong_shape": wrong_shape,
            "checked": len(sample)
        }
    
    def _check_label_distribution(self) -> dict:
        """Check for class imbalance"""
        with open(self.processed_dir / "metadata.json") as f:
            data = json.load(f)
        
        label_counts = {}
        for record in data:
            for label in record["labels"]:
                label_counts[label] = label_counts.get(label, 0) + 1
        
        # Check for severe imbalance
        if label_counts:
            max_count = max(label_counts.values())
            min_count = min(label_counts.values())
            imbalance_ratio = max_count / max(min_count, 1)
        else:
            imbalance_ratio = 1
        
        return {
            "passed": True,  # Info only
            "label_counts": label_counts,
            "imbalance_ratio": imbalance_ratio,
            "warning": "Severe imbalance" if imbalance_ratio > 100 else None
        }
    
    def _check_data_quality(self) -> dict:
        """Check for data quality issues"""
        with open(self.processed_dir / "metadata.json") as f:
            data = json.load(f)
        
        issues = []
        
        # Check age range
        ages = [r["age"] for r in data if r.get("age")]
        if min(ages) < 0 or max(ages) > 120:
            issues.append("Invalid age values detected")
        
        # Check sex values
        sexes = set(r["sex"] for r in data if r.get("sex"))
        valid_sexes = {"M", "F", "Male", "Female"}
        if not sexes.issubset(valid_sexes):
            issues.append(f"Invalid sex values: {sexes - valid_sexes}")
        
        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "total_records": len(data)
        }

if __name__ == "__main__":
    validator = DataValidator("data/processed/nih_chestxray")
    results = validator.validate()
    
    print("\n=== DATA VALIDATION RESULTS ===")
    for check, result in results.items():
        status = "✅" if result.get("passed", result) else "❌"
        print(f"{status} {check}: {result}")
```

## Validation
- [ ] NIH ChestX-ray14 metadata downloaded
- [ ] Sample images processed (5000 for dev)
- [ ] Train/val/test splits created
- [ ] Data validation passes all checks

# STAGE 3: VISION MODULE (Weeks 2-3)

## Goal
Implement medical image analysis with BiomedCLIP and safety checks.

## Step 3.1: Image Preprocessor
```python
# src/vision/preprocessor.py
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional
from loguru import logger

class ImagePreprocessor:
    """Preprocess medical images with quality checks"""
    
    def __init__(self):
        self.target_size = (224, 224)
        self.min_sharpness = 100.0  # Laplacian variance threshold
    
    def preprocess(self, image_path: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Preprocess image with quality validation.
        Returns: (processed_image, error_message)
        """
        try:
            # Load image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return None, "Failed to load image"
            
            # Quality check: Sharpness
            sharpness = cv2.Laplacian(image, cv2.CV_64F).var()
            if sharpness < self.min_sharpness:
                logger.warning(f"Low sharpness: {sharpness:.2f}")
                return None, f"Image too blurry (sharpness: {sharpness:.1f}). Please retake."
            
            # Normalize contrast (CLAHE)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            image = clahe.apply(image)
            
            # Resize
            image = cv2.resize(image, self.target_size)
            
            # Normalize to 0-1
            image = image.astype(np.float32) / 255.0
            
            # Convert to 3-channel for CLIP
            image = np.stack([image] * 3, axis=-1)
            
            return image, None
            
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            return None, f"Processing error: {str(e)}"
    
    def detect_artifacts(self, image: np.ndarray) -> list[str]:
        """Detect common image artifacts that may affect diagnosis"""
        artifacts = []
        
        # Check for metal artifacts (very bright spots)
        bright_pixels = np.sum(image > 0.95) / image.size
        if bright_pixels > 0.01:
            artifacts.append("metal_artifact_possible")
        
        # Check for motion blur
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        if laplacian.var() < 50:
            artifacts.append("motion_blur_detected")
        
        return artifacts
```

## Step 3.2: BiomedCLIP Analyzer
```python
# src/vision/biomedclip.py
import torch
import open_clip
from PIL import Image
import numpy as np
from typing import Dict, List
from loguru import logger

class BiomedCLIPAnalyzer:
    """Medical image analysis using BiomedCLIP"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading BiomedCLIP on {self.device}")
        
        self.model, self.preprocess, _ = open_clip.create_model_and_transforms(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = open_clip.get_tokenizer(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        
        # Medical finding prompts for chest X-rays
        self.chest_xray_findings = [
            "normal chest radiograph",
            "pneumonia with lung consolidation",
            "pulmonary edema",
            "pleural effusion",
            "lung nodule or mass",
            "cardiomegaly",
            "pneumothorax",
            "atelectasis",
            "tuberculosis pattern"
        ]
    
    def analyze_chest_xray(self, image: np.ndarray) -> Dict:
        """
        Analyze chest X-ray and return findings with confidence.
        
        Returns:
            {
                "findings": [{"finding": str, "confidence": float}, ...],
                "top_finding": str,
                "confidence": float,
                "needs_review": bool,
                "review_reason": str
            }
        """
        try:
            # Convert to PIL
            if isinstance(image, np.ndarray):
                image = Image.fromarray((image * 255).astype(np.uint8))
            
            # Preprocess
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Tokenize findings
            text_tokens = self.tokenizer(self.chest_xray_findings).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                text_features = self.model.encode_text(text_tokens)
                
                # Normalize
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                probs = similarity[0].cpu().numpy()
            
            # Build results
            findings = []
            for i, (finding, prob) in enumerate(zip(self.chest_xray_findings, probs)):
                findings.append({
                    "finding": finding,
                    "confidence": float(prob)
                })
            
            # Sort by confidence
            findings.sort(key=lambda x: x["confidence"], reverse=True)
            
            top_finding = findings[0]
            
            # Safety check: Low confidence
            needs_review = False
            review_reason = None
            
            if top_finding["confidence"] < 0.4:
                needs_review = True
                review_reason = "Low confidence - ambiguous image"
            
            # Safety check: Close second finding
            if len(findings) > 1:
                diff = findings[0]["confidence"] - findings[1]["confidence"]
                if diff < 0.1:
                    needs_review = True
                    review_reason = f"Uncertain between {findings[0]['finding']} and {findings[1]['finding']}"
            
            return {
                "findings": findings[:5],  # Top 5
                "top_finding": top_finding["finding"],
                "confidence": top_finding["confidence"],
                "needs_review": needs_review,
                "review_reason": review_reason
            }
            
        except Exception as e:
            logger.error(f"BiomedCLIP analysis error: {e}")
            return {
                "findings": [],
                "top_finding": "analysis_failed",
                "confidence": 0.0,
                "needs_review": True,
                "review_reason": f"Analysis failed: {str(e)}"
            }
    
    def get_embedding(self, image: np.ndarray) -> np.ndarray:
        """Get image embedding for similarity search"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray((image * 255).astype(np.uint8))
        
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model.encode_image(image_tensor)
            features /= features.norm(dim=-1, keepdim=True)
        
        return features[0].cpu().numpy()
```

## Step 3.3: Vision Module with Explainability
```python
# src/vision/explainer.py
import torch
import numpy as np
import cv2
from typing import Dict, Tuple

class GradCAMExplainer:
    """Generate visual explanations using GradCAM"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_heatmap(self, image_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """Generate GradCAM heatmap for target class"""
        self.model.eval()
        
        # Forward pass
        output = self.model(image_tensor)
        
        # Backward pass for target class
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Compute GradCAM
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Resize to image size
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        
        return cam
    
    def overlay_heatmap(self, image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        """Overlay heatmap on original image"""
        # Convert heatmap to colormap
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        
        # Blend with original
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        blended = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)
        
        return blended
```

## Validation
- [ ] BiomedCLIP loads successfully
- [ ] Test image analysis: Returns findings with confidences
- [ ] GradCAM heatmaps generate correctly
- [ ] Low-quality images rejected with message

---

# STAGE 4: NLP MODULE (Weeks 3-4)

## Goal
Extract symptoms from clinical notes using BioBERT with negation handling.

## Step 4.1: Clinical NLP Pipeline
```python
# src/nlp/clinical_nlp.py
import spacy
from scispacy.linking import EntityLinker
from typing import List, Dict
from loguru import logger

class ClinicalNLPPipeline:
    """Extract medical entities from clinical notes"""
    
    def __init__(self):
        logger.info("Loading clinical NLP models...")
        
        # Load SciSpacy model
        self.nlp = spacy.load("en_core_sci_md")
        
        # Add entity linker for UMLS
        self.nlp.add_pipe("scispacy_linker", config={
            "resolve_abbreviations": True,
            "linker_name": "umls"
        })
        
        # Medical abbreviations dictionary
        self.abbreviations = {
            "SOB": "shortness of breath",
            "CP": "chest pain",
            "N/V": "nausea and vomiting",
            "HTN": "hypertension",
            "DM": "diabetes mellitus",
            "CAD": "coronary artery disease",
            "CHF": "congestive heart failure",
            "COPD": "chronic obstructive pulmonary disease",
            "MI": "myocardial infarction",
            "CVA": "cerebrovascular accident",
            "DVT": "deep vein thrombosis",
            "PE": "pulmonary embolism",
            "UTI": "urinary tract infection",
            "BID": "twice daily",
            "TID": "three times daily",
            "PRN": "as needed",
        }
        
        # Negation patterns
        self.negation_cues = [
            "no", "not", "denies", "denied", "without", "negative for",
            "rules out", "ruled out", "absence of", "no evidence of",
            "never", "none", "free of"
        ]
    
    def expand_abbreviations(self, text: str) -> str:
        """Expand medical abbreviations"""
        for abbr, expansion in self.abbreviations.items():
            text = text.replace(f" {abbr} ", f" {expansion} ")
            text = text.replace(f" {abbr}.", f" {expansion}.")
            text = text.replace(f" {abbr},", f" {expansion},")
        return text
    
    def extract_symptoms(self, clinical_note: str) -> List[Dict]:
        """
        Extract symptoms with negation detection.
        
        Returns:
            [
                {
                    "symptom": "fever",
                    "canonical": "C0015967",  # UMLS CUI
                    "negated": False,
                    "context": "patient reports fever x2 days"
                },
                ...
            ]
        """
        # Expand abbreviations
        text = self.expand_abbreviations(clinical_note)
        
        # Process with NLP
        doc = self.nlp(text)
        
        symptoms = []
        
        for ent in doc.ents:
            # Check if symptom-related
            if self._is_symptom_entity(ent):
                # Check for negation
                negated = self._check_negation(doc, ent)
                
                # Get UMLS linking
                umls_cui = None
                if ent._.kb_ents:
                    umls_cui = ent._.kb_ents[0][0]  # Top CUI
                
                # Get context window
                start = max(0, ent.start - 5)
                end = min(len(doc), ent.end + 5)
                context = doc[start:end].text
                
                symptoms.append({
                    "symptom": ent.text.lower(),
                    "canonical": umls_cui,
                    "negated": negated,
                    "context": context,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char
                })
        
        return symptoms
    
    def _is_symptom_entity(self, ent) -> bool:
        """Check if entity is a symptom/finding"""
        symptom_types = ["DISEASE", "SIGN_SYMPTOM", "FINDING"]
        return ent.label_ in symptom_types or len(ent._.kb_ents) > 0
    
    def _check_negation(self, doc, ent) -> bool:
        """Check if entity is negated"""
        # Look at tokens before entity
        start = max(0, ent.start - 4)
        preceding_text = doc[start:ent.start].text.lower()
        
        for cue in self.negation_cues:
            if cue in preceding_text:
                return True
        
        return False
    
    def extract_vitals(self, text: str) -> Dict:
        """Extract vital signs from text"""
        import re
        
        vitals = {}
        
        # Temperature patterns
        temp_match = re.search(r'temp(?:erature)?[:\s]+(\d+\.?\d*)\s*[°]?[FC]?', text, re.I)
        if temp_match:
            vitals["temperature"] = float(temp_match.group(1))
        
        # Blood pressure
        bp_match = re.search(r'BP[:\s]+(\d+)/(\d+)', text, re.I)
        if bp_match:
            vitals["bp_systolic"] = int(bp_match.group(1))
            vitals["bp_diastolic"] = int(bp_match.group(2))
        
        # Heart rate
        hr_match = re.search(r'(?:HR|pulse)[:\s]+(\d+)', text, re.I)
        if hr_match:
            vitals["heart_rate"] = int(hr_match.group(1))
        
        # Oxygen saturation
        spo2_match = re.search(r'(?:SpO2|O2\s*sat)[:\s]+(\d+)%?', text, re.I)
        if spo2_match:
            vitals["spo2"] = int(spo2_match.group(1))
        
        return vitals
```

## Validation
- [ ] SciSpacy loads with UMLS linker
- [ ] Abbreviations expanded correctly
- [ ] Negation detected: "denies fever" → negated=True
- [ ] Vitals extracted from free text

---

**Continue to Part 2 for: Knowledge Graph Reasoning, LangGraph Agent, Explainability, Safety Layer, API, and Deployment**
