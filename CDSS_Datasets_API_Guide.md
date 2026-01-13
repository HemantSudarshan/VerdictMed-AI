# 🔌 CDSS DATASETS - API & LOW-STORAGE GUIDE

## For Developers Without 100GB Storage

**Your Situation:** No storage for large datasets  
**Solution:** Use APIs + Small Samples + Cloud Processing  
**Total Storage Needed:** < 5 GB  
**Cost:** ₹0 (100% free)

---

# PART 1: API-BASED DATASETS (NO DOWNLOAD)

## 1. 🩻 Medical Images via HuggingFace API

### Option A: Streaming from HuggingFace Hub
```python
# No download - streams directly from cloud
from datasets import load_dataset

# NIH Chest X-ray (streams, no storage)
dataset = load_dataset(
    "alkzar90/NIH-Chest-X-ray-dataset",
    split="train",
    streaming=True  # KEY: Stream mode uses ~0 storage
)

# Process one image at a time
for i, sample in enumerate(dataset):
    image = sample["image"]
    label = sample["labels"]
    
    # Process image here
    process_image(image, label)
    
    if i >= 1000:  # Stop after 1000 samples
        break
```

### HuggingFace Medical Datasets (All Streamable)
| Dataset | HF Path | Size | Use Case |
|---------|---------|------|----------|
| NIH ChestX-ray | `alkzar90/NIH-Chest-X-ray-dataset` | Stream | Vision Module |
| CheXpert | `stanfordaimi/CheXpert` | Stream | Vision Module |
| PadChest | `BIMCV-CSUSP/PADCHEST` | Stream | Vision Module |
| RSNA Pneumonia | `rsna/pneumonia-detection` | Stream | Vision Module |

```python
# Example: Stream CheXpert
from datasets import load_dataset

chexpert = load_dataset(
    "stanfordaimi/CheXpert",
    split="train",
    streaming=True
)

for sample in chexpert.take(100):  # Get 100 samples
    print(sample["path"], sample["labels"])
```

---

## 2. 📝 Clinical Text via APIs

### Option A: PubMed API (Free, No Login)
```python
# PubMed E-utilities API - 35 million medical papers
import requests
from typing import List, Dict

class PubMedAPI:
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    def search(self, query: str, max_results: int = 100) -> List[str]:
        """Search PubMed for articles"""
        url = f"{self.BASE_URL}/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json"
        }
        response = requests.get(url, params=params)
        data = response.json()
        return data["esearchresult"]["idlist"]
    
    def get_abstracts(self, pmids: List[str]) -> List[Dict]:
        """Get abstracts for given PMIDs"""
        url = f"{self.BASE_URL}/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml"
        }
        response = requests.get(url, params=params)
        # Parse XML and extract abstracts
        return self._parse_abstracts(response.text)

# Usage
api = PubMedAPI()
pmids = api.search("pneumonia symptoms diagnosis", max_results=50)
abstracts = api.get_abstracts(pmids)
```

### Option B: BioPortal API (Medical Ontologies)
```python
# Free API for medical terminologies
import requests

class BioPortalAPI:
    BASE_URL = "https://data.bioontology.org"
    API_KEY = "YOUR_FREE_API_KEY"  # Get free at bioportal.bioontology.org
    
    def search_concept(self, term: str):
        """Search medical concepts"""
        url = f"{self.BASE_URL}/search"
        params = {
            "q": term,
            "apikey": self.API_KEY,
            "ontologies": "SNOMEDCT,ICD10CM,LOINC"
        }
        response = requests.get(url, params=params)
        return response.json()["collection"]
    
    def get_synonyms(self, concept_id: str):
        """Get synonyms for a medical concept"""
        url = f"{self.BASE_URL}/ontologies/SNOMEDCT/classes/{concept_id}"
        params = {"apikey": self.API_KEY}
        response = requests.get(url, params=params)
        return response.json().get("synonym", [])

# Usage
api = BioPortalAPI()
concepts = api.search_concept("pneumonia")
print(concepts[0]["prefLabel"], concepts[0]["@id"])
```

### Option C: UMLS REST API
```python
# Unified Medical Language System API
import requests

class UMLSAPI:
    AUTH_URL = "https://utslogin.nlm.nih.gov/cas/v1/api-key"
    BASE_URL = "https://uts-ws.nlm.nih.gov/rest"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.tgt = self._get_ticket_granting_ticket()
    
    def _get_ticket_granting_ticket(self):
        response = requests.post(self.AUTH_URL, data={"apikey": self.api_key})
        return response.headers["Location"]
    
    def get_service_ticket(self):
        response = requests.post(self.tgt, data={"service": "http://umlsks.nlm.nih.gov"})
        return response.text
    
    def search_cui(self, term: str):
        """Search for UMLS Concept Unique Identifier"""
        ticket = self.get_service_ticket()
        url = f"{self.BASE_URL}/search/current"
        params = {"string": term, "ticket": ticket}
        response = requests.get(url, params=params)
        return response.json()["result"]["results"]

# Get free API key at: https://uts.nlm.nih.gov/uts/
```

---

## 3. 🧠 Medical Knowledge Graph APIs

### SNOMED CT Browser API (Free)
```python
# SNOMED CT - Medical terminology standard
import requests

class SNOMEDApi:
    BASE_URL = "https://browser.ihtsdotools.org/snowstorm/snomed-ct"
    
    def search(self, term: str, limit: int = 10):
        """Search SNOMED CT concepts"""
        url = f"{self.BASE_URL}/MAIN/concepts"
        params = {
            "term": term,
            "activeFilter": True,
            "limit": limit
        }
        response = requests.get(url, params=params)
        return response.json()["items"]
    
    def get_relationships(self, concept_id: str):
        """Get relationships for a concept"""
        url = f"{self.BASE_URL}/MAIN/concepts/{concept_id}/relationships"
        response = requests.get(url)
        return response.json()["items"]

# Usage
api = SNOMEDApi()
results = api.search("pneumonia")
for r in results:
    print(f"{r['conceptId']}: {r['fsn']['term']}")
```

### Disease Ontology API
```python
# Free API for disease information
import requests

def get_disease_info(disease_name: str):
    url = "https://www.disease-ontology.org/api/search"
    params = {"query": disease_name}
    response = requests.get(url, params=params)
    return response.json()
```

---

# PART 2: SMALL DOWNLOADABLE DATASETS (< 5 GB Total)

## Essential Small Datasets

### 1. MTSamples - Medical Transcriptions (20 MB)
```python
# Download from Kaggle
# pip install kaggle
# kaggle datasets download -d tboyle10/medicaltranscriptions

import pandas as pd

# Load and use
df = pd.read_csv("mtsamples.csv")
print(f"Total samples: {len(df)}")
print(df.columns)  # ['description', 'medical_specialty', 'sample_name', 'transcription', 'keywords']
```

### 2. COVID-19 X-rays (300 MB)
```python
# Smaller dataset, perfect for development
from datasets import load_dataset

dataset = load_dataset("keremberke/chest-xray-classification", split="train")
print(f"Total images: {len(dataset)}")
```

### 3. Synthea - Synthetic Patients (Generate Any Size)
```bash
# Generate 500 patients (takes ~5 minutes, ~100 MB)
git clone https://github.com/synthetichealth/synthea.git
cd synthea
./gradlew build
./run_synthea -p 500 --exporter.csv.export true
```

```python
# Load generated data
import pandas as pd

patients = pd.read_csv("output/csv/patients.csv")
conditions = pd.read_csv("output/csv/conditions.csv")
observations = pd.read_csv("output/csv/observations.csv")
```

### 4. ICD-10 Codes (10 MB)
```python
# Download ICD-10 codes directly
import pandas as pd

# Official CMS ICD-10 codes
url = "https://www.cms.gov/files/zip/2024-icd-10-cm-codes-file.zip"
# Download and extract, then:
icd10 = pd.read_csv("icd10cm_codes_2024.txt", sep="\t", header=None)
```

### 5. DrugBank Subset (Free via API)
```python
# Drug information API
import requests

def search_drugs(query: str):
    url = f"https://go.drugbank.com/unearth/q?utf8=✓&searcher=drugs&query={query}"
    # Note: Full API requires registration, but web search is free
```

---

# PART 3: CLOUD-BASED PROCESSING (ZERO LOCAL STORAGE)

## Option A: Google Colab (Free GPU + 100GB Storage)
```python
# Run in Google Colab - free GPU and storage
# 1. Go to colab.research.google.com
# 2. Create new notebook
# 3. Enable GPU: Runtime > Change runtime type > GPU

# Mount Google Drive for persistent storage
from google.colab import drive
drive.mount('/content/drive')

# Download dataset to Colab (temporary, deleted after session)
!wget -q https://nihcc.box.com/shared/static/...
!unzip -q dataset.zip

# Process data
# ... your training code ...

# Save only models/results to Drive (small files)
model.save('/content/drive/MyDrive/cdss_models/model.pth')
```

## Option B: Kaggle Kernels (Free GPU + 50GB Storage)
```python
# Run on Kaggle - free GPU and access to datasets
# 1. Go to kaggle.com/code
# 2. Create new notebook
# 3. Add dataset from sidebar

# Kaggle has many medical datasets pre-loaded
import pandas as pd

# Access dataset directly (already mounted)
df = pd.read_csv("/kaggle/input/nih-chest-xrays/Data_Entry_2017.csv")
```

## Option C: HuggingFace Spaces (Free Deployment)
```python
# Deploy your CDSS directly to HuggingFace Spaces
# No local storage needed - runs entirely in cloud
# Just upload your model and Streamlit app
```

---

# PART 4: RECOMMENDED APPROACH FOR NO-STORAGE SETUP

## Week 1: Development Setup (2 GB total)
```bash
# 1. Download small datasets
kaggle datasets download -d tboyle10/medicaltranscriptions  # 20 MB

# 2. Generate synthetic data
./run_synthea -p 100  # 50 MB

# 3. Download sample X-rays (subset)
# Use HuggingFace streaming for most images
```

## Week 2-3: Model Training (Google Colab)
```python
# Train on Colab with free GPU
# Stream large datasets, save only models

from datasets import load_dataset

# Stream training data
train_data = load_dataset("alkzar90/NIH-Chest-X-ray-dataset", streaming=True)

# Train model
for batch in train_data.take(10000):  # 10k samples
    train_step(batch)

# Save small model checkpoint
torch.save(model.state_dict(), "model.pth")  # ~500 MB
```

## Week 4+: Inference Setup (Local)
```python
# Load only the trained model (small)
# Use APIs for runtime data
# No large datasets needed

model = load_model("model.pth")  # 500 MB

# Stream test data via API
test_data = load_dataset("...", streaming=True, split="test")
```

---

# 📊 COMPARISON: DOWNLOAD vs API

| Aspect | Full Download | API Streaming |
|--------|---------------|---------------|
| Storage | 92-500 GB | < 5 GB |
| Time to Start | 24-48 hours | 2 hours |
| Internet | One-time download | Always needed |
| Cost | External HDD ₹3,500 | ₹0 |
| Flexibility | All data local | Stream as needed |
| Best For | Production | Development/Portfolio |

---

# ✅ QUICK START CHECKLIST

- [ ] Create HuggingFace account (free): https://huggingface.co/
- [ ] Create Kaggle account (free): https://kaggle.com/
- [ ] Get UMLS API key (free): https://uts.nlm.nih.gov/
- [ ] Get BioPortal API key (free): https://bioportal.bioontology.org/
- [ ] Download MTSamples (20 MB)
- [ ] Generate Synthea data (100 patients)
- [ ] Test streaming with HuggingFace datasets

---

# 🔗 ALL FREE API LINKS

| Service | URL | Registration |
|---------|-----|--------------|
| HuggingFace Datasets | https://huggingface.co/datasets | Free account |
| PubMed API | https://www.ncbi.nlm.nih.gov/home/develop/api/ | No registration |
| UMLS API | https://uts.nlm.nih.gov/uts/ | Free account |
| BioPortal API | https://bioportal.bioontology.org/ | Free account |
| SNOMED Browser | https://browser.ihtsdotools.org/ | No registration |
| Disease Ontology | https://disease-ontology.org/ | No registration |
| Google Colab | https://colab.research.google.com/ | Google account |
| Kaggle Kernels | https://www.kaggle.com/code | Free account |

---

**Bottom Line: You can build CDSS with < 5 GB storage using APIs and cloud compute. Start today! 🚀**
