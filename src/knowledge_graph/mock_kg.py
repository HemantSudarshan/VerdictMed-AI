"""
Mock Knowledge Graph
In-memory disease-symptom database for offline operation.
Works without Neo4j Docker container.
"""

from typing import List, Dict, Optional
from loguru import logger


class MockKnowledgeGraph:
    """
    In-memory knowledge graph with 50+ disease-symptom mappings.
    Provides same interface as Neo4j-based MedicalKnowledgeGraph.
    """
    
    # Disease database with symptoms, ICD-10 codes, and severity
    DISEASES = {
        # Respiratory
        "Pneumonia": {
            "icd10": "J18.9",
            "category": "respiratory",
            "severity": "moderate",
            "symptoms": ["fever", "cough", "shortness of breath", "chest pain", "sputum", "fatigue", "chills"]
        },
        "Tuberculosis": {
            "icd10": "A15.9",
            "category": "respiratory",
            "severity": "high",
            "symptoms": ["chronic cough", "night sweats", "weight loss", "fever", "hemoptysis", "fatigue"]
        },
        "Asthma": {
            "icd10": "J45.9",
            "category": "respiratory",
            "severity": "moderate",
            "symptoms": ["wheezing", "shortness of breath", "chest tightness", "cough", "breathing difficulty"]
        },
        "COPD": {
            "icd10": "J44.9",
            "category": "respiratory",
            "severity": "moderate",
            "symptoms": ["shortness of breath", "chronic cough", "sputum", "wheezing", "fatigue"]
        },
        "Bronchitis": {
            "icd10": "J40",
            "category": "respiratory",
            "severity": "low",
            "symptoms": ["cough", "sputum", "chest discomfort", "fatigue", "low fever"]
        },
        "Common Cold": {
            "icd10": "J00",
            "category": "respiratory",
            "severity": "low",
            "symptoms": ["runny nose", "sore throat", "cough", "congestion", "sneezing"]
        },
        "Influenza": {
            "icd10": "J11.1",
            "category": "respiratory",
            "severity": "moderate",
            "symptoms": ["fever", "body aches", "fatigue", "cough", "headache", "chills"]
        },
        
        # Cardiovascular - CRITICAL
        "Myocardial Infarction": {
            "icd10": "I21.9",
            "category": "cardiovascular",
            "severity": "critical",
            "symptoms": ["chest pain", "arm pain", "shortness of breath", "sweating", "nausea", "jaw pain", "diaphoresis"]
        },
        "Heart Failure": {
            "icd10": "I50.9",
            "category": "cardiovascular",
            "severity": "high",
            "symptoms": ["shortness of breath", "fatigue", "swelling", "edema", "weight gain", "orthopnea"]
        },
        "Hypertension": {
            "icd10": "I10",
            "category": "cardiovascular",
            "severity": "moderate",
            "symptoms": ["headache", "dizziness", "blurred vision", "chest pain", "fatigue"]
        },
        "Atrial Fibrillation": {
            "icd10": "I48.91",
            "category": "cardiovascular",
            "severity": "high",
            "symptoms": ["palpitations", "shortness of breath", "fatigue", "dizziness", "chest discomfort"]
        },
        "Pulmonary Embolism": {
            "icd10": "I26.99",
            "category": "cardiovascular",
            "severity": "critical",
            "symptoms": ["sudden shortness of breath", "chest pain", "cough", "leg swelling", "tachycardia", "hemoptysis"]
        },
        
        # Neurological - CRITICAL
        "Stroke": {
            "icd10": "I63.9",
            "category": "neurological",
            "severity": "critical",
            "symptoms": ["sudden weakness", "facial droop", "slurred speech", "confusion", "severe headache", "vision loss", "numbness"]
        },
        "Meningitis": {
            "icd10": "G03.9",
            "category": "neurological",
            "severity": "critical",
            "symptoms": ["severe headache", "stiff neck", "fever", "photophobia", "confusion", "nausea", "rash"]
        },
        "Migraine": {
            "icd10": "G43.909",
            "category": "neurological",
            "severity": "low",
            "symptoms": ["headache", "nausea", "light sensitivity", "aura", "throbbing pain"]
        },
        "Epilepsy": {
            "icd10": "G40.909",
            "category": "neurological",
            "severity": "moderate",
            "symptoms": ["seizures", "confusion", "loss of consciousness", "staring", "jerking movements"]
        },
        
        # Infectious
        "Sepsis": {
            "icd10": "A41.9",
            "category": "infectious",
            "severity": "critical",
            "symptoms": ["fever", "tachycardia", "hypotension", "confusion", "rapid breathing", "chills", "elevated wbc"]
        },
        "COVID-19": {
            "icd10": "U07.1",
            "category": "infectious",
            "severity": "moderate",
            "symptoms": ["fever", "cough", "fatigue", "loss of taste", "loss of smell", "shortness of breath", "body aches"]
        },
        "Urinary Tract Infection": {
            "icd10": "N39.0",
            "category": "infectious",
            "severity": "low",
            "symptoms": ["dysuria", "frequency", "urgency", "suprapubic pain", "cloudy urine", "hematuria"]
        },
        "Cellulitis": {
            "icd10": "L03.90",
            "category": "infectious",
            "severity": "moderate",
            "symptoms": ["redness", "swelling", "warmth", "pain", "fever", "skin tenderness"]
        },
        
        # Gastrointestinal
        "Gastritis": {
            "icd10": "K29.70",
            "category": "gastrointestinal",
            "severity": "low",
            "symptoms": ["epigastric pain", "nausea", "bloating", "indigestion", "loss of appetite"]
        },
        "Appendicitis": {
            "icd10": "K35.80",
            "category": "gastrointestinal",
            "severity": "high",
            "symptoms": ["right lower quadrant pain", "nausea", "vomiting", "fever", "loss of appetite", "rebound tenderness"]
        },
        "Pancreatitis": {
            "icd10": "K85.9",
            "category": "gastrointestinal",
            "severity": "high",
            "symptoms": ["severe abdominal pain", "nausea", "vomiting", "fever", "tachycardia", "back pain"]
        },
        "Cholecystitis": {
            "icd10": "K81.9",
            "category": "gastrointestinal",
            "severity": "moderate",
            "symptoms": ["right upper quadrant pain", "nausea", "vomiting", "fever", "murphy sign"]
        },
        "GERD": {
            "icd10": "K21.0",
            "category": "gastrointestinal",
            "severity": "low",
            "symptoms": ["heartburn", "regurgitation", "chest pain", "difficulty swallowing", "chronic cough"]
        },
        
        # Endocrine
        "Diabetes Type 2": {
            "icd10": "E11.9",
            "category": "endocrine",
            "severity": "moderate",
            "symptoms": ["polyuria", "polydipsia", "fatigue", "blurred vision", "slow healing", "weight loss"]
        },
        "Diabetic Ketoacidosis": {
            "icd10": "E11.10",
            "category": "endocrine",
            "severity": "critical",
            "symptoms": ["nausea", "vomiting", "abdominal pain", "confusion", "fruity breath", "rapid breathing"]
        },
        "Hypothyroidism": {
            "icd10": "E03.9",
            "category": "endocrine",
            "severity": "low",
            "symptoms": ["fatigue", "weight gain", "cold intolerance", "constipation", "dry skin", "depression"]
        },
        "Hyperthyroidism": {
            "icd10": "E05.90",
            "category": "endocrine",
            "severity": "moderate",
            "symptoms": ["weight loss", "rapid heartbeat", "anxiety", "tremor", "heat intolerance", "sweating"]
        },
        
        # Musculoskeletal
        "Rheumatoid Arthritis": {
            "icd10": "M06.9",
            "category": "musculoskeletal",
            "severity": "moderate",
            "symptoms": ["joint pain", "joint swelling", "morning stiffness", "fatigue", "joint deformity"]
        },
        "Osteoarthritis": {
            "icd10": "M19.90",
            "category": "musculoskeletal",
            "severity": "low",
            "symptoms": ["joint pain", "stiffness", "decreased range of motion", "crepitus", "joint swelling"]
        },
        "Gout": {
            "icd10": "M10.9",
            "category": "musculoskeletal",
            "severity": "moderate",
            "symptoms": ["sudden joint pain", "swelling", "redness", "warmth", "big toe pain"]
        },
        
        # Renal
        "Acute Kidney Injury": {
            "icd10": "N17.9",
            "category": "renal",
            "severity": "high",
            "symptoms": ["decreased urine output", "swelling", "fatigue", "confusion", "nausea", "shortness of breath"]
        },
        "Kidney Stones": {
            "icd10": "N20.0",
            "category": "renal",
            "severity": "moderate",
            "symptoms": ["severe flank pain", "hematuria", "nausea", "vomiting", "urinary frequency"]
        },
        
        # Psychiatric
        "Major Depression": {
            "icd10": "F32.9",
            "category": "psychiatric",
            "severity": "moderate",
            "symptoms": ["depressed mood", "loss of interest", "fatigue", "sleep changes", "appetite changes", "concentration difficulty"]
        },
        "Generalized Anxiety": {
            "icd10": "F41.1",
            "category": "psychiatric",
            "severity": "low",
            "symptoms": ["excessive worry", "restlessness", "fatigue", "difficulty concentrating", "muscle tension", "sleep disturbance"]
        },
        
        # Hematologic
        "Anemia": {
            "icd10": "D64.9",
            "category": "hematologic",
            "severity": "low",
            "symptoms": ["fatigue", "weakness", "pale skin", "shortness of breath", "dizziness", "cold hands"]
        },
        "Deep Vein Thrombosis": {
            "icd10": "I82.40",
            "category": "hematologic",
            "severity": "high",
            "symptoms": ["leg swelling", "leg pain", "warmth", "redness", "tenderness"]
        },
        
        # Dermatologic
        "Eczema": {
            "icd10": "L30.9",
            "category": "dermatologic",
            "severity": "low",
            "symptoms": ["itching", "rash", "dry skin", "redness", "skin thickening"]
        },
        "Psoriasis": {
            "icd10": "L40.9",
            "category": "dermatologic",
            "severity": "low",
            "symptoms": ["scaly patches", "itching", "dry skin", "joint pain", "nail changes"]
        },
        
        # Allergic
        "Anaphylaxis": {
            "icd10": "T78.2",
            "category": "allergic",
            "severity": "critical",
            "symptoms": ["difficulty breathing", "swelling", "hives", "rapid pulse", "dizziness", "nausea", "hypotension"]
        },
        "Allergic Rhinitis": {
            "icd10": "J30.9",
            "category": "allergic",
            "severity": "low",
            "symptoms": ["sneezing", "runny nose", "itchy eyes", "congestion", "postnasal drip"]
        },
    }
    
    # Symptom synonyms for better matching
    SYMPTOM_SYNONYMS = {
        "sob": "shortness of breath",
        "dyspnea": "shortness of breath",
        "chest tightness": "chest pain",
        "diaphoresis": "sweating",
        "sweats": "sweating",
        "cva": "stroke",
        "mi": "myocardial infarction",
        "heart attack": "myocardial infarction",
        "pe": "pulmonary embolism",
        "uti": "urinary tract infection",
        "headache": "headache",
        "ha": "headache",
        "n/v": "nausea",
        "nausea/vomiting": "nausea",
        "htn": "hypertension",
        "dm": "diabetes",
        "copd": "chronic obstructive pulmonary disease",
    }
    
    def __init__(self):
        logger.info("MockKnowledgeGraph initialized with {} diseases".format(len(self.DISEASES)))
    
    def _normalize_symptom(self, symptom: str) -> str:
        """Normalize symptom text for matching"""
        symptom = symptom.lower().strip()
        return self.SYMPTOM_SYNONYMS.get(symptom, symptom)
    
    def find_diseases_by_symptoms(
        self, 
        symptoms: List[str], 
        limit: int = 10
    ) -> List[Dict]:
        """
        Find diseases matching the given symptoms.
        
        Args:
            symptoms: List of symptom strings
            limit: Maximum number of results
            
        Returns:
            List of disease matches with scores
        """
        # Normalize input symptoms
        normalized_symptoms = [self._normalize_symptom(s) for s in symptoms]
        symptom_set = set(normalized_symptoms)
        
        matches = []
        
        for disease_name, disease_data in self.DISEASES.items():
            disease_symptoms = set(s.lower() for s in disease_data["symptoms"])
            
            # Calculate match score
            matched_symptoms = symptom_set & disease_symptoms
            
            # Also check for partial matches
            for input_sym in symptom_set:
                for disease_sym in disease_symptoms:
                    if input_sym in disease_sym or disease_sym in input_sym:
                        matched_symptoms.add(disease_sym)
            
            if matched_symptoms:
                match_ratio = len(matched_symptoms) / len(disease_symptoms)
                
                matches.append({
                    "disease": disease_name,
                    "icd10": disease_data["icd10"],
                    "category": disease_data["category"],
                    "severity": disease_data["severity"],
                    "matched_symptoms": list(matched_symptoms),
                    "match_ratio": round(match_ratio, 3),
                    "total_symptoms": len(disease_symptoms)
                })
        
        # Sort by match ratio (descending)
        matches.sort(key=lambda x: x["match_ratio"], reverse=True)
        
        return matches[:limit]
    
    def get_disease_info(self, disease_name: str) -> Optional[Dict]:
        """Get full info for a specific disease"""
        return self.DISEASES.get(disease_name)
    
    def get_differential_diagnosis(
        self, 
        symptoms: List[str],
        exclude_severity: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Generate differential diagnosis list.
        
        Args:
            symptoms: List of symptom strings
            exclude_severity: Severity levels to exclude
            
        Returns:
            Ranked list of possible diagnoses
        """
        matches = self.find_diseases_by_symptoms(symptoms, limit=20)
        
        if exclude_severity:
            matches = [m for m in matches if m["severity"] not in exclude_severity]
        
        return matches[:5]
    
    def check_critical_conditions(self, symptoms: List[str]) -> List[Dict]:
        """
        Check if symptoms match any critical conditions.
        Always prioritize these in diagnosis.
        
        Returns:
            List of matched critical conditions
        """
        matches = self.find_diseases_by_symptoms(symptoms, limit=20)
        critical = [m for m in matches if m["severity"] == "critical"]
        return critical


# Convenience function
def get_knowledge_graph() -> MockKnowledgeGraph:
    """Get a MockKnowledgeGraph instance"""
    return MockKnowledgeGraph()


if __name__ == "__main__":
    # Test the mock KG
    kg = MockKnowledgeGraph()
    
    print("Testing symptom search...")
    results = kg.find_diseases_by_symptoms(["fever", "cough", "shortness of breath"])
    for r in results[:5]:
        print(f"  {r['disease']}: {r['match_ratio']:.0%} - {r['severity']}")
    
    print("\nTesting critical conditions...")
    critical = kg.check_critical_conditions(["chest pain", "sweating", "arm pain"])
    for c in critical:
        print(f"  ⚠️ {c['disease']}: {c['severity']}")
