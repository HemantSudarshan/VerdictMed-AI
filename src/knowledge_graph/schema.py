"""
Medical Knowledge Graph Schema
Neo4j schema and initial data population for disease-symptom-test relationships.
"""

from py2neo import Graph, Node, Relationship
from loguru import logger
from typing import List, Dict


class MedicalKnowledgeGraphSchema:
    """Create and populate the medical knowledge graph in Neo4j"""
    
    def __init__(self, uri: str, user: str, password: str):
        """
        Initialize connection to Neo4j.
        
        Args:
            uri: Neo4j bolt URI (e.g., bolt://localhost:7687)
            user: Neo4j username
            password: Neo4j password
        """
        self.graph = Graph(uri, auth=(user, password))
        logger.info(f"Connected to Neo4j at {uri}")
    
    def create_schema(self):
        """Create constraints and indexes for the medical knowledge graph"""
        
        # Constraints for unique identifiers
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Disease) REQUIRE d.icd10 IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Symptom) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Test) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Medication) REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Condition) REQUIRE c.name IS UNIQUE",
        ]
        
        for constraint in constraints:
            try:
                self.graph.run(constraint)
                logger.debug(f"Created constraint: {constraint[:50]}...")
            except Exception as e:
                logger.warning(f"Constraint may already exist: {e}")
        
        # Indexes for faster lookups
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (d:Disease) ON (d.name)",
            "CREATE INDEX IF NOT EXISTS FOR (d:Disease) ON (d.category)",
            "CREATE INDEX IF NOT EXISTS FOR (s:Symptom) ON (s.category)",
            "CREATE INDEX IF NOT EXISTS FOR (d:Disease) ON (d.severity)",
        ]
        
        for index in indexes:
            try:
                self.graph.run(index)
                logger.debug(f"Created index: {index[:50]}...")
            except Exception as e:
                logger.warning(f"Index may already exist: {e}")
        
        logger.info("Neo4j schema created successfully")
    
    def populate_sample_diseases(self):
        """
        Populate knowledge graph with sample disease data.
        Covers common respiratory, cardiovascular, and infectious diseases.
        """
        
        diseases = [
            # Respiratory
            {
                "icd10": "J18.9",
                "name": "Pneumonia, unspecified",
                "category": "respiratory",
                "symptoms": ["fever", "cough", "dyspnea", "chest_pain", "sputum_production"],
                "tests": ["chest_xray", "cbc", "sputum_culture", "procalcitonin"],
                "severity": "moderate",
                "description": "Infection causing inflammation in the air sacs of the lungs"
            },
            {
                "icd10": "A15.0",
                "name": "Tuberculosis of lung",
                "category": "infectious",
                "symptoms": ["cough", "fever", "night_sweats", "weight_loss", "hemoptysis", "fatigue"],
                "tests": ["chest_xray", "sputum_afb", "genexpert", "tuberculin_test"],
                "severity": "high",
                "description": "Bacterial infection caused by Mycobacterium tuberculosis"
            },
            {
                "icd10": "J44.1",
                "name": "Chronic obstructive pulmonary disease with acute exacerbation",
                "category": "respiratory",
                "symptoms": ["dyspnea", "cough", "wheezing", "sputum_production"],
                "tests": ["chest_xray", "spirometry", "arterial_blood_gas"],
                "severity": "moderate",
                "description": "Chronic inflammatory lung disease with acute worsening"
            },
            
            # Cardiovascular
            {
                "icd10": "I21.9",
                "name": "Acute myocardial infarction",
                "category": "cardiovascular",
                "symptoms": ["chest_pain", "dyspnea", "diaphoresis", "nausea", "left_arm_pain"],
                "tests": ["ecg", "troponin", "cbc", "bnp", "echocardiogram"],
                "severity": "critical",
                "description": "Heart attack due to blocked coronary artery"
            },
            {
                "icd10": "I50.9",
                "name": "Heart failure, unspecified",
                "category": "cardiovascular",
                "symptoms": ["dyspnea", "edema", "fatigue", "orthopnea", "paroxysmal_nocturnal_dyspnea"],
                "tests": ["chest_xray", "bnp", "echocardiogram", "ecg"],
                "severity": "high",
                "description": "Heart's inability to pump blood effectively"
            },
            {
                "icd10": "I26.9",
                "name": "Pulmonary embolism",
                "category": "cardiovascular",
                "symptoms": ["dyspnea", "chest_pain", "tachycardia", "hemoptysis", "syncope"],
                "tests": ["d_dimer", "ct_pulmonary_angiography", "ecg", "chest_xray"],
                "severity": "critical",
                "description": "Blood clot blocking pulmonary arteries"
            },
            
            # Infectious
            {
                "icd10": "A41.9",
                "name": "Sepsis, unspecified",
                "category": "infectious",
                "symptoms": ["fever", "tachycardia", "hypotension", "altered_mental_status", "tachypnea"],
                "tests": ["blood_culture", "cbc", "lactate", "procalcitonin", "crp"],
                "severity": "critical",
                "description": "Life-threatening organ dysfunction due to infection"
            },
            {
                "icd10": "G03.9",
                "name": "Meningitis, unspecified",
                "category": "neurological",
                "symptoms": ["headache", "fever", "neck_stiffness", "photophobia", "altered_mental_status"],
                "tests": ["lumbar_puncture", "cbc", "blood_culture", "ct_head"],
                "severity": "critical",
                "description": "Inflammation of the membranes surrounding brain and spinal cord"
            },
            
            # Others
            {
                "icd10": "K35.9",
                "name": "Acute appendicitis",
                "category": "gastrointestinal",
                "symptoms": ["abdominal_pain", "nausea", "vomiting", "fever", "loss_of_appetite"],
                "tests": ["cbc", "ct_abdomen", "ultrasound_abdomen"],
                "severity": "moderate",
                "description": "Inflammation of the appendix"
            },
            {
                "icd10": "N10",
                "name": "Acute pyelonephritis",
                "category": "urological",
                "symptoms": ["fever", "flank_pain", "dysuria", "frequency", "nausea"],
                "tests": ["urinalysis", "urine_culture", "cbc", "creatinine"],
                "severity": "moderate",
                "description": "Kidney infection usually from bacterial spread"
            }
        ]
        
        for disease_data in diseases:
            self._create_disease_with_relationships(disease_data)
        
        logger.info(f"Populated {len(diseases)} diseases with relationships")
    
    def _create_disease_with_relationships(self, disease_data: Dict):
        """Create a disease node and all its relationships"""
        
        # Create disease node
        disease = Node(
            "Disease",
            icd10=disease_data["icd10"],
            name=disease_data["name"],
            category=disease_data["category"],
            severity=disease_data["severity"],
            description=disease_data.get("description", "")
        )
        self.graph.merge(disease, "Disease", "icd10")
        
        # Create symptom relationships
        for symptom_name in disease_data["symptoms"]:
            symptom = Node("Symptom", name=symptom_name)
            self.graph.merge(symptom, "Symptom", "name")
            
            rel = Relationship(disease, "PRESENTS_WITH", symptom)
            self.graph.merge(rel)
        
        # Create test relationships
        for test_name in disease_data["tests"]:
            test = Node("Test", name=test_name)
            self.graph.merge(test, "Test", "name")
            
            rel = Relationship(disease, "DIAGNOSED_BY", test)
            self.graph.merge(rel)
    
    def populate_medications_and_contraindications(self):
        """Add medications and their contraindications"""
        
        medications = [
            {
                "name": "aspirin",
                "contraindicated_for": ["peptic_ulcer", "bleeding_disorder", "aspirin_allergy"]
            },
            {
                "name": "metformin",
                "contraindicated_for": ["kidney_disease", "liver_disease", "heart_failure"]
            },
            {
                "name": "lisinopril",
                "contraindicated_for": ["pregnancy", "angioedema_history", "hyperkalemia"]
            },
            {
                "name": "warfarin",
                "contraindicated_for": ["active_bleeding", "pregnancy", "severe_hypertension"]
            },
            {
                "name": "amoxicillin",
                "contraindicated_for": ["penicillin_allergy"]
            }
        ]
        
        for med_data in medications:
            med = Node("Medication", name=med_data["name"])
            self.graph.merge(med, "Medication", "name")
            
            for condition_name in med_data["contraindicated_for"]:
                condition = Node("Condition", name=condition_name)
                self.graph.merge(condition, "Condition", "name")
                
                rel = Relationship(med, "CONTRAINDICATED_FOR", condition)
                self.graph.merge(rel)
        
        logger.info(f"Populated {len(medications)} medications with contraindications")
    
    def setup_complete(self):
        """Run complete schema setup"""
        self.create_schema()
        self.populate_sample_diseases()
        self.populate_medications_and_contraindications()
        logger.info("Knowledge graph setup complete!")


# Convenience function
def init_knowledge_graph(uri: str, user: str, password: str):
    """Initialize and populate the medical knowledge graph"""
    schema = MedicalKnowledgeGraphSchema(uri, user, password)
    schema.setup_complete()
    return schema
