"""
Weaviate Vector Store Schema
Schema definitions for patient case embeddings and medical documents.
"""

import weaviate
from loguru import logger
from typing import Optional


class VectorStoreSchema:
    """Create and manage Weaviate schema for CDSS"""
    
    def __init__(self, url: str):
        """
        Initialize Weaviate client.
        
        Args:
            url: Weaviate server URL (e.g., http://localhost:8080)
        """
        self.client = weaviate.Client(url)
        logger.info(f"Connected to Weaviate at {url}")
    
    def create_schema(self):
        """Create all required Weaviate classes"""
        
        schema = {
            "classes": [
                {
                    "class": "PatientCase",
                    "description": "Embedded patient case for similarity search",
                    "vectorizer": "none",  # We provide our own embeddings
                    "properties": [
                        {"name": "case_id", "dataType": ["string"], "description": "Unique case identifier"},
                        {"name": "symptoms", "dataType": ["string[]"], "description": "List of symptoms"},
                        {"name": "lab_summary", "dataType": ["text"], "description": "Lab results summary"},
                        {"name": "image_findings", "dataType": ["string[]"], "description": "X-ray/CT findings"},
                        {"name": "diagnosis", "dataType": ["string"], "description": "Final diagnosis"},
                        {"name": "icd10", "dataType": ["string"], "description": "ICD-10 code"},
                        {"name": "outcome", "dataType": ["string"], "description": "Patient outcome"},
                        {"name": "age_group", "dataType": ["string"], "description": "Age bracket"},
                        {"name": "sex", "dataType": ["string"], "description": "Patient sex"},
                        {"name": "severity", "dataType": ["string"], "description": "Case severity"},
                        {"name": "timestamp", "dataType": ["date"], "description": "When case was recorded"},
                    ]
                },
                {
                    "class": "MedicalDocument",
                    "description": "Medical literature, guidelines, and reference documents",
                    "vectorizer": "none",
                    "properties": [
                        {"name": "doc_id", "dataType": ["string"], "description": "Document identifier"},
                        {"name": "title", "dataType": ["string"], "description": "Document title"},
                        {"name": "content", "dataType": ["text"], "description": "Full text content"},
                        {"name": "source", "dataType": ["string"], "description": "Source (e.g., PubMed, UpToDate)"},
                        {"name": "category", "dataType": ["string"], "description": "Medical category"},
                        {"name": "publication_date", "dataType": ["date"], "description": "When published"},
                        {"name": "authors", "dataType": ["string[]"], "description": "Author list"},
                    ]
                },
                {
                    "class": "ClinicalGuideline",
                    "description": "Treatment protocols and clinical guidelines",
                    "vectorizer": "none",
                    "properties": [
                        {"name": "guideline_id", "dataType": ["string"], "description": "Guideline identifier"},
                        {"name": "title", "dataType": ["string"], "description": "Guideline title"},
                        {"name": "condition", "dataType": ["string"], "description": "Target condition"},
                        {"name": "recommendations", "dataType": ["text"], "description": "Key recommendations"},
                        {"name": "source_org", "dataType": ["string"], "description": "Issuing organization"},
                        {"name": "version", "dataType": ["string"], "description": "Guideline version"},
                        {"name": "last_updated", "dataType": ["date"], "description": "Last update date"},
                    ]
                }
            ]
        }
        
        # Delete existing classes (careful in production!)
        for class_obj in schema["classes"]:
            class_name = class_obj["class"]
            try:
                if self.client.schema.exists(class_name):
                    logger.warning(f"Deleting existing class: {class_name}")
                    self.client.schema.delete_class(class_name)
            except Exception as e:
                logger.debug(f"Class {class_name} doesn't exist or error: {e}")
        
        # Create new schema
        self.client.schema.create(schema)
        logger.info(f"Created {len(schema['classes'])} Weaviate classes")
    
    def verify_schema(self) -> bool:
        """Verify that all required classes exist"""
        required_classes = ["PatientCase", "MedicalDocument", "ClinicalGuideline"]
        
        for class_name in required_classes:
            if not self.client.schema.exists(class_name):
                logger.error(f"Missing required class: {class_name}")
                return False
        
        logger.info("All Weaviate classes verified")
        return True
    
    def get_class_count(self, class_name: str) -> int:
        """Get count of objects in a class"""
        result = self.client.query.aggregate(class_name).with_meta_count().do()
        return result["data"]["Aggregate"][class_name][0]["meta"]["count"]


def init_vector_store(url: str) -> VectorStoreSchema:
    """Initialize and verify vector store schema"""
    schema = VectorStoreSchema(url)
    schema.create_schema()
    return schema
