"""
Vector Store Service
Service for similar case retrieval using Weaviate vector store.
"""

from typing import List, Dict, Optional
from loguru import logger


class VectorStoreService:
    """
    Service for storing and retrieving similar medical cases.
    Uses Weaviate for vector similarity search.
    """
    
    def __init__(self, url: str = "http://localhost:8080"):
        """
        Initialize vector store service.
        
        Args:
            url: Weaviate server URL
        """
        self.url = url
        self._client = None
        self._connected = False
    
    @property
    def client(self):
        """Lazy load Weaviate client"""
        if self._client is None:
            try:
                import weaviate
                self._client = weaviate.Client(self.url)
                self._connected = True
                logger.info(f"Connected to Weaviate at {self.url}")
            except Exception as e:
                logger.warning(f"Weaviate not available: {e}")
                self._connected = False
        return self._client
    
    def is_connected(self) -> bool:
        """Check if connected to Weaviate"""
        try:
            if self.client:
                return self.client.is_ready()
        except:
            pass
        return False
    
    def store_case(
        self,
        case_id: str,
        symptoms: List[str],
        diagnosis: str,
        icd10: Optional[str] = None,
        lab_summary: Optional[str] = None,
        image_findings: Optional[List[str]] = None,
        outcome: Optional[str] = None,
        severity: str = "unknown",
        embedding: Optional[List[float]] = None
    ) -> bool:
        """
        Store a patient case in vector store.
        
        Args:
            case_id: Unique case identifier
            symptoms: List of symptoms
            diagnosis: Final diagnosis
            icd10: ICD-10 code
            lab_summary: Lab results summary
            image_findings: X-ray/CT findings
            outcome: Patient outcome
            severity: Case severity
            embedding: Pre-computed embedding vector
            
        Returns:
            True if successful
        """
        if not self.is_connected():
            logger.warning("Vector store not available")
            return False
        
        try:
            data_object = {
                "case_id": case_id,
                "symptoms": symptoms,
                "diagnosis": diagnosis,
                "icd10": icd10 or "",
                "lab_summary": lab_summary or "",
                "image_findings": image_findings or [],
                "outcome": outcome or "",
                "severity": severity
            }
            
            # Add to Weaviate
            if embedding:
                self.client.data_object.create(
                    data_object,
                    "PatientCase",
                    vector=embedding
                )
            else:
                self.client.data_object.create(
                    data_object,
                    "PatientCase"
                )
            
            logger.info(f"Stored case {case_id} in vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store case: {e}")
            return False
    
    def retrieve_similar_cases(
        self,
        symptoms: List[str],
        limit: int = 5,
        min_certainty: float = 0.7
    ) -> List[Dict]:
        """
        Retrieve similar cases based on symptoms.
        
        Args:
            symptoms: List of symptoms to match
            limit: Maximum cases to return
            min_certainty: Minimum similarity threshold
            
        Returns:
            List of similar cases with similarity scores
        """
        if not self.is_connected():
            logger.warning("Vector store not available, returning empty results")
            return []
        
        try:
            # Create symptom query text
            symptom_text = ", ".join(symptoms)
            
            # Query Weaviate
            result = self.client.query.get(
                "PatientCase",
                ["case_id", "symptoms", "diagnosis", "icd10", "outcome", "severity"]
            ).with_near_text({
                "concepts": [symptom_text],
                "certainty": min_certainty
            }).with_additional(["certainty"]).with_limit(limit).do()
            
            cases = result.get("data", {}).get("Get", {}).get("PatientCase", [])
            
            # Format results
            similar = []
            for case in cases:
                similar.append({
                    "case_id": case.get("case_id"),
                    "symptoms": case.get("symptoms", []),
                    "diagnosis": case.get("diagnosis"),
                    "icd10": case.get("icd10"),
                    "outcome": case.get("outcome"),
                    "severity": case.get("severity"),
                    "similarity": case.get("_additional", {}).get("certainty", 0.0)
                })
            
            logger.info(f"Found {len(similar)} similar cases")
            return similar
            
        except Exception as e:
            logger.error(f"Similar case retrieval failed: {e}")
            return []
    
    def get_case_count(self) -> int:
        """Get total number of stored cases"""
        if not self.is_connected():
            return 0
        
        try:
            result = self.client.query.aggregate("PatientCase").with_meta_count().do()
            return result["data"]["Aggregate"]["PatientCase"][0]["meta"]["count"]
        except Exception as e:
            logger.error(f"Failed to get case count: {e}")
            return 0


# Singleton instance
_vector_service = None


def get_vector_service(url: str = "http://localhost:8080") -> VectorStoreService:
    """Get singleton vector store service"""
    global _vector_service
    if _vector_service is None:
        _vector_service = VectorStoreService(url)
    return _vector_service
