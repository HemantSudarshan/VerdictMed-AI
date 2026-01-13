"""
Neo4j Knowledge Graph Service
Real Neo4j integration for production use.
Falls back to MockKnowledgeGraph when Neo4j is unavailable.
"""

from typing import List, Dict, Optional
from loguru import logger
import os

# Try to import neo4j driver
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logger.warning("neo4j driver not installed. Using mock KG.")

from .mock_kg import MockKnowledgeGraph


class Neo4jKnowledgeGraph:
    """
    Neo4j-based Medical Knowledge Graph.
    Connects to Neo4j database for disease-symptom queries.
    """
    
    def __init__(
        self, 
        uri: str = None, 
        user: str = None, 
        password: str = None
    ):
        """
        Initialize Neo4j connection.
        
        Args:
            uri: Neo4j connection URI (default: from env)
            user: Neo4j username (default: from env)
            password: Neo4j password (default: from env)
        """
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        
        self._driver = None
        self._connected = False
        self._mock = MockKnowledgeGraph()
        
        self._connect()
    
    def _connect(self):
        """Establish Neo4j connection"""
        if not NEO4J_AVAILABLE:
            logger.warning("Neo4j driver not available, using mock")
            return
        
        try:
            self._driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.user, self.password)
            )
            # Test connection
            with self._driver.session() as session:
                session.run("RETURN 1")
            self._connected = True
            logger.info(f"Connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.warning(f"Neo4j connection failed: {e}. Using mock KG.")
            self._connected = False
    
    def close(self):
        """Close Neo4j connection"""
        if self._driver:
            self._driver.close()
    
    def find_diseases_by_symptoms(
        self, 
        symptoms: List[str], 
        limit: int = 10
    ) -> List[Dict]:
        """
        Find diseases matching symptoms.
        
        Args:
            symptoms: List of symptom strings
            limit: Maximum results
            
        Returns:
            List of disease matches with confidence scores
        """
        if not self._connected:
            return self._mock.find_diseases_by_symptoms(symptoms, limit)
        
        try:
            with self._driver.session() as session:
                query = """
                UNWIND $symptoms AS symptom
                MATCH (s:Symptom)-[r:INDICATES]->(d:Disease)
                WHERE toLower(s.name) CONTAINS toLower(symptom)
                WITH d, COUNT(DISTINCT s) as matchCount, COLLECT(s.name) as matchedSymptoms
                MATCH (d)<-[:INDICATES]-(allSymptoms:Symptom)
                WITH d, matchCount, matchedSymptoms, COUNT(allSymptoms) as totalSymptoms
                RETURN d.name as disease,
                       d.icd10 as icd10,
                       d.severity as severity,
                       d.category as category,
                       matchedSymptoms,
                       matchCount,
                       totalSymptoms,
                       toFloat(matchCount) / totalSymptoms as match_ratio
                ORDER BY match_ratio DESC
                LIMIT $limit
                """
                result = session.run(query, symptoms=symptoms, limit=limit)
                
                matches = []
                for record in result:
                    matches.append({
                        "disease": record["disease"],
                        "icd10": record["icd10"],
                        "severity": record["severity"],
                        "category": record["category"],
                        "matched_symptoms": record["matchedSymptoms"],
                        "match_ratio": record["match_ratio"],
                        "total_symptoms": record["totalSymptoms"]
                    })
                
                return matches
                
        except Exception as e:
            logger.error(f"Neo4j query failed: {e}. Falling back to mock.")
            return self._mock.find_diseases_by_symptoms(symptoms, limit)
    
    def get_disease_info(self, disease_name: str) -> Optional[Dict]:
        """Get detailed disease information"""
        if not self._connected:
            return self._mock.get_disease_info(disease_name)
        
        try:
            with self._driver.session() as session:
                query = """
                MATCH (d:Disease {name: $name})
                OPTIONAL MATCH (s:Symptom)-[:INDICATES]->(d)
                RETURN d.name as name,
                       d.icd10 as icd10,
                       d.severity as severity,
                       d.category as category,
                       COLLECT(s.name) as symptoms
                """
                result = session.run(query, name=disease_name)
                record = result.single()
                
                if record:
                    return {
                        "name": record["name"],
                        "icd10": record["icd10"],
                        "severity": record["severity"],
                        "category": record["category"],
                        "symptoms": record["symptoms"]
                    }
                return None
                
        except Exception as e:
            logger.error(f"Neo4j query failed: {e}")
            return self._mock.get_disease_info(disease_name)
    
    def check_critical_conditions(self, symptoms: List[str]) -> List[Dict]:
        """Check for critical conditions matching symptoms"""
        if not self._connected:
            return self._mock.check_critical_conditions(symptoms)
        
        try:
            matches = self.find_diseases_by_symptoms(symptoms, limit=20)
            critical = [m for m in matches if m.get("severity") == "critical"]
            return critical
        except Exception as e:
            logger.error(f"Critical condition check failed: {e}")
            return self._mock.check_critical_conditions(symptoms)
    
    def get_differential_diagnosis(
        self, 
        symptoms: List[str],
        exclude_severity: Optional[List[str]] = None
    ) -> List[Dict]:
        """Generate differential diagnosis list"""
        if not self._connected:
            return self._mock.get_differential_diagnosis(symptoms, exclude_severity)
        
        matches = self.find_diseases_by_symptoms(symptoms, limit=20)
        
        if exclude_severity:
            matches = [m for m in matches if m.get("severity") not in exclude_severity]
        
        return matches[:5]
    
    @property
    def is_connected(self) -> bool:
        """Check if Neo4j is connected"""
        return self._connected


# Singleton instance
_kg_instance = None


def get_knowledge_graph(use_neo4j: bool = True) -> Neo4jKnowledgeGraph:
    """
    Get knowledge graph instance.
    
    Args:
        use_neo4j: Whether to try Neo4j (falls back to mock if unavailable)
        
    Returns:
        Knowledge graph instance
    """
    global _kg_instance
    
    if _kg_instance is None:
        if use_neo4j:
            _kg_instance = Neo4jKnowledgeGraph()
        else:
            _kg_instance = MockKnowledgeGraph()
    
    return _kg_instance


if __name__ == "__main__":
    # Test the service
    kg = get_knowledge_graph()
    
    print(f"Connected to Neo4j: {kg.is_connected if hasattr(kg, 'is_connected') else 'Mock mode'}")
    
    results = kg.find_diseases_by_symptoms(["fever", "cough", "shortness of breath"])
    print(f"\nFound {len(results)} matches for fever, cough, shortness of breath:")
    for r in results[:3]:
        print(f"  - {r['disease']}: {r['match_ratio']:.0%}")
