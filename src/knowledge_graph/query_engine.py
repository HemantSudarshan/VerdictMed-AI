"""
Medical Knowledge Graph Query Engine
Query Neo4j for symptom-to-disease reasoning and diagnostic support.
"""

from py2neo import Graph
from typing import List, Dict, Optional
from loguru import logger


class MedicalKnowledgeGraph:
    """Query medical knowledge graph for diagnostic reasoning"""
    
    def __init__(self, uri: str, user: str, password: str):
        """
        Initialize connection to Neo4j knowledge graph.
        
        Args:
            uri: Neo4j bolt URI
            user: Neo4j username
            password: Neo4j password
        """
        self.graph = Graph(uri, auth=(user, password))
        logger.info(f"Connected to medical knowledge graph at {uri}")
    
    def find_diseases_by_symptoms(
        self, 
        symptoms: List[str], 
        limit: int = 10
    ) -> List[Dict]:
        """
        Find diseases that match given symptoms.
        
        Uses symptom overlap and match ratio to rank diseases.
        
        Args:
            symptoms: List of symptom names
            limit: Maximum diseases to return
            
        Returns:
            List of disease dicts with match scores, sorted by relevance
        """
        if not symptoms:
            return []
        
        query = """
        UNWIND $symptoms AS symptom_name
        MATCH (d:Disease)-[r:PRESENTS_WITH]->(s:Symptom)
        WHERE toLower(s.name) = toLower(symptom_name)
        WITH d, COUNT(DISTINCT s) as symptom_matches, COLLECT(s.name) as matched_symptoms
        MATCH (d)-[:PRESENTS_WITH]->(all_symptoms:Symptom)
        WITH d, symptom_matches, matched_symptoms, COUNT(all_symptoms) as total_symptoms
        RETURN d.name as disease,
               d.icd10 as icd10,
               d.severity as severity,
               d.category as category,
               d.description as description,
               symptom_matches,
               total_symptoms,
               toFloat(symptom_matches) / toFloat(total_symptoms) as match_ratio,
               matched_symptoms
        ORDER BY symptom_matches DESC, match_ratio DESC
        LIMIT $limit
        """
        
        results = self.graph.run(
            query, 
            symptoms=[s.lower().replace(" ", "_") for s in symptoms], 
            limit=limit
        ).data()
        
        logger.debug(f"Found {len(results)} diseases for symptoms: {symptoms}")
        return results
    
    def get_disease_details(self, icd10: str) -> Optional[Dict]:
        """
        Get full disease information including symptoms, tests, and treatments.
        
        Args:
            icd10: ICD-10 code of the disease
            
        Returns:
            Dict with disease details or None if not found
        """
        query = """
        MATCH (d:Disease {icd10: $icd10})
        OPTIONAL MATCH (d)-[:PRESENTS_WITH]->(s:Symptom)
        OPTIONAL MATCH (d)-[:DIAGNOSED_BY]->(t:Test)
        OPTIONAL MATCH (d)-[:TREATED_BY]->(m:Medication)
        RETURN d.name as name,
               d.icd10 as icd10,
               d.category as category,
               d.severity as severity,
               d.description as description,
               COLLECT(DISTINCT s.name) as symptoms,
               COLLECT(DISTINCT t.name) as diagnostic_tests,
               COLLECT(DISTINCT m.name) as treatments
        """
        
        result = self.graph.run(query, icd10=icd10).data()
        return result[0] if result else None
    
    def find_differential_diagnoses(
        self, 
        symptoms: List[str], 
        exclude: List[str] = None
    ) -> List[Dict]:
        """
        Find alternative diagnoses for differential diagnosis.
        
        Args:
            symptoms: List of symptom names
            exclude: ICD-10 codes to exclude (e.g., primary diagnosis)
            
        Returns:
            List of alternative diagnoses
        """
        exclude = exclude or []
        
        query = """
        UNWIND $symptoms AS symptom_name
        MATCH (d:Disease)-[:PRESENTS_WITH]->(s:Symptom)
        WHERE toLower(s.name) = toLower(symptom_name)
        AND NOT d.icd10 IN $exclude
        WITH d, COUNT(DISTINCT s) as matches, COLLECT(s.name) as matched_symptoms
        RETURN d.name as disease,
               d.icd10 as icd10,
               d.severity as severity,
               d.category as category,
               matches,
               matched_symptoms
        ORDER BY matches DESC
        LIMIT 5
        """
        
        return self.graph.run(
            query, 
            symptoms=[s.lower().replace(" ", "_") for s in symptoms], 
            exclude=exclude
        ).data()
    
    def check_contraindications(
        self, 
        medications: List[str], 
        patient_conditions: List[str]
    ) -> List[Dict]:
        """
        Check for drug contraindications based on patient conditions.
        
        Args:
            medications: List of medication names to check
            patient_conditions: List of patient's existing conditions
            
        Returns:
            List of contraindications found
        """
        if not medications or not patient_conditions:
            return []
        
        query = """
        UNWIND $medications AS med_name
        MATCH (m:Medication)-[:CONTRAINDICATED_FOR]->(c:Condition)
        WHERE toLower(m.name) = toLower(med_name)
        AND toLower(c.name) IN $conditions
        RETURN m.name as medication,
               c.name as contraindicated_condition,
               "DO NOT USE - Contraindicated" as warning
        """
        
        results = self.graph.run(
            query, 
            medications=medications,
            conditions=[c.lower() for c in patient_conditions]
        ).data()
        
        if results:
            logger.warning(f"Found {len(results)} contraindications!")
        
        return results
    
    def get_recommended_tests(
        self, 
        symptoms: List[str]
    ) -> List[Dict]:
        """
        Get recommended diagnostic tests based on symptoms.
        
        Args:
            symptoms: List of symptom names
            
        Returns:
            List of tests with their associated diseases
        """
        query = """
        UNWIND $symptoms AS symptom_name
        MATCH (d:Disease)-[:PRESENTS_WITH]->(s:Symptom)
        WHERE toLower(s.name) = toLower(symptom_name)
        MATCH (d)-[:DIAGNOSED_BY]->(t:Test)
        WITH t, COLLECT(DISTINCT d.name) as diseases, COUNT(DISTINCT d) as disease_count
        RETURN t.name as test,
               diseases,
               disease_count
        ORDER BY disease_count DESC
        """
        
        return self.graph.run(
            query, 
            symptoms=[s.lower().replace(" ", "_") for s in symptoms]
        ).data()
    
    def get_symptom_statistics(self) -> Dict:
        """Get statistics about symptoms in the knowledge graph"""
        query = """
        MATCH (s:Symptom)<-[:PRESENTS_WITH]-(d:Disease)
        WITH s.name as symptom, COUNT(d) as disease_count
        RETURN symptom, disease_count
        ORDER BY disease_count DESC
        """
        
        results = self.graph.run(query).data()
        return {r["symptom"]: r["disease_count"] for r in results}


# Convenience function
def get_knowledge_graph(uri: str, user: str, password: str) -> MedicalKnowledgeGraph:
    """Get a configured knowledge graph instance"""
    return MedicalKnowledgeGraph(uri, user, password)
