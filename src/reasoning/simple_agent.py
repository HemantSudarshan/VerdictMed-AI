"""
Simple Diagnostic Agent (Async / Caching Optimized)
A simplified diagnostic agent that orchestrates NLP, Vision, and Safety modules.
Now fully async and cached.
"""

from typing import Dict, Optional, List
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from loguru import logger

from src.nlp.clinical_nlp import ClinicalNLPPipeline
from src.nlp.lab_processor import get_lab_processor, LabProcessor
from src.safety.validator import SafetyValidator
from src.cache.redis_service import get_redis_service


class SimpleDiagnosticAgent:
    """
    Async diagnostic reasoning agent.
    Orchestrates symptom extraction, knowledge graph queries, and safety validation.
    Optimized with Redis Caching and ThreadPoolExecutor for CPU-bound tasks.
    """
    
    def __init__(self, config):
        """
        Initialize the diagnostic agent.
        
        Args:
            config: Configuration object with settings
        """
        self.config = config
        
        # Initialize components
        # Note: NLP startup can be slow, might want to defer loading in production
        self.nlp = ClinicalNLPPipeline(load_models=False)  
        self.safety = SafetyValidator(config)
        self.cache = get_redis_service()
        self.lab_processor = get_lab_processor()
        
        # Executor for CPU-bound tasks (NLP, Image Analysis, Network calls if sync)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Lazy loaded components
        self._kg = None
        self._vision = None
        
        logger.info("SimpleDiagnosticAgent (Async) initialized")
    
    @property
    def kg(self):
        """Lazy load knowledge graph"""
        if self._kg is None:
            try:
                from src.knowledge_graph.query_engine import MedicalKnowledgeGraph
                self._kg = MedicalKnowledgeGraph(
                    uri=self.config.neo4j_uri,
                    user=self.config.neo4j_user,
                    password=self.config.neo4j_password
                )
            except Exception as e:
                logger.warning(f"Knowledge graph not available: {e}")
        return self._kg
    
    @property
    def vision(self):
        """Lazy load vision analyzer"""
        if self._vision is None:
            try:
                from src.vision.biomedclip import BiomedCLIPAnalyzer
                self._vision = BiomedCLIPAnalyzer()
            except Exception as e:
                logger.warning(f"Vision analyzer not available: {e}")
        return self._vision
    
    async def run(self, patient_data: Dict) -> Dict:
        """
        Execute the diagnostic workflow (Async).
        
        Args:
            patient_data: Dict with symptoms, optional image_path, labs, etc.
            
        Returns:
            Complete diagnostic result
        """
        # 1. Check Cache
        symptoms_text = patient_data.get("symptoms", "")
        if not patient_data.get("image_path"): # Only cache text-only queries for now
            cache_key = self.cache.generate_cache_key("diagnosis:text", {"symptoms": symptoms_text})
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.info("Diagnosis cache hit")
                return cached_result
        
        start_time = time.time()
        
        result = {
            "patient_id": patient_data.get("patient_id", ""),
            "extracted_symptoms": [],
            "extracted_vitals": {},
            "lab_findings": {},
            "image_findings": {},
            "gradcam_explanation": None,
            "kg_diseases": [],
            "similar_cases": [],
            "differential_diagnoses": [],
            "primary_diagnosis": {},
            "confidence": 0.0,
            "modality_scores": {},
            "conflicts_detected": [],
            "safety_alerts": [],
            "needs_escalation": False,
            "escalation_reason": None,
            "explanation": ""
        }
        
        # 2. Parallel Execution: NLP & Image Analysis
        loop = asyncio.get_event_loop()
        
        # Task: NLP Analysis (CPU bound)
        nlp_task = loop.run_in_executor(
            self.executor, 
            self._run_nlp, 
            symptoms_text
        )
        
        # Task: Image Analysis (CPU/GPU bound)
        image_task = None
        image_path = patient_data.get("image_path")
        if image_path:
            image_task = loop.run_in_executor(
                self.executor,
                self._run_image_analysis,
                image_path
            )
            
        # Wait for NLP
        nlp_result = await nlp_task
        if nlp_result:
            result["extracted_symptoms"] = nlp_result["symptoms"]
            result["extracted_vitals"] = nlp_result["vitals"]
            result["symptoms_text"] = symptoms_text
            
        # Wait for Image (if any)
        if image_task:
            result["image_findings"] = await image_task
        
        # 2b. Lab Processing (if provided)
        lab_results = patient_data.get("labs") or patient_data.get("lab_results")
        if lab_results:
            result["lab_findings"] = self.lab_processor.process(lab_results)
            
        # 3. Knowledge Graph Query (IO bound - but running in thread here for simplicity)
        positive_symptoms = [
            s["symptom"] for s in result["extracted_symptoms"] 
            if not s.get("negated", False)
        ]
        
        if positive_symptoms:
            result["kg_diseases"] = await loop.run_in_executor(
                self.executor,
                self._query_kg,
                positive_symptoms
            )
        
        # 4. Multimodal Fusion & Reasoning
        fusion_result = self._fuse_modalities(result)
        result["modality_scores"] = fusion_result["modality_scores"]
        result["conflicts_detected"] = fusion_result["conflicts"]
        
        # Generate differential with fusion scores
        differential = self._generate_differential(result)
        result["differential_diagnoses"] = differential
        
        if differential:
            result["primary_diagnosis"] = differential[0]
            result["confidence"] = differential[0].get("confidence", 0.0)
        
        # Safety validation
        safety_result = self.safety.validate_diagnosis(result)
        result["safety_alerts"] = safety_result.get("alerts", [])
        result["needs_escalation"] = safety_result.get("requires_escalation", False)
        if result["needs_escalation"]:
            reasons = safety_result.get("escalation_reasons", [])
            result["escalation_reason"] = "; ".join(reasons) if reasons else "Safety check triggered"
        
        # Generate explanation
        result["explanation"] = self._generate_explanation(result)
        
        # Add processing time
        result["processing_time_ms"] = int((time.time() - start_time) * 1000)
        
        # 5. Cache Result (1 hour TTL)
        if not patient_data.get("image_path"):
            self.cache.set(cache_key, result, ttl_seconds=3600)
        
        return result

    def _run_nlp(self, text):
        """Helper for running NLP in thread"""
        if not text:
            return None
        return self.nlp.analyze_clinical_note(text)
        
    def _run_image_analysis(self, path):
        """Helper for running vision in thread"""
        if self.vision:
            try:
                return self.vision.analyze_chest_xray(path)
            except Exception as e:
                logger.error(f"Image analysis failed: {e}")
                return {"error": str(e)}
        return {}

    def _query_kg(self, symptoms):
        """Helper for KG query in thread"""
        if self.kg:
            try:
                return self.kg.find_diseases_by_symptoms(symptoms, limit=10)
            except Exception as e:
                logger.error(f"KG query failed: {e}")
        return []
    
    def _fuse_modalities(self, state: Dict) -> Dict:
        """
        Fuse signals from multiple modalities with weighted scoring.
        
        Weights:
        - Symptoms (NLP): 0.30
        - Labs: 0.35
        - Image: 0.35
        
        Returns:
            Dict with modality_scores and conflicts
        """
        modality_scores = {}
        conflicts = []
        
        # Score each modality (0-1 scale for infection/abnormality)
        
        # 1. Symptom Score
        symptoms = state.get("extracted_symptoms", [])
        positive_symptoms = [s for s in symptoms if not s.get("negated", False)]
        infection_keywords = ["fever", "cough", "fatigue", "chills", "pain"]
        
        symptom_score = 0.0
        if positive_symptoms:
            infection_count = sum(1 for s in positive_symptoms 
                                  if any(k in s.get("symptom", "").lower() for k in infection_keywords))
            symptom_score = min(1.0, infection_count / 3)  # 3+ infection symptoms = 1.0
        modality_scores["symptoms"] = symptom_score
        
        # 2. Lab Score
        lab_findings = state.get("lab_findings", {})
        lab_score = lab_findings.get("severity_score", 0.0)
        modality_scores["labs"] = lab_score
        
        # 3. Image Score
        image_findings = state.get("image_findings", {})
        image_score = 0.0
        if image_findings.get("top_finding"):
            image_conf = image_findings.get("confidence", 0.5)
            # Higher score for abnormal findings
            if image_findings.get("is_abnormal", True):
                image_score = image_conf
        modality_scores["image"] = image_score
        
        # 4. Conflict Detection
        scores = [s for s in [symptom_score, lab_score, image_score] if s > 0]
        if len(scores) >= 2:
            max_diff = max(scores) - min(scores)
            if max_diff > 0.5:  # Significant disagreement
                high_mod = [k for k, v in modality_scores.items() if v == max(scores)][0]
                low_mod = [k for k, v in modality_scores.items() if v == min(scores)][0]
                conflicts.append({
                    "type": "MODALITY_DISAGREEMENT",
                    "description": f"{high_mod.title()} suggests positive ({max(scores):.2f}) but {low_mod.title()} suggests negative ({min(scores):.2f})",
                    "recommendation": "Specialist review recommended due to conflicting signals"
                })
        
        # Check specific conflicts
        lab_flags = lab_findings.get("flags", [])
        if lab_score > 0.6 and image_score < 0.3 and image_findings.get("top_finding"):
            conflicts.append({
                "type": "LAB_IMAGE_CONFLICT",
                "description": "Labs indicate infection/abnormality but image appears normal",
                "recommendation": "Consider early-stage disease or image limitations"
            })
        
        if symptom_score > 0.7 and lab_score < 0.2 and lab_findings:
            conflicts.append({
                "type": "SYMPTOM_LAB_CONFLICT",
                "description": "Strong symptoms but labs are normal",
                "recommendation": "Consider functional/psychological causes or repeat labs"
            })
        
        # 5. Calculate weighted fusion score
        weights = {"symptoms": 0.30, "labs": 0.35, "image": 0.35}
        
        # Only use available modalities
        available_weight = 0.0
        weighted_score = 0.0
        for modality, weight in weights.items():
            if modality_scores.get(modality, 0) > 0 or (modality == "symptoms" and symptoms):
                available_weight += weight
                weighted_score += modality_scores.get(modality, 0) * weight
        
        # Normalize to available modalities
        fusion_score = weighted_score / available_weight if available_weight > 0 else 0.0
        modality_scores["fusion"] = fusion_score
        modality_scores["data_completeness"] = available_weight
        
        return {
            "modality_scores": modality_scores,
            "conflicts": conflicts,
            "fusion_score": fusion_score
        }

    # --- Reusing existing logic from synchronous agent ---
    
    def _generate_differential(self, state: Dict) -> List[Dict]:
        """Generate differential diagnoses from collected evidence"""
        differential = []
        kg_diseases = state.get("kg_diseases", [])
        image_findings = state.get("image_findings", {})
        
        for disease in kg_diseases[:5]:
            confidence = self._calculate_confidence(disease, image_findings, state)
            differential.append({
                "disease": disease.get("disease", "Unknown"),
                "icd10": disease.get("icd10"),
                "confidence": confidence,
                "severity": disease.get("severity", "unknown"),
                "category": disease.get("category"),
                "matched_symptoms": disease.get("matched_symptoms", []),
            })
        
        differential.sort(key=lambda x: x["confidence"], reverse=True)
        
        if not differential:
            differential = self._fallback_suggestions(state)
        
        return differential
    
    def _calculate_confidence(self, disease, image_findings, state) -> float:
        """Calculate confidence combining multiple signals"""
        base_score = disease.get("match_ratio", 0.5)
        top_finding = image_findings.get("top_finding", "").lower()
        disease_name = disease.get("disease", "").lower()
        
        if top_finding and any(word in top_finding for word in disease_name.split()):
            base_score *= 1.2
        
        # Data completeness factor
        score = 0.4
        positive = [s for s in state["extracted_symptoms"] if not s.get("negated")]
        if len(positive) >= 2: score += 0.3
        elif len(positive) >= 1: score += 0.15
        if top_finding: score += 0.2
        if state.get("extracted_vitals"): score += 0.1
        data_completeness = min(score, 1.0)
        
        confidence = base_score * data_completeness
        return min(confidence, 0.99)
    
    def _fallback_suggestions(self, state: Dict) -> List[Dict]:
        """Fallback pattern matching suggestions"""
        symptoms = [s["symptom"] for s in state.get("extracted_symptoms", []) if not s.get("negated")]
        symptom_text = " ".join(symptoms).lower()
        suggestions = []
        
        if "fever" in symptom_text and "cough" in symptom_text:
            suggestions.append({"disease": "Respiratory Infection", "icd10": "J06.9", "confidence": 0.6, "severity": "moderate"})
        if "chest" in symptom_text and "pain" in symptom_text:
            suggestions.append({"disease": "Chest Pain, Unspecified", "icd10": "R07.9", "confidence": 0.5, "severity": "moderate"})
            
        if not suggestions:
            suggestions.append({"disease": "Unspecified condition", "icd10": None, "confidence": 0.3, "severity": "unknown"})
            
        return suggestions
    
    def _generate_explanation(self, state: Dict) -> str:
        """Generate explanation string"""
        primary = state.get("primary_diagnosis", {})
        disease_name = primary.get("disease", "Unknown condition")
        confidence = state.get("confidence", 0)
        
        explanation = f"DIAGNOSIS: {disease_name}\nCONFIDENCE: {confidence:.1%}\n\nKEY FINDINGS:\n"
        for symptom in state.get("extracted_symptoms", [])[:5]:
            neg = " (denied)" if symptom.get("negated") else ""
            explanation += f"• {symptom['symptom'].title()}{neg}\n"
            
        img = state.get("image_findings", {})
        if img.get("top_finding"):
            explanation += f"• X-ray: {img['top_finding']} ({img.get('confidence', 0):.1%})\n"
            
        if state.get("safety_alerts"):
            explanation += f"\n⚠️ ALERTS: {', '.join(state['safety_alerts'][:3])}\n"
            
        explanation += "\n⚠️ AI-assisted analysis. Verify with physician."
        return explanation
