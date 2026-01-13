"""
Clinical NLP Pipeline
Extract medical entities from clinical notes using SciSpacy and custom rules.
"""

import re
from typing import List, Dict, Optional
from loguru import logger


class ClinicalNLPPipeline:
    """Extract medical entities from clinical notes"""
    
    def __init__(self, load_models: bool = True):
        """
        Initialize clinical NLP pipeline.
        
        Args:
            load_models: Whether to load SpaCy models on init
        """
        self.nlp = None
        self._loaded = False
        
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
            "WNL": "within normal limits",
            "NAD": "no acute distress",
            "A&O": "alert and oriented",
            "RRR": "regular rate and rhythm",
            "CTA": "clear to auscultation",
            "HEENT": "head eyes ears nose throat",
            "ROM": "range of motion",
            "DTR": "deep tendon reflexes",
        }
        
        # Negation patterns
        self.negation_cues = [
            "no", "not", "denies", "denied", "without", "negative for",
            "rules out", "ruled out", "absence of", "no evidence of",
            "never", "none", "free of", "no sign of", "no signs of",
            "does not have", "doesn't have", "no history of"
        ]
        
        # Common symptoms for pattern matching fallback
        self.common_symptoms = [
            "fever", "cough", "dyspnea", "chest pain", "headache",
            "nausea", "vomiting", "diarrhea", "fatigue", "weakness",
            "shortness of breath", "abdominal pain", "back pain",
            "dizziness", "syncope", "edema", "rash", "weight loss",
            "night sweats", "hemoptysis", "sputum", "wheezing",
            "palpitations", "diaphoresis", "chills"
        ]
        
        if load_models:
            self._load_models()
        
        logger.info("ClinicalNLPPipeline initialized")
    
    def _load_models(self):
        """Load SpaCy and SciSpacy models"""
        if self._loaded:
            return
        
        try:
            import spacy
            from scispacy.linking import EntityLinker
            
            logger.info("Loading clinical NLP models...")
            
            self.nlp = spacy.load("en_core_sci_md")
            
            # Add UMLS entity linker
            try:
                self.nlp.add_pipe("scispacy_linker", config={
                    "resolve_abbreviations": True,
                    "linker_name": "umls"
                })
            except Exception as e:
                logger.warning(f"Could not add UMLS linker: {e}")
            
            self._loaded = True
            logger.info("Clinical NLP models loaded")
            
        except ImportError as e:
            logger.warning(f"SciSpacy not available: {e}. Using pattern-based extraction.")
            self._loaded = False
        except Exception as e:
            logger.warning(f"Failed to load models: {e}. Using pattern-based extraction.")
            self._loaded = False
    
    def expand_abbreviations(self, text: str) -> str:
        """
        Expand medical abbreviations.
        
        Args:
            text: Raw clinical text
            
        Returns:
            Text with abbreviations expanded
        """
        result = text
        for abbr, expansion in self.abbreviations.items():
            # Replace with word boundaries
            pattern = r'\b' + re.escape(abbr) + r'\b'
            result = re.sub(pattern, expansion, result, flags=re.IGNORECASE)
        return result
    
    def extract_symptoms(self, clinical_note: str) -> List[Dict]:
        """
        Extract symptoms with negation detection.
        
        Args:
            clinical_note: Clinical text to analyze
            
        Returns:
            List of symptom dicts with name, negation status, and context
        """
        # Expand abbreviations first
        text = self.expand_abbreviations(clinical_note)
        
        if self._loaded and self.nlp:
            return self._extract_with_spacy(text)
        else:
            return self._extract_with_patterns(text)
    
    def _extract_with_spacy(self, text: str) -> List[Dict]:
        """Extract using SpaCy NER"""
        doc = self.nlp(text)
        
        symptoms = []
        
        for ent in doc.ents:
            # Check if symptom-related entity
            if self._is_symptom_entity(ent):
                # Check for negation
                negated = self._check_negation(doc, ent)
                
                # Get UMLS CUI
                umls_cui = None
                if hasattr(ent._, 'kb_ents') and ent._.kb_ents:
                    umls_cui = ent._.kb_ents[0][0]
                
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
    
    def _extract_with_patterns(self, text: str) -> List[Dict]:
        """Fallback pattern-based extraction"""
        text_lower = text.lower()
        symptoms = []
        
        for symptom in self.common_symptoms:
            # Find all occurrences
            pattern = r'\b' + re.escape(symptom) + r'\b'
            for match in re.finditer(pattern, text_lower):
                start = match.start()
                end = match.end()
                
                # Get context for negation check
                context_start = max(0, start - 50)
                context = text[context_start:end]
                
                # Check negation
                negated = any(cue in context.lower() for cue in self.negation_cues)
                
                symptoms.append({
                    "symptom": symptom,
                    "canonical": None,
                    "negated": negated,
                    "context": context,
                    "start_char": start,
                    "end_char": end
                })
        
        return symptoms
    
    def _is_symptom_entity(self, ent) -> bool:
        """Check if entity is a symptom/finding"""
        symptom_types = ["DISEASE", "SIGN_SYMPTOM", "FINDING", "PROBLEM"]
        return ent.label_ in symptom_types or (hasattr(ent._, 'kb_ents') and len(ent._.kb_ents) > 0)
    
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
        """
        Extract vital signs from text.
        
        Args:
            text: Clinical text
            
        Returns:
            Dict with extracted vital signs
        """
        vitals = {}
        
        # Temperature patterns
        temp_patterns = [
            r'temp(?:erature)?[:\s]+(\d+\.?\d*)\s*[°]?[FC]?',
            r'T[:\s]+(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*[°][FC]'
        ]
        for pattern in temp_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                temp = float(match.group(1))
                # Normalize to Fahrenheit if likely Celsius
                if temp < 50:
                    temp = temp * 9/5 + 32
                vitals["temperature"] = temp
                break
        
        # Blood pressure
        bp_match = re.search(r'BP[:\s]+(\d+)/(\d+)', text, re.IGNORECASE)
        if bp_match:
            vitals["bp_systolic"] = int(bp_match.group(1))
            vitals["bp_diastolic"] = int(bp_match.group(2))
        
        # Heart rate
        hr_patterns = [
            r'(?:HR|heart\s*rate|pulse)[:\s]+(\d+)',
            r'P[:\s]+(\d+)\s*bpm'
        ]
        for pattern in hr_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                vitals["heart_rate"] = int(match.group(1))
                break
        
        # Respiratory rate
        rr_match = re.search(r'(?:RR|resp(?:iratory)?\s*rate)[:\s]+(\d+)', text, re.IGNORECASE)
        if rr_match:
            vitals["respiratory_rate"] = int(rr_match.group(1))
        
        # Oxygen saturation
        spo2_patterns = [
            r'(?:SpO2|O2\s*sat|SaO2)[:\s]+(\d+)%?',
            r'sat(?:uration)?[:\s]+(\d+)%?'
        ]
        for pattern in spo2_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                vitals["spo2"] = int(match.group(1))
                break
        
        return vitals
    
    def analyze_clinical_note(self, text: str) -> Dict:
        """
        Complete analysis of clinical note.
        
        Args:
            text: Clinical note text
            
        Returns:
            Dict with symptoms, vitals, and metadata
        """
        expanded_text = self.expand_abbreviations(text)
        
        symptoms = self.extract_symptoms(text)
        vitals = self.extract_vitals(text)
        
        # Separate positive and negative symptoms
        positive_symptoms = [s for s in symptoms if not s["negated"]]
        negative_symptoms = [s for s in symptoms if s["negated"]]
        
        return {
            "original_text": text,
            "expanded_text": expanded_text,
            "symptoms": symptoms,
            "positive_symptoms": positive_symptoms,
            "negative_symptoms": negative_symptoms,
            "vitals": vitals,
            "symptom_count": len(positive_symptoms)
        }
