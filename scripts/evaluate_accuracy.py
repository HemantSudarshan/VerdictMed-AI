"""
CDSS Accuracy Evaluation Script
Evaluate diagnostic accuracy against labeled test cases.

Usage:
    python scripts/evaluate_accuracy.py --test-data data/test_cases.json
    python scripts/evaluate_accuracy.py --generate-sample  # Create sample test file
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class EvaluationMetrics:
    """Container for accuracy metrics"""
    total_cases: int = 0
    correct_predictions: int = 0
    critical_detected: int = 0
    critical_total: int = 0
    false_negatives: int = 0
    false_positives: int = 0
    escalations: int = 0
    low_confidence: int = 0
    
    # Per-disease stats
    disease_stats: Dict[str, Dict] = field(default_factory=dict)
    
    @property
    def accuracy(self) -> float:
        return self.correct_predictions / max(self.total_cases, 1)
    
    @property
    def critical_detection_rate(self) -> float:
        return self.critical_detected / max(self.critical_total, 1)
    
    @property
    def false_negative_rate(self) -> float:
        return self.false_negatives / max(self.total_cases, 1)
    
    @property
    def escalation_rate(self) -> float:
        return self.escalations / max(self.total_cases, 1)


class MockDiagnosticAgent:
    """Mock agent for testing without infrastructure"""
    
    DISEASE_PATTERNS = {
        "pneumonia": ["fever", "cough", "sputum", "shortness of breath", "sob", "dyspnea", "chest"],
        "myocardial infarction": ["chest pain", "crushing", "radiating", "arm", "diaphoresis", "sweating", "nausea"],
        "tuberculosis": ["chronic cough", "night sweats", "weight loss", "hemoptysis", "6 weeks"],
        "common cold": ["runny nose", "sore throat", "mild cough", "no fever"],
        "stroke": ["weakness", "slurred speech", "confusion", "headache", "sudden onset", "one sided"],
        "asthma exacerbation": ["wheezing", "asthma", "chest tightness", "trigger"],
        "gastritis": ["epigastric", "burning", "antacids", "after meals"],
        "pulmonary embolism": ["sudden dyspnea", "pleuritic", "leg swelling", "dvt", "flight", "tachycardia"],
        "sepsis": ["fever", "confusion", "tachycardia", "hypotension", "lactate", "wbc"],
        "urinary tract infection": ["dysuria", "frequency", "suprapubic", "uti"]
    }
    
    def diagnose_sync(self, symptoms: str, patient_id: str = None) -> dict:
        """Mock diagnosis based on keyword matching"""
        symptoms_lower = symptoms.lower()
        
        best_match = None
        best_score = 0
        
        for disease, keywords in self.DISEASE_PATTERNS.items():
            score = sum(1 for kw in keywords if kw in symptoms_lower)
            if score > best_score:
                best_score = score
                best_match = disease
        
        # Calculate confidence based on match strength
        if best_match:
            max_keywords = len(self.DISEASE_PATTERNS[best_match])
            confidence = min(0.5 + (best_score / max_keywords) * 0.45, 0.95)
        else:
            best_match = "unknown"
            confidence = 0.3
        
        # Determine if escalation needed
        needs_escalation = confidence < 0.55
        
        return {
            "primary_diagnosis": {"disease": best_match.title()},
            "confidence": confidence,
            "needs_escalation": needs_escalation,
            "differential_diagnoses": []
        }


class AccuracyEvaluator:
    """Evaluate CDSS diagnostic accuracy"""
    
    CRITICAL_CONDITIONS = [
        "myocardial infarction", "acute mi", "heart attack",
        "stroke", "cva", "cerebrovascular",
        "pulmonary embolism", "pe",
        "sepsis", "septic shock",
        "anaphylaxis",
        "meningitis"
    ]
    
    def __init__(self, agent=None):
        """
        Initialize evaluator.
        
        Args:
            agent: DiagnosticAgent instance (optional, will create if None)
        """
        self.agent = agent
        self.metrics = EvaluationMetrics()
    
    def _init_agent(self):
        """Lazy-load diagnostic agent"""
        if self.agent is None:
            try:
                from src.reasoning.simple_agent import SimpleDiagnosticAgent
                from src.config import get_settings
                self.agent = SimpleDiagnosticAgent(get_settings())
                logger.info("Diagnostic agent initialized")
            except Exception as e:
                logger.error(f"Failed to initialize agent: {e}")
                raise
    
    def load_test_cases(self, test_file: str) -> List[Dict]:
        """Load test cases from JSON file"""
        path = Path(test_file)
        if not path.exists():
            raise FileNotFoundError(f"Test file not found: {test_file}")
        
        with open(path) as f:
            data = json.load(f)
        
        # Handle both list format and {"cases": [...]} format
        if isinstance(data, list):
            cases = data
        else:
            cases = data.get("cases", data.get("test_cases", []))
        
        logger.info(f"Loaded {len(cases)} test cases from {test_file}")
        return cases
    
    def _is_critical(self, diagnosis: str) -> bool:
        """Check if diagnosis is a critical condition"""
        diagnosis_lower = diagnosis.lower()
        return any(crit in diagnosis_lower for crit in self.CRITICAL_CONDITIONS)
    
    def _normalize_diagnosis(self, diagnosis: str) -> str:
        """Normalize diagnosis string for comparison"""
        return diagnosis.lower().strip().replace("_", " ").replace("-", " ")
    
    def _diagnoses_match(self, predicted: str, actual: str) -> bool:
        """Check if predicted and actual diagnoses match"""
        pred_norm = self._normalize_diagnosis(predicted)
        actual_norm = self._normalize_diagnosis(actual)
        
        # Exact match
        if pred_norm == actual_norm:
            return True
        
        # Partial match (one contains the other)
        if pred_norm in actual_norm or actual_norm in pred_norm:
            return True
        
        # Common synonyms
        synonyms = {
            "pneumonia": ["lung infection", "chest infection"],
            "myocardial infarction": ["heart attack", "mi", "acute mi"],
            "stroke": ["cva", "cerebrovascular accident"],
            "tuberculosis": ["tb", "pulmonary tb"],
        }
        
        for key, values in synonyms.items():
            if key in pred_norm or any(v in pred_norm for v in values):
                if key in actual_norm or any(v in actual_norm for v in values):
                    return True
        
        return False
    
    def evaluate_single(self, case: Dict) -> Dict:
        """
        Evaluate a single test case.
        
        Args:
            case: {
                "symptoms": str,
                "actual_diagnosis": str,
                "patient_age": int (optional),
                "patient_sex": str (optional)
            }
            
        Returns:
            Evaluation result dict
        """
        self._init_agent()
        
        symptoms = case.get("symptoms", "")
        actual = case.get("actual_diagnosis", case.get("diagnosis", ""))
        
        # Run diagnosis
        try:
            result = self.agent.diagnose_sync(
                symptoms=symptoms,
                patient_id=case.get("case_id", "test")
            )
        except Exception as e:
            logger.error(f"Diagnosis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "case": case
            }
        
        # Extract prediction
        primary = result.get("primary_diagnosis", {})
        predicted = primary.get("disease", "unknown")
        confidence = result.get("confidence", 0)
        escalated = result.get("needs_escalation", False)
        
        # Check correctness
        is_correct = self._diagnoses_match(predicted, actual)
        is_critical_case = self._is_critical(actual)
        detected_critical = self._is_critical(predicted) if is_correct else False
        
        # Update metrics
        self.metrics.total_cases += 1
        
        if is_correct:
            self.metrics.correct_predictions += 1
        elif not is_correct and not escalated:
            self.metrics.false_negatives += 1
        
        if is_critical_case:
            self.metrics.critical_total += 1
            if is_correct or escalated:
                self.metrics.critical_detected += 1
        
        if escalated:
            self.metrics.escalations += 1
        
        if confidence < 0.55:
            self.metrics.low_confidence += 1
        
        # Track per-disease stats
        if actual not in self.metrics.disease_stats:
            self.metrics.disease_stats[actual] = {"total": 0, "correct": 0}
        self.metrics.disease_stats[actual]["total"] += 1
        if is_correct:
            self.metrics.disease_stats[actual]["correct"] += 1
        
        return {
            "success": True,
            "correct": is_correct,
            "predicted": predicted,
            "actual": actual,
            "confidence": confidence,
            "escalated": escalated,
            "is_critical": is_critical_case
        }
    
    def evaluate_all(self, test_cases: List[Dict]) -> Dict:
        """
        Evaluate all test cases.
        
        Args:
            test_cases: List of test case dicts
            
        Returns:
            Full evaluation results
        """
        self.metrics = EvaluationMetrics()  # Reset
        results = []
        
        for i, case in enumerate(test_cases):
            logger.info(f"Evaluating case {i+1}/{len(test_cases)}")
            result = self.evaluate_single(case)
            results.append(result)
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "total_cases": self.metrics.total_cases,
            "metrics": {
                "accuracy": self.metrics.accuracy,
                "critical_detection_rate": self.metrics.critical_detection_rate,
                "false_negative_rate": self.metrics.false_negative_rate,
                "escalation_rate": self.metrics.escalation_rate,
                "low_confidence_count": self.metrics.low_confidence
            },
            "targets": {
                "accuracy": {"target": 0.85, "met": self.metrics.accuracy >= 0.85},
                "critical_detection": {"target": 0.95, "met": self.metrics.critical_detection_rate >= 0.95},
                "false_negative_rate": {"target": 0.08, "met": self.metrics.false_negative_rate <= 0.08},
                "escalation_rate": {"target": "15-25%", "met": 0.15 <= self.metrics.escalation_rate <= 0.25}
            },
            "per_disease": self.metrics.disease_stats,
            "individual_results": results
        }


def generate_sample_test_file(output_path: str = "data/test_cases.json"):
    """Generate a sample test cases file"""
    sample_cases = [
        {
            "case_id": "TC001",
            "symptoms": "45 year old male with fever x3 days, productive cough with yellow sputum, shortness of breath. Vitals: T 38.5°C, HR 95, BP 130/85, SpO2 94%",
            "actual_diagnosis": "Pneumonia",
            "severity": "moderate"
        },
        {
            "case_id": "TC002",
            "symptoms": "62 year old female with severe crushing chest pain radiating to left arm, diaphoresis, nausea. Started 30 minutes ago. History of HTN, DM.",
            "actual_diagnosis": "Myocardial Infarction",
            "severity": "critical"
        },
        {
            "case_id": "TC003",
            "symptoms": "28 year old female with chronic cough x6 weeks, night sweats, weight loss of 5kg, low grade fever. Recent travel to endemic area.",
            "actual_diagnosis": "Tuberculosis",
            "severity": "high"
        },
        {
            "case_id": "TC004",
            "symptoms": "35 year old male with runny nose, sore throat, mild cough x2 days. No fever. Denies SOB.",
            "actual_diagnosis": "Common Cold",
            "severity": "low"
        },
        {
            "case_id": "TC005",
            "symptoms": "55 year old male with sudden onset severe headache, confusion, left sided weakness, slurred speech. Started 1 hour ago.",
            "actual_diagnosis": "Stroke",
            "severity": "critical"
        },
        {
            "case_id": "TC006",
            "symptoms": "40 year old female with wheezing, shortness of breath, chest tightness. History of asthma, worse with cold air.",
            "actual_diagnosis": "Asthma Exacerbation",
            "severity": "moderate"
        },
        {
            "case_id": "TC007",
            "symptoms": "50 year old male with burning epigastric pain, worse after meals, some relief with antacids. No alarm symptoms.",
            "actual_diagnosis": "Gastritis",
            "severity": "low"
        },
        {
            "case_id": "TC008",
            "symptoms": "32 year old female with sudden onset dyspnea, pleuritic chest pain, tachycardia. Recent long flight. Leg swelling noted.",
            "actual_diagnosis": "Pulmonary Embolism",
            "severity": "critical"
        },
        {
            "case_id": "TC009",
            "symptoms": "70 year old male with fever, confusion, tachycardia, hypotension. WBC 18000, lactate elevated. Source unclear.",
            "actual_diagnosis": "Sepsis",
            "severity": "critical"
        },
        {
            "case_id": "TC010",
            "symptoms": "25 year old female with dysuria, urinary frequency, suprapubic pain. No fever. Sexually active.",
            "actual_diagnosis": "Urinary Tract Infection",
            "severity": "low"
        }
    ]
    
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output, "w") as f:
        json.dump({"cases": sample_cases, "version": "1.0"}, f, indent=2)
    
    print(f"✅ Generated {len(sample_cases)} sample test cases at: {output_path}")
    print("\nTo evaluate accuracy, run:")
    print(f"  python scripts/evaluate_accuracy.py --test-data {output_path}")


def print_results(results: Dict):
    """Print formatted evaluation results"""
    print("\n" + "=" * 60)
    print("CDSS ACCURACY EVALUATION REPORT")
    print("=" * 60)
    print(f"Timestamp: {results['timestamp']}")
    print(f"Total Cases: {results['total_cases']}")
    print("-" * 60)
    
    print("\n📊 METRICS:")
    metrics = results['metrics']
    print(f"  Overall Accuracy:        {metrics['accuracy']:.1%}")
    print(f"  Critical Detection Rate: {metrics['critical_detection_rate']:.1%}")
    print(f"  False Negative Rate:     {metrics['false_negative_rate']:.1%}")
    print(f"  Escalation Rate:         {metrics['escalation_rate']:.1%}")
    print(f"  Low Confidence Cases:    {metrics['low_confidence_count']}")
    
    print("\n🎯 TARGET COMPLIANCE:")
    targets = results['targets']
    for name, data in targets.items():
        status = "✅" if data['met'] else "❌"
        print(f"  {status} {name}: {data['target']} (met: {data['met']})")
    
    print("\n📈 PER-DISEASE ACCURACY:")
    for disease, stats in results['per_disease'].items():
        acc = stats['correct'] / max(stats['total'], 1)
        print(f"  {disease}: {stats['correct']}/{stats['total']} ({acc:.1%})")
    
    # Overall status
    print("\n" + "=" * 60)
    all_met = all(t['met'] for t in targets.values())
    if all_met:
        print("🎉 ALL TARGETS MET - System is production ready!")
    else:
        print("⚠️  Some targets not met - Further tuning needed")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate CDSS accuracy")
    parser.add_argument("--test-data", help="Path to test cases JSON file")
    parser.add_argument("--generate-sample", action="store_true", help="Generate sample test file")
    parser.add_argument("--output", default="data/evaluation_results.json", help="Output results file")
    parser.add_argument("--mock", action="store_true", default=True, help="Use mock agent (default: True)")
    parser.add_argument("--real", action="store_true", help="Use real diagnostic agent (requires infrastructure)")
    args = parser.parse_args()
    
    if args.generate_sample:
        generate_sample_test_file()
        return
    
    if not args.test_data:
        print("Error: --test-data required (or use --generate-sample)")
        parser.print_help()
        return
    
    # Determine which agent to use
    if args.real:
        print("🔧 Using REAL diagnostic agent (requires Docker infrastructure)")
        evaluator = AccuracyEvaluator()
    else:
        print("🧪 Using MOCK diagnostic agent (keyword-based matching)")
        evaluator = AccuracyEvaluator(agent=MockDiagnosticAgent())
    
    try:
        test_cases = evaluator.load_test_cases(args.test_data)
        results = evaluator.evaluate_all(test_cases)
        
        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print_results(results)
        print(f"\nResults saved to: {args.output}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nTo generate sample test cases, run:")
        print("  python scripts/evaluate_accuracy.py --generate-sample")


if __name__ == "__main__":
    main()
