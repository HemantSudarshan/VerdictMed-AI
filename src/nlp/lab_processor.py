"""
Lab Values Processor
Interprets structured lab results against clinical thresholds.
Provides abnormality detection, severity scoring, and clinical flags.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class AbnormalityLevel(str, Enum):
    """Severity level for lab abnormalities"""
    NORMAL = "normal"
    LOW = "low"
    HIGH = "high"
    CRITICAL_LOW = "critical_low"
    CRITICAL_HIGH = "critical_high"


@dataclass
class LabReference:
    """Reference range for a lab value"""
    name: str
    unit: str
    low_normal: float
    high_normal: float
    critical_low: Optional[float] = None
    critical_high: Optional[float] = None
    clinical_significance: str = ""


# Standard clinical reference ranges
LAB_REFERENCES: Dict[str, LabReference] = {
    # Complete Blood Count (CBC)
    "wbc": LabReference(
        name="White Blood Cell Count",
        unit="x10^9/L",
        low_normal=4.5,
        high_normal=11.0,
        critical_low=2.0,
        critical_high=30.0,
        clinical_significance="Infection, inflammation, leukemia"
    ),
    "rbc": LabReference(
        name="Red Blood Cell Count",
        unit="x10^12/L",
        low_normal=4.5,
        high_normal=5.5,
        critical_low=2.5,
        critical_high=8.0,
        clinical_significance="Anemia, polycythemia"
    ),
    "hemoglobin": LabReference(
        name="Hemoglobin",
        unit="g/dL",
        low_normal=12.0,
        high_normal=17.0,
        critical_low=7.0,
        critical_high=20.0,
        clinical_significance="Anemia, dehydration"
    ),
    "hematocrit": LabReference(
        name="Hematocrit",
        unit="%",
        low_normal=36.0,
        high_normal=50.0,
        critical_low=20.0,
        critical_high=60.0,
        clinical_significance="Blood volume status"
    ),
    "platelets": LabReference(
        name="Platelet Count",
        unit="x10^9/L",
        low_normal=150.0,
        high_normal=400.0,
        critical_low=50.0,
        critical_high=1000.0,
        clinical_significance="Bleeding risk, clotting disorders"
    ),
    
    # Inflammatory Markers
    "crp": LabReference(
        name="C-Reactive Protein",
        unit="mg/L",
        low_normal=0.0,
        high_normal=10.0,
        critical_high=100.0,
        clinical_significance="Inflammation, infection severity"
    ),
    "esr": LabReference(
        name="Erythrocyte Sedimentation Rate",
        unit="mm/hr",
        low_normal=0.0,
        high_normal=20.0,
        critical_high=100.0,
        clinical_significance="Chronic inflammation"
    ),
    "procalcitonin": LabReference(
        name="Procalcitonin",
        unit="ng/mL",
        low_normal=0.0,
        high_normal=0.5,
        critical_high=10.0,
        clinical_significance="Bacterial infection marker"
    ),
    
    # Cardiac Markers
    "troponin": LabReference(
        name="Troponin I",
        unit="ng/mL",
        low_normal=0.0,
        high_normal=0.04,
        critical_high=0.4,
        clinical_significance="Myocardial infarction"
    ),
    "bnp": LabReference(
        name="Brain Natriuretic Peptide",
        unit="pg/mL",
        low_normal=0.0,
        high_normal=100.0,
        critical_high=900.0,
        clinical_significance="Heart failure"
    ),
    "ck_mb": LabReference(
        name="CK-MB",
        unit="ng/mL",
        low_normal=0.0,
        high_normal=5.0,
        critical_high=25.0,
        clinical_significance="Cardiac muscle damage"
    ),
    
    # Metabolic Panel
    "glucose": LabReference(
        name="Blood Glucose",
        unit="mg/dL",
        low_normal=70.0,
        high_normal=100.0,
        critical_low=40.0,
        critical_high=500.0,
        clinical_significance="Diabetes, hypoglycemia"
    ),
    "creatinine": LabReference(
        name="Creatinine",
        unit="mg/dL",
        low_normal=0.6,
        high_normal=1.2,
        critical_high=10.0,
        clinical_significance="Kidney function"
    ),
    "bun": LabReference(
        name="Blood Urea Nitrogen",
        unit="mg/dL",
        low_normal=7.0,
        high_normal=20.0,
        critical_high=100.0,
        clinical_significance="Kidney function, dehydration"
    ),
    "sodium": LabReference(
        name="Sodium",
        unit="mEq/L",
        low_normal=136.0,
        high_normal=145.0,
        critical_low=120.0,
        critical_high=160.0,
        clinical_significance="Electrolyte balance"
    ),
    "potassium": LabReference(
        name="Potassium",
        unit="mEq/L",
        low_normal=3.5,
        high_normal=5.0,
        critical_low=2.5,
        critical_high=6.5,
        clinical_significance="Cardiac arrhythmia risk"
    ),
    
    # Liver Function
    "alt": LabReference(
        name="Alanine Aminotransferase",
        unit="U/L",
        low_normal=7.0,
        high_normal=56.0,
        critical_high=1000.0,
        clinical_significance="Liver damage"
    ),
    "ast": LabReference(
        name="Aspartate Aminotransferase",
        unit="U/L",
        low_normal=10.0,
        high_normal=40.0,
        critical_high=1000.0,
        clinical_significance="Liver/muscle damage"
    ),
    "bilirubin": LabReference(
        name="Total Bilirubin",
        unit="mg/dL",
        low_normal=0.1,
        high_normal=1.2,
        critical_high=15.0,
        clinical_significance="Liver function, hemolysis"
    ),
    
    # Coagulation
    "pt": LabReference(
        name="Prothrombin Time",
        unit="seconds",
        low_normal=11.0,
        high_normal=13.5,
        critical_high=30.0,
        clinical_significance="Bleeding risk"
    ),
    "inr": LabReference(
        name="INR",
        unit="ratio",
        low_normal=0.8,
        high_normal=1.1,
        critical_high=5.0,
        clinical_significance="Anticoagulation monitoring"
    ),
    
    # Blood Gases
    "ph": LabReference(
        name="Blood pH",
        unit="",
        low_normal=7.35,
        high_normal=7.45,
        critical_low=7.20,
        critical_high=7.60,
        clinical_significance="Acid-base balance"
    ),
    "pao2": LabReference(
        name="Partial Pressure of Oxygen",
        unit="mmHg",
        low_normal=80.0,
        high_normal=100.0,
        critical_low=60.0,
        clinical_significance="Oxygenation"
    ),
    "paco2": LabReference(
        name="Partial Pressure of CO2",
        unit="mmHg",
        low_normal=35.0,
        high_normal=45.0,
        critical_low=20.0,
        critical_high=70.0,
        clinical_significance="Ventilation"
    ),
    "lactate": LabReference(
        name="Lactate",
        unit="mmol/L",
        low_normal=0.5,
        high_normal=2.0,
        critical_high=4.0,
        clinical_significance="Tissue hypoxia, sepsis"
    ),
}


class LabProcessor:
    """
    Process and interpret structured lab values against clinical thresholds.
    Provides abnormality detection, severity scoring, and clinical flags.
    """
    
    def __init__(self, references: Optional[Dict[str, LabReference]] = None):
        """
        Initialize lab processor with reference ranges.
        
        Args:
            references: Custom reference ranges (uses defaults if None)
        """
        self.references = references or LAB_REFERENCES
    
    def process(self, lab_results: Dict) -> Dict:
        """
        Process lab results and return clinical interpretation.
        
        Args:
            lab_results: Dict of lab_name -> value
            
        Returns:
            Dict with abnormalities, severity_score, flags, and recommendations
        """
        if not lab_results:
            return {
                "abnormalities": [],
                "severity_score": 0.0,
                "flags": [],
                "recommendations": [],
                "summary": "No lab results provided"
            }
        
        abnormalities = []
        critical_flags = []
        recommendations = []
        severity_scores = []
        
        for lab_name, value in lab_results.items():
            # Normalize lab name
            normalized_name = lab_name.lower().replace(" ", "_").replace("-", "_")
            
            if normalized_name not in self.references:
                logger.warning(f"Unknown lab: {lab_name}")
                continue
            
            ref = self.references[normalized_name]
            level, severity = self._evaluate_value(value, ref)
            
            if level != AbnormalityLevel.NORMAL:
                abnormality = {
                    "lab": ref.name,
                    "value": value,
                    "unit": ref.unit,
                    "level": level.value,
                    "reference_range": f"{ref.low_normal}-{ref.high_normal}",
                    "clinical_significance": ref.clinical_significance
                }
                abnormalities.append(abnormality)
                severity_scores.append(severity)
                
                # Check for critical values
                if level in [AbnormalityLevel.CRITICAL_LOW, AbnormalityLevel.CRITICAL_HIGH]:
                    critical_flags.append(f"CRITICAL: {ref.name} = {value} {ref.unit}")
                    recommendations.append(f"Immediate attention needed for {ref.name}")
        
        # Calculate overall severity (0-1)
        overall_severity = max(severity_scores) if severity_scores else 0.0
        
        # Generate clinical flags
        flags = self._generate_clinical_flags(lab_results, abnormalities)
        flags.extend(critical_flags)
        
        # Generate recommendations
        recommendations.extend(self._generate_recommendations(abnormalities))
        
        return {
            "abnormalities": abnormalities,
            "severity_score": overall_severity,
            "flags": list(set(flags)),  # Remove duplicates
            "recommendations": list(set(recommendations)),
            "summary": self._generate_summary(abnormalities, overall_severity)
        }
    
    def _evaluate_value(self, value: float, ref: LabReference) -> Tuple[AbnormalityLevel, float]:
        """
        Evaluate a lab value against reference range.
        
        Returns:
            Tuple of (abnormality_level, severity_score 0-1)
        """
        try:
            value = float(value)
        except (ValueError, TypeError):
            return AbnormalityLevel.NORMAL, 0.0
        
        # Check critical values first
        if ref.critical_low is not None and value < ref.critical_low:
            severity = min(1.0, (ref.critical_low - value) / ref.critical_low + 0.8)
            return AbnormalityLevel.CRITICAL_LOW, severity
        
        if ref.critical_high is not None and value > ref.critical_high:
            severity = min(1.0, (value - ref.critical_high) / ref.critical_high + 0.8)
            return AbnormalityLevel.CRITICAL_HIGH, severity
        
        # Check normal range
        if value < ref.low_normal:
            severity = (ref.low_normal - value) / (ref.low_normal - (ref.critical_low or 0))
            return AbnormalityLevel.LOW, min(0.6, severity * 0.5)
        
        if value > ref.high_normal:
            severity = (value - ref.high_normal) / ((ref.critical_high or ref.high_normal * 2) - ref.high_normal)
            return AbnormalityLevel.HIGH, min(0.6, severity * 0.5)
        
        return AbnormalityLevel.NORMAL, 0.0
    
    def _generate_clinical_flags(self, lab_results: Dict, abnormalities: List) -> List[str]:
        """Generate high-level clinical flags based on lab patterns."""
        flags = []
        
        # Normalize all lab names
        normalized = {k.lower().replace(" ", "_"): v for k, v in lab_results.items()}
        
        # Check for sepsis indicators
        if normalized.get("wbc", 0) > 12 and normalized.get("lactate", 0) > 2:
            flags.append("SEPSIS_RISK: Elevated WBC + Lactate")
        
        if normalized.get("procalcitonin", 0) > 0.5:
            flags.append("BACTERIAL_INFECTION: Elevated Procalcitonin")
        
        # Check for cardiac indicators
        if normalized.get("troponin", 0) > 0.04:
            flags.append("CARDIAC_DAMAGE: Elevated Troponin")
        
        if normalized.get("bnp", 0) > 100:
            flags.append("HEART_FAILURE_RISK: Elevated BNP")
        
        # Check for kidney indicators
        if normalized.get("creatinine", 0) > 1.5 and normalized.get("bun", 0) > 25:
            flags.append("RENAL_DYSFUNCTION: Elevated Creatinine + BUN")
        
        # Check for liver indicators
        if normalized.get("alt", 0) > 100 or normalized.get("ast", 0) > 100:
            flags.append("LIVER_DYSFUNCTION: Elevated Transaminases")
        
        # Check for coagulation issues
        if normalized.get("inr", 0) > 2.0:
            flags.append("BLEEDING_RISK: Elevated INR")
        
        # Check for electrolyte imbalance
        potassium = normalized.get("potassium", 4.0)
        if potassium < 3.0 or potassium > 5.5:
            flags.append("ELECTROLYTE_IMBALANCE: Abnormal Potassium")
        
        return flags
    
    def _generate_recommendations(self, abnormalities: List) -> List[str]:
        """Generate clinical recommendations based on abnormalities."""
        recommendations = []
        
        for abn in abnormalities:
            lab = abn["lab"].lower()
            level = abn["level"]
            
            if "troponin" in lab and level in ["high", "critical_high"]:
                recommendations.append("Consider cardiac workup: ECG, echocardiogram")
            
            if "wbc" in lab and level == "critical_high":
                recommendations.append("Consider blood cultures, infection workup")
            
            if "creatinine" in lab and level in ["high", "critical_high"]:
                recommendations.append("Assess kidney function, review medications")
            
            if "glucose" in lab and level == "critical_high":
                recommendations.append("Check for diabetic ketoacidosis")
            
            if "hemoglobin" in lab and level in ["low", "critical_low"]:
                recommendations.append("Consider transfusion evaluation")
        
        return recommendations
    
    def _generate_summary(self, abnormalities: List, severity: float) -> str:
        """Generate human-readable summary of lab findings."""
        if not abnormalities:
            return "All lab values within normal limits"
        
        critical = [a for a in abnormalities if "critical" in a["level"]]
        high_low = [a for a in abnormalities if a["level"] in ["high", "low"]]
        
        parts = []
        
        if critical:
            parts.append(f"{len(critical)} critical abnormalities")
        
        if high_low:
            parts.append(f"{len(high_low)} out-of-range values")
        
        severity_text = "mild" if severity < 0.3 else "moderate" if severity < 0.6 else "significant"
        
        return f"Found {', '.join(parts)} with {severity_text} clinical significance"


# Singleton instance
_lab_processor = None


def get_lab_processor() -> LabProcessor:
    """Get singleton lab processor instance"""
    global _lab_processor
    if _lab_processor is None:
        _lab_processor = LabProcessor()
    return _lab_processor
