"""
Review Service
Manages human-in-the-loop workflow for doctor review and confirmation.
Handles pending review queue, diagnosis confirmation, and specialist escalation.
"""

from typing import List, Optional, Dict
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from loguru import logger

from src.database.models import Diagnosis, Patient
from src.database.session import get_db_session
from src.database.audit_service import get_audit_service


class ReviewService:
    """
    Service for managing human-in-the-loop review workflow.
    Enables doctors to review, confirm, reject, and escalate AI diagnoses.
    """
    
    def __init__(self, db_session: Optional[Session] = None):
        """
        Initialize review service.
        
        Args:
            db_session: SQLAlchemy session (optional)
        """
        self.db = db_session
        self.audit = get_audit_service(db_session)
    
    def get_pending_reviews(
        self,
        doctor_id: Optional[str] = None,
        min_confidence: Optional[float] = None,
        max_confidence: Optional[float] = None,
        escalated_only: bool = False,
        limit: int = 50
    ) -> List[Dict]:
        """
        Get diagnoses pending doctor review.
        
        Args:
            doctor_id: Filter by assigned doctor
            min_confidence: Minimum confidence threshold
            max_confidence: Maximum confidence threshold
            escalated_only: Only escalated cases
            limit: Max results to return
            
        Returns:
            List of diagnosis records with patient info
        """
        db = self._get_db()
        
        # Build query
        query = db.query(Diagnosis).join(Patient, Patient.id == Diagnosis.patient_id, isouter=True)
        
        # Filter: not yet confirmed
        query = query.filter(Diagnosis.doctor_confirmed == False)
        
        if doctor_id:
            query = query.filter(Diagnosis.doctor_id == doctor_id)
        
        if min_confidence is not None:
            query = query.filter(Diagnosis.confidence >= min_confidence)
        
        if max_confidence is not None:
            query = query.filter(Diagnosis.confidence <= max_confidence)
        
        if escalated_only:
            query = query.filter(Diagnosis.escalated_to_human == True)
        
        # Order by priority: escalated first, then by confidence (low first), then by date
        query = query.order_by(
            Diagnosis.escalated_to_human.desc(),
            Diagnosis.confidence.asc(),
            Diagnosis.created_at.asc()
        )
        
        diagnoses = query.limit(limit).all()
        
        # Convert to dict with patient info
        results = []
        for diag in diagnoses:
            results.append({
                "diagnosis_id": diag.id,
                "patient_id": diag.patient_id,
                "patient_age": diag.patient.age if diag.patient else None,
                "patient_sex": diag.patient.sex if diag.patient else None,
                "symptoms": diag.symptoms_text,
                "predicted_diagnosis": diag.predicted_diagnosis,
                "predicted_icd10": diag.predicted_icd10,
                "confidence": diag.confidence,
                "differential_diagnoses": diag.differential_diagnoses,
                "explanation": diag.explanation,
                "image_findings": diag.image_findings,
                "conflicts_detected": diag.conflict_detected,
                "conflicts_details": diag.conflicts_details,
                "escalated": diag.escalated_to_human,
                "escalation_reason": diag.escalation_reason,
                "safety_alerts": diag.safety_alerts,
                "created_at": diag.created_at.isoformat(),
                "processing_time_ms": diag.processing_time_ms
            })
        
        logger.info(f"Found {len(results)} pending reviews")
        return results
    
    def confirm_diagnosis(
        self,
        diagnosis_id: str,
        doctor_id: str,
        confirmed: bool,
        actual_diagnosis: Optional[str] = None,
        actual_icd10: Optional[str] = None,
        notes: Optional[str] = None
    ) -> Dict:
        """
        Doctor confirms or rejects AI diagnosis.
        
        Args:
            diagnosis_id: ID of diagnosis to confirm
            doctor_id: ID of confirming doctor
            confirmed: True if AI was correct, False if incorrect
            actual_diagnosis: Actual diagnosis if different
            actual_icd10: Actual ICD-10 code if different
            notes: Doctor's notes
            
        Returns:
            Updated diagnosis info
        """
        db = self._get_db()
        
        # Get diagnosis
        diag = db.query(Diagnosis).filter(Diagnosis.id == diagnosis_id).first()
        
        if not diag:
            raise ValueError(f"Diagnosis {diagnosis_id} not found")
        
        if diag.doctor_confirmed:
            logger.warning(f"Diagnosis {diagnosis_id} already confirmed")
        
        # Update diagnosis
        diag.doctor_id = doctor_id
        diag.doctor_confirmed = True
        diag.confirmed_at = datetime.utcnow()
        diag.doctor_notes = notes
        
        if not confirmed:
            # AI was wrong - record actual diagnosis
            diag.actual_diagnosis = actual_diagnosis or "Unknown"
            diag.actual_icd10 = actual_icd10
        else:
            # AI was correct - actual matches predicted
            diag.actual_diagnosis = diag.predicted_diagnosis
            diag.actual_icd10 = diag.predicted_icd10
        
        db.commit()
        db.refresh(diag)
        
        # Audit log
        self.audit.log_confirmation(
            diagnosis_id=diagnosis_id,
            doctor_id=doctor_id,
            confirmed=confirmed,
            actual_diagnosis=diag.actual_diagnosis,
            notes=notes
        )
        
        logger.info(f"Diagnosis {diagnosis_id} {'confirmed' if confirmed else 'rejected'} by {doctor_id}")
        
        return {
            "diagnosis_id": diag.id,
            "confirmed": confirmed,
            "ai_correct": confirmed,
            "predicted_diagnosis": diag.predicted_diagnosis,
            "actual_diagnosis": diag.actual_diagnosis,
            "doctor_notes": diag.doctor_notes,
            "confirmed_at": diag.confirmed_at.isoformat()
        }
    
    def escalate_to_specialist(
        self,
        diagnosis_id: str,
        escalated_by: str,
        specialist_id: str,
        reason: str,
        notes: Optional[str] = None
    ) -> Dict:
        """
        Escalate case to specialist for second opinion.
        
        Args:
            diagnosis_id: ID of diagnosis to escalate
            escalated_by: User ID initiating escalation
            specialist_id: User ID of specialist
            reason: Escalation reason
            notes: Additional notes
            
        Returns:
            Escalation result
        """
        db = self._get_db()
        
        # Get diagnosis
        diag = db.query(Diagnosis).filter(Diagnosis.id == diagnosis_id).first()
        
        if not diag:
            raise ValueError(f"Diagnosis {diagnosis_id} not found")
        
        # Update diagnosis
        diag.escalated_to_human = True
        diag.escalation_reason = reason
        diag.doctor_id = specialist_id  # Assign to specialist
        
        # Add to notes
        escalation_note = f"[{datetime.utcnow().isoformat()}] Escalated to specialist. Reason: {reason}"
        if notes:
            escalation_note += f"\nNotes: {notes}"
        
        if diag.doctor_notes:
            diag.doctor_notes += f"\n\n{escalation_note}"
        else:
            diag.doctor_notes = escalation_note
        
        db.commit()
        db.refresh(diag)
        
        # Audit log
        self.audit.log_escalation(
            diagnosis_id=diagnosis_id,
            escalated_to=specialist_id,
            escalated_by=escalated_by,
            reason=reason,
            details={"notes": notes}
        )
        
        logger.info(f"Diagnosis {diagnosis_id} escalated to {specialist_id}")
        
        return {
            "diagnosis_id": diag.id,
            "escalated_to": specialist_id,
            "escalated_by": escalated_by,
            "reason": reason,
            "status": "escalated"
        }
    
    def get_review_stats(
        self,
        doctor_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """
        Get review statistics for monitoring.
        
        Args:
            doctor_id: Filter by doctor
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            Statistics dict
        """
        db = self._get_db()
        
        query = db.query(Diagnosis)
        
        if doctor_id:
            query = query.filter(Diagnosis.doctor_id == doctor_id)
        if start_date:
            query = query.filter(Diagnosis.created_at >= start_date)
        if end_date:
            query = query.filter(Diagnosis.created_at <= end_date)
        
        all_diagnoses = query.all()
        confirmed = query.filter(Diagnosis.doctor_confirmed == True).all()
        
        # Calculate accuracy
        correct = sum(1 for d in confirmed if d.predicted_diagnosis == d.actual_diagnosis)
        accuracy = (correct / len(confirmed) * 100) if confirmed else 0
        
        # Escalation rate
        escalated = sum(1 for d in all_diagnoses if d.escalated_to_human)
        escalation_rate = (escalated / len(all_diagnoses) * 100) if all_diagnoses else 0
        
        # Average confidence
        avg_confidence = sum(d.confidence for d in all_diagnoses) / len(all_diagnoses) if all_diagnoses else 0
        
        return {
            "total_diagnoses": len(all_diagnoses),
            "confirmed_count": len(confirmed),
            "pending_count": len(all_diagnoses) - len(confirmed),
            "correct_count": correct,
            "incorrect_count": len(confirmed) - correct,
            "accuracy_percent": round(accuracy, 2),
            "escalated_count": escalated,
            "escalation_rate_percent": round(escalation_rate, 2),
            "average_confidence": round(avg_confidence, 3)
        }
    
    def _get_db(self) -> Session:
        """Get database session"""
        if self.db is None:
            self.db = next(get_db_session())
        return self.db


# Singleton instance
_review_service = None


def get_review_service(db_session: Optional[Session] = None) -> ReviewService:
    """Get singleton review service instance"""
    global _review_service
    if _review_service is None or db_session is not None:
        _review_service = ReviewService(db_session)
    return _review_service
