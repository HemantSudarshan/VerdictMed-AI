"""
Audit Service
Service for writing audit entries to PostgreSQL for HIPAA compliance.
Logs all data access, diagnoses, and system actions.
"""

from typing import Dict, Optional, List
from datetime import datetime
from sqlalchemy.orm import Session
from loguru import logger

from src.database.models import AuditLog
from src.database.session import get_db_session


class AuditService:
    """
    Service for managing audit logs in PostgreSQL.
    Ensures complete audit trail for compliance and security monitoring.
    """
    
    def __init__(self, db_session: Optional[Session] = None):
        """
        Initialize audit service.
        
        Args:
            db_session: SQLAlchemy session (optional, will create if None)
        """
        self.db = db_session
    
    def log_diagnosis(
        self,
        diagnosis_id: str,
        patient_id: str,
        user_id: str,
        action: str = "DIAGNOSE",
        details: Optional[Dict] = None,
        request_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> AuditLog:
        """
        Log a diagnosis action.
        
        Args:
            diagnosis_id: ID of diagnosis record
            patient_id: ID of patient
            user_id: ID of user who performed action
            action: Action type (DIAGNOSE, REVIEW, CONFIRM, etc.)
            details: Additional metadata
            request_id: Request tracking ID
            ip_address: User's IP address
            user_agent: Browser/client info
            
        Returns:
            Created AuditLog entry
        """
        return self._create_audit_entry(
            user_id=user_id,
            user_role="system",
            action=action,
            resource_type="diagnosis",
            resource_id=diagnosis_id,
            details={
                **(details or {}),
                "patient_id": patient_id
            },
            request_id=request_id,
            ip_address=ip_address,
            user_agent=user_agent
        )
    
    def log_data_access(
        self,
        patient_id: str,
        user_id: str,
        user_role: str,
        action: str = "READ",
        details: Optional[Dict] = None,
        ip_address: Optional[str] = None,
        success: bool = True
    ) -> AuditLog:
        """
        Log patient data access for HIPAA compliance.
        
        Args:
            patient_id: ID of patient whose data was accessed
            user_id: ID of user accessing data
            user_role: Role (doctor, nurse, admin)
            action: Type of access (READ, WRITE, DELETE)
            details: Additional context
            ip_address: User's IP
            success: Whether access was successful
            
        Returns:
            Created AuditLog entry
        """
        return self._create_audit_entry(
            user_id=user_id,
            user_role=user_role,
            action=action,
            resource_type="patient",
            resource_id=patient_id,
            details=details,
            ip_address=ip_address,
            success=success
        )
    
    def log_escalation(
        self,
        diagnosis_id: str,
        escalated_to: str,
        escalated_by: str,
        reason: str,
        details: Optional[Dict] = None
    ) -> AuditLog:
        """
        Log case escalation to specialist.
        
        Args:
            diagnosis_id: ID of diagnosis being escalated
            escalated_to: User ID of specialist
            escalated_by: User ID who initiated escalation
            reason: Escalation reason
            details: Additional metadata
            
        Returns:
            Created AuditLog entry
        """
        return self._create_audit_entry(
            user_id=escalated_by,
            user_role="system",
            action="ESCALATE",
            resource_type="diagnosis",
            resource_id=diagnosis_id,
            details={
                **(details or {}),
                "escalated_to": escalated_to,
                "reason": reason
            }
        )
    
    def log_confirmation(
        self,
        diagnosis_id: str,
        doctor_id: str,
        confirmed: bool,
        actual_diagnosis: Optional[str] = None,
        notes: Optional[str] = None
    ) -> AuditLog:
        """
        Log doctor confirmation/rejection of AI diagnosis.
        
        Args:
            diagnosis_id: ID of diagnosis
            doctor_id: ID of confirming doctor
            confirmed: Whether diagnosis was confirmed
            actual_diagnosis: Doctor's actual diagnosis if different
            notes: Doctor's notes
            
        Returns:
            Created AuditLog entry
        """
        return self._create_audit_entry(
            user_id=doctor_id,
            user_role="doctor",
            action="CONFIRM" if confirmed else "REJECT",
            resource_type="diagnosis",
            resource_id=diagnosis_id,
            details={
                "confirmed": confirmed,
                "actual_diagnosis": actual_diagnosis,
                "notes": notes
            }
        )
    
    def log_system_action(
        self,
        action: str,
        resource_type: str,
        resource_id: str,
        details: Optional[Dict] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> AuditLog:
        """
        Log automated system actions.
        
        Args:
            action: Action type
            resource_type: Type of resource
            resource_id: Resource identifier
            details: Action details
            success: Whether action succeeded
            error_message: Error if failed
            
        Returns:
            Created AuditLog entry
        """
        return self._create_audit_entry(
            user_id="system",
            user_role="system",
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            success=success,
            error_message=error_message
        )
    
    def get_audit_trail(
        self,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AuditLog]:
        """
        Query audit logs with filters.
        
        Args:
            resource_type: Filter by resource type
            resource_id: Filter by resource ID
            user_id: Filter by user
            start_date: Filter by start date
            end_date: Filter by end date
            limit: Maximum entries to return
            
        Returns:
            List of matching audit log entries
        """
        db = self._get_db()
        
        query = db.query(AuditLog)
        
        if resource_type:
            query = query.filter(AuditLog.resource_type == resource_type)
        if resource_id:
            query = query.filter(AuditLog.resource_id == resource_id)
        if user_id:
            query = query.filter(AuditLog.user_id == user_id)
        if start_date:
            query = query.filter(AuditLog.timestamp >= start_date)
        if end_date:
            query = query.filter(AuditLog.timestamp <= end_date)
        
        return query.order_by(AuditLog.timestamp.desc()).limit(limit).all()
    
    def _create_audit_entry(
        self,
        user_id: str,
        user_role: str,
        action: str,
        resource_type: str,
        resource_id: str,
        details: Optional[Dict] = None,
        request_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> AuditLog:
        """
        Internal method to create audit log entry.
        
        Returns:
            Created AuditLog entry
        """
        db = self._get_db()
        
        try:
            entry = AuditLog(
                timestamp=datetime.utcnow(),
                user_id=user_id,
                user_role=user_role,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                details=details,
                request_id=request_id,
                ip_address=ip_address,
                user_agent=user_agent,
                success=success,
                error_message=error_message
            )
            
            db.add(entry)
            db.commit()
            db.refresh(entry)
            
            logger.info(f"Audit: {action} on {resource_type}/{resource_id} by {user_id}")
            
            return entry
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create audit log: {e}")
            raise
    
    def _get_db(self) -> Session:
        """Get database session"""
        if self.db is None:
            self.db = next(get_db_session())
        return self.db


# Singleton instance
_audit_service = None


def get_audit_service(db_session: Optional[Session] = None) -> AuditService:
    """Get singleton audit service instance"""
    global _audit_service
    if _audit_service is None or db_session is not None:
        _audit_service = AuditService(db_session)
    return _audit_service
