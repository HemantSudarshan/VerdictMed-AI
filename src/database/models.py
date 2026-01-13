"""
CDSS Database Models
SQLAlchemy models for patient data, diagnoses, and audit logging.
"""

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, JSON, 
    ForeignKey, Text, Boolean, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()


def generate_uuid() -> str:
    """Generate a unique identifier"""
    return str(uuid.uuid4())


class Patient(Base):
    """
    Patient demographic and contact information.
    PII fields are encrypted at rest.
    """
    __tablename__ = "patients"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    age = Column(Integer)
    sex = Column(String(1))  # M, F
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
    
    # Encrypted PII fields (use encryption_key from config)
    encrypted_name = Column(Text)
    encrypted_contact = Column(Text)
    encrypted_address = Column(Text)
    
    # Medical history summary (non-PII)
    comorbidities = Column(JSON)  # List of conditions
    allergies = Column(JSON)  # Known allergies
    
    # Relationships
    diagnoses = relationship("Diagnosis", back_populates="patient", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('ix_patients_created_at', 'created_at'),
    )


class Diagnosis(Base):
    """
    AI-generated diagnosis with confidence scores and safety flags.
    Links to patient and includes full audit trail.
    """
    __tablename__ = "diagnoses"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    patient_id = Column(String(36), ForeignKey("patients.id"), nullable=True)
    
    # Input data
    symptoms_text = Column(Text)  # Raw clinical notes
    symptoms_extracted = Column(JSON)  # Extracted symptom entities
    lab_results = Column(JSON)  # Lab values
    image_path = Column(String(500))  # Path to X-ray/CT
    
    # AI Output - Primary Diagnosis
    predicted_diagnosis = Column(String(255))
    predicted_icd10 = Column(String(10))  # ICD-10 code
    confidence = Column(Float)
    confidence_interval_low = Column(Float)
    confidence_interval_high = Column(Float)
    
    # Differential diagnoses (top 5)
    differential_diagnoses = Column(JSON)
    
    # Explanation (human-readable)
    explanation = Column(Text)
    feature_importances = Column(JSON)  # SHAP values
    
    # Image analysis results
    image_findings = Column(JSON)
    image_confidence = Column(Float)
    
    # Safety flags
    conflict_detected = Column(Boolean, default=False)
    conflicts_details = Column(JSON)
    escalated_to_human = Column(Boolean, default=False)
    escalation_reason = Column(Text)
    safety_alerts = Column(JSON)  # List of alert codes
    
    # Doctor verification
    doctor_id = Column(String(36))
    doctor_confirmed = Column(Boolean, default=False)
    actual_diagnosis = Column(String(255))
    actual_icd10 = Column(String(10))
    doctor_notes = Column(Text)
    confirmed_at = Column(DateTime)
    
    # Processing metadata
    processing_time_ms = Column(Integer)
    model_versions = Column(JSON)  # {vision: "v1.0", nlp: "v1.0"}
    
    # Audit timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
    
    # Relationships
    patient = relationship("Patient", back_populates="diagnoses")
    
    __table_args__ = (
        Index('ix_diagnoses_created_at', 'created_at'),
        Index('ix_diagnoses_patient_id', 'patient_id'),
        Index('ix_diagnoses_escalated', 'escalated_to_human'),
        Index('ix_diagnoses_confidence', 'confidence'),
    )


class AuditLog(Base):
    """
    Complete audit trail for HIPAA compliance.
    Logs all data access and system actions.
    """
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Who
    user_id = Column(String(36), index=True)
    user_role = Column(String(50))  # doctor, nurse, admin, system
    
    # What
    action = Column(String(50), index=True)  # READ, WRITE, DIAGNOSE, ESCALATE, DELETE
    resource_type = Column(String(50))  # patient, diagnosis, settings
    resource_id = Column(String(36))
    
    # Details
    details = Column(JSON)  # Action-specific metadata
    request_id = Column(String(36))  # For tracing
    
    # Where
    ip_address = Column(String(45))  # IPv4 or IPv6
    user_agent = Column(String(500))
    
    # Result
    success = Column(Boolean, default=True)
    error_message = Column(Text)
    
    __table_args__ = (
        Index('ix_audit_user_timestamp', 'user_id', 'timestamp'),
        Index('ix_audit_resource', 'resource_type', 'resource_id'),
    )


class ModelMetrics(Base):
    """
    Track model performance metrics over time for drift detection.
    """
    __tablename__ = "model_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    recorded_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Model identification
    model_name = Column(String(100))
    model_version = Column(String(50))
    
    # Performance metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    
    # Volume metrics
    total_predictions = Column(Integer)
    escalations = Column(Integer)
    false_negatives = Column(Integer)
    false_positives = Column(Integer)
    
    # Latency metrics (milliseconds)
    latency_p50 = Column(Float)
    latency_p95 = Column(Float)
    latency_p99 = Column(Float)
    
    # Breakdown by category
    metrics_by_category = Column(JSON)
    
    __table_args__ = (
        Index('ix_metrics_model_date', 'model_name', 'recorded_at'),
    )
