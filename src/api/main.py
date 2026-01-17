"""
CDSS FastAPI Application
Main API for the Clinical Decision Support System.
"""

from fastapi import FastAPI, HTTPException, Depends, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import uuid
import time
from loguru import logger

from src.config import get_settings, Settings


# Request/Response models
class DiagnosisRequest(BaseModel):
    """Request model for diagnosis endpoint"""
    patient_id: Optional[str] = Field(None, description="Patient identifier")
    symptoms: str = Field(..., description="Chief complaint and symptoms")
    age: Optional[int] = Field(None, ge=0, le=150, description="Patient age")
    sex: Optional[str] = Field(None, pattern="^[MF]$", description="Patient sex (M/F)")
    lab_results: Optional[Dict] = Field(None, description="Lab values")


class DiagnosisFinding(BaseModel):
    """A single diagnostic finding"""
    disease: str
    icd10: Optional[str] = None
    confidence: float
    severity: Optional[str] = None


class DiagnosisResponse(BaseModel):
    """Response model for diagnosis endpoint"""
    request_id: str
    primary_diagnosis: Optional[DiagnosisFinding] = None
    differential_diagnoses: List[DiagnosisFinding] = []
    confidence: float
    explanation: str
    safety_alerts: List[str] = []
    requires_review: bool
    processing_time_ms: int


# Create FastAPI app
app = FastAPI(
    title="VerdictMed AI - CDSS API",
    description="Clinical Decision Support System - AI-assisted diagnostic support",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Audit Middleware (Before Auth to capture all attempts)
from src.api.middleware import AuditMiddleware
app.add_middleware(AuditMiddleware)

# Auth Dependency
from src.api.auth import get_api_key

# Exception Handlers
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "Internal Server Error", "detail": str(exc)}
    )

# Dependencies
def get_agent():
    """Get diagnostic agent (lazy loading)"""
    from src.reasoning.simple_agent import SimpleDiagnosticAgent
    return SimpleDiagnosticAgent(get_settings())


# Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint - Public"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "service": "CDSS API"
    }


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "VerdictMed AI - Clinical Decision Support System",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.post("/api/v1/diagnose", response_model=DiagnosisResponse, dependencies=[Depends(get_api_key)])
async def diagnose(request: DiagnosisRequest):
    """
    Generate diagnosis from patient symptoms.
    Protected by API Key.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    logger.info(f"[{request_id}] Diagnosis request received")
    
    try:
        # Get agent
        agent = get_agent()
        
        # Run diagnosis (async)
        result = await agent.run({
            "patient_id": request.patient_id or request_id,
            "symptoms": request.symptoms,
            "age": request.age,
            "sex": request.sex,
            "labs": request.lab_results
        })
        
        processing_time = int((time.time() - start_time) * 1000)
        
        # Build response
        primary = result.get("primary_diagnosis", {})
        
        return DiagnosisResponse(
            request_id=request_id,
            primary_diagnosis=DiagnosisFinding(
                disease=primary.get("disease", "Unknown"),
                icd10=primary.get("icd10"),
                confidence=result.get("confidence", 0.0),
                severity=primary.get("severity")
            ) if primary else None,
            differential_diagnoses=[
                DiagnosisFinding(
                    disease=d.get("disease", "Unknown"),
                    icd10=d.get("icd10"),
                    confidence=d.get("confidence", 0.0),
                    severity=d.get("severity")
                )
                for d in result.get("differential_diagnoses", [])[:5]
            ],
            confidence=result.get("confidence", 0.0),
            explanation=result.get("explanation", ""),
            safety_alerts=result.get("safety_alerts", []),
            requires_review=result.get("needs_escalation", False),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"[{request_id}] Diagnosis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/diagnose-with-image", dependencies=[Depends(get_api_key)])
async def diagnose_with_image(
    symptoms: str,
    image: UploadFile = File(...),
):
    """
    Generate diagnosis including medical image analysis.
    
    Accepts an X-ray or CT image file along with symptoms.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    logger.info(f"[{request_id}] Diagnosis with image request")
    
    try:
        import tempfile
        import os
        
        # Save uploaded image
        suffix = os.path.splitext(image.filename)[1] if image.filename else ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await image.read()
            tmp.write(content)
            image_path = tmp.name
        
        # Get agent and run (async)
        agent = get_agent()
        result = await agent.run({
            "symptoms": symptoms,
            "image_path": image_path
        })
        
        # Clean up temp file
        try:
            os.unlink(image_path)
        except:
            pass
        
        result["request_id"] = request_id
        result["processing_time_ms"] = int((time.time() - start_time) * 1000)
        
        return result
        
    except Exception as e:
        logger.error(f"[{request_id}] Image diagnosis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/symptoms")
async def list_common_symptoms():
    """List common symptoms for quick input"""
    return {
        "symptoms": [
            "fever", "cough", "shortness of breath", "chest pain",
            "headache", "fatigue", "nausea", "vomiting",
            "abdominal pain", "diarrhea", "dizziness", "weakness"
        ]
    }


# Review Workflow Endpoints

class ConfirmRequest(BaseModel):
    """Request model for diagnosis confirmation"""
    doctor_id: str = Field(..., description="ID of confirming doctor")
    confirmed: bool = Field(..., description="True if AI was correct")
    actual_diagnosis: Optional[str] = Field(None, description="Actual diagnosis if different")
    actual_icd10: Optional[str] = Field(None, description="Actual ICD-10 code")
    notes: Optional[str] = Field(None, description="Doctor's notes")


class EscalateRequest(BaseModel):
    """Request model for case escalation"""
    escalated_by: str = Field(..., description="User initiating escalation")
    specialist_id: str = Field(..., description="Specialist to escalate to")
    reason: str = Field(..., description="Escalation reason")
    notes: Optional[str] = Field(None, description="Additional notes")


@app.get("/api/v1/reviews/pending", dependencies=[Depends(get_api_key)])
async def get_pending_reviews(
    doctor_id: Optional[str] = None,
    min_confidence: Optional[float] = None,
    max_confidence: Optional[float] = None,
    escalated_only: bool = False,
    limit: int = 50
):
    """
    Get diagnoses pending doctor review.
    Returns cases needing human-in-the-loop confirmation.
    """
    try:
        from src.services.review_service import get_review_service
        
        review_service = get_review_service()
        pending = review_service.get_pending_reviews(
            doctor_id=doctor_id,
            min_confidence=min_confidence,
            max_confidence=max_confidence,
            escalated_only=escalated_only,
            limit=limit
        )
        
        return {
            "count": len(pending),
            "reviews": pending
        }
        
    except Exception as e:
        logger.error(f"Failed to get pending reviews: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/reviews/{diagnosis_id}/confirm", dependencies=[Depends(get_api_key)])
async def confirm_diagnosis(diagnosis_id: str, request: ConfirmRequest):
    """
    Doctor confirms or rejects AI diagnosis.
    Records actual diagnosis for accuracy tracking.
    """
    try:
        from src.services.review_service import get_review_service
        
        review_service = get_review_service()
        result = review_service.confirm_diagnosis(
            diagnosis_id=diagnosis_id,
            doctor_id=request.doctor_id,
            confirmed=request.confirmed,
            actual_diagnosis=request.actual_diagnosis,
            actual_icd10=request.actual_icd10,
            notes=request.notes
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to confirm diagnosis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/reviews/{diagnosis_id}/escalate", dependencies=[Depends(get_api_key)])
async def escalate_case(diagnosis_id: str, request: EscalateRequest):
    """
    Escalate case to specialist for second opinion.
    Adds to specialist's review queue.
    """
    try:
        from src.services.review_service import get_review_service
        
        review_service = get_review_service()
        result = review_service.escalate_to_specialist(
            diagnosis_id=diagnosis_id,
            escalated_by=request.escalated_by,
            specialist_id=request.specialist_id,
            reason=request.reason,
            notes=request.notes
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to escalate case: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/reviews/stats", dependencies=[Depends(get_api_key)])
async def get_review_stats(
    doctor_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Get review statistics for monitoring.
    Returns accuracy, escalation rate, and volume metrics.
    """
    try:
        from src.services.review_service import get_review_service
        from datetime import datetime
        
        review_service = get_review_service()
        
        # Parse dates if provided
        start = datetime.fromisoformat(start_date) if start_date else None
        end = datetime.fromisoformat(end_date) if end_date else None
        
        stats = review_service.get_review_stats(
            doctor_id=doctor_id,
            start_date=start,
            end_date=end
        )
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get review stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Prometheus metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        from fastapi.responses import Response
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except ImportError:
        return {"error": "prometheus_client not installed"}


# Add disclaimer
DISCLAIMER = """
⚠️ IMPORTANT: This system provides AI-assisted diagnostic suggestions only.
All outputs require verification by a qualified healthcare provider.
This is NOT a replacement for professional medical judgment.
"""


@app.get("/api/v1/disclaimer")
async def get_disclaimer():
    """Get medical AI disclaimer"""
    return {"disclaimer": DISCLAIMER}

