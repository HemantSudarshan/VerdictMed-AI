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
