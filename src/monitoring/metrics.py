"""
Prometheus Metrics for CDSS
Exposes application metrics for monitoring and alerting.
"""

from prometheus_client import Counter, Histogram, Gauge, Info
import time
from functools import wraps
from loguru import logger


# Application Info
app_info = Info('cdss_app', 'CDSS application information')
app_info.info({
    'version': '1.0.0',
    'name': 'VerdictMed AI CDSS'
})

# Request Metrics
REQUEST_COUNT = Counter(
    'cdss_requests_total',
    'Total number of requests',
    ['endpoint', 'method', 'status']
)

REQUEST_LATENCY = Histogram(
    'cdss_request_latency_seconds',
    'Request latency in seconds',
    ['endpoint'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# Diagnosis Metrics
DIAGNOSIS_COUNT = Counter(
    'cdss_diagnoses_total',
    'Total number of diagnoses generated',
    ['primary_diagnosis', 'escalated']
)

DIAGNOSIS_CONFIDENCE = Histogram(
    'cdss_diagnosis_confidence',
    'Distribution of diagnosis confidence scores',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Safety Metrics
SAFETY_ALERTS = Counter(
    'cdss_safety_alerts_total',
    'Number of safety alerts triggered',
    ['alert_type']
)

ESCALATIONS = Counter(
    'cdss_escalations_total',
    'Number of cases requiring physician escalation',
    ['reason']
)

LOW_CONFIDENCE_COUNT = Counter(
    'cdss_low_confidence_total',
    'Number of diagnoses with confidence below threshold'
)

# Cache Metrics
CACHE_HITS = Counter(
    'cdss_cache_hits_total',
    'Number of cache hits'
)

CACHE_MISSES = Counter(
    'cdss_cache_misses_total',
    'Number of cache misses'
)

# Model Metrics
INFERENCE_TIME = Histogram(
    'cdss_inference_time_seconds',
    'Time taken for model inference',
    ['model'],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

ACTIVE_REQUESTS = Gauge(
    'cdss_active_requests',
    'Number of currently active requests'
)


def track_request(endpoint: str):
    """Decorator to track request metrics"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            ACTIVE_REQUESTS.inc()
            start_time = time.time()
            status = "success"
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                ACTIVE_REQUESTS.dec()
                duration = time.time() - start_time
                REQUEST_COUNT.labels(
                    endpoint=endpoint,
                    method="POST",
                    status=status
                ).inc()
                REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
        
        return wrapper
    return decorator


def record_diagnosis(diagnosis_result: dict):
    """Record metrics for a diagnosis result"""
    try:
        primary = diagnosis_result.get("primary_diagnosis", {})
        disease = primary.get("disease", "unknown")[:50]  # Limit cardinality
        escalated = str(diagnosis_result.get("needs_escalation", False))
        confidence = diagnosis_result.get("confidence", 0)
        
        DIAGNOSIS_COUNT.labels(
            primary_diagnosis=disease,
            escalated=escalated
        ).inc()
        
        DIAGNOSIS_CONFIDENCE.observe(confidence)
        
        if confidence < 0.55:
            LOW_CONFIDENCE_COUNT.inc()
        
        # Record safety alerts
        for alert in diagnosis_result.get("safety_alerts", []):
            SAFETY_ALERTS.labels(alert_type=alert[:30]).inc()
        
        if diagnosis_result.get("needs_escalation"):
            reason = diagnosis_result.get("escalation_reason", "unknown")[:50]
            ESCALATIONS.labels(reason=reason).inc()
            
    except Exception as e:
        logger.warning(f"Failed to record diagnosis metrics: {e}")


def record_cache_hit():
    """Record a cache hit"""
    CACHE_HITS.inc()


def record_cache_miss():
    """Record a cache miss"""
    CACHE_MISSES.inc()


def record_inference_time(model: str, duration: float):
    """Record model inference time"""
    INFERENCE_TIME.labels(model=model).observe(duration)
