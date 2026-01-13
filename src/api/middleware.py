"""
Audit Middleware
Log all API requests and responses for HIPAA compliance/audit trail.
"""

from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
import time
import json
import uuid
from loguru import logger
from datetime import datetime

from src.database.session import get_db_session
from src.database.models import AuditLog


class AuditMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log request/response details to AuditLog table.
    """
    
    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        start_time = time.time()
        
        # Capture request details (careful not to consume body stream if needed later)
        # For simplicity, we log metadata. Body logging requires specialized handling.
        client_host = request.client.host if request.client else "unknown"
        method = request.method
        url = str(request.url)
        
        # Get user ID from state (set by Auth dependency if available)
        # Note: Middleware runs before dependency, so auth info might not be available yet
        # unless stored in earlier middleware or verified differently.
        # Here we just placeholder.
        user_id = "anonymous" 
        
        response = None
        error_message = None
        success = True
        
        try:
            response = await call_next(request)
            
        except Exception as e:
            success = False
            error_message = str(e)
            logger.error(f"Audit middleware caught error: {e}")
            raise e
            
        finally:
            try:
                # Calculate duration
                duration_ms = int((time.time() - start_time) * 1000)
                
                # Use background task or fire-and-forget to avoid blocking response
                # For this implementation, we do a quick synchronous write or use a task queue
                # We'll use a direct sync write for simplicity but in prod use async/background
                self._log_audit_entry(
                    request_id=request_id,
                    user_id=user_id,
                    action=f"{method} {request.url.path}",
                    resource="api",
                    details={
                        "method": method,
                        "url": url,
                        "client_ip": client_host,
                        "duration_ms": duration_ms,
                        "status_code": response.status_code if response else 500
                    },
                    success=success,
                    error=error_message
                )
                
            except Exception as log_error:
                logger.error(f"Failed to write audit log: {log_error}")
        
        return response

    def _log_audit_entry(self, request_id, user_id, action, resource, details, success, error):
        """Write entry to DB"""
        try:
            with get_db_session() as db:
                entry = AuditLog(
                    timestamp=datetime.utcnow(),
                    user_id=user_id,
                    user_role="api_client",
                    action=action,
                    resource_type=resource,
                    resource_id=None,
                    details=details,
                    request_id=request_id,
                    ip_address=details.get("client_ip"),
                    success=success,
                    error_message=error
                )
                db.add(entry)
                # db.commit() - managed by context manager
        except Exception as e:
            logger.error(f"DB Audit Write Failed: {e}")
