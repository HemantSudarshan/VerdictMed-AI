"""
API Authentication
Implement API Key authentication/authorization dependency.
"""

from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader, APIKeyQuery
from loguru import logger
from typing import Optional

from src.config import get_settings

# define security scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)


async def get_api_key(
    api_key_header: str = Security(api_key_header),
    api_key_query: str = Security(api_key_query)
) -> str:
    """
    Validate API Key from header or query param.
    
    Args:
        api_key_header: Key from X-API-Key header
        api_key_query: Key from api_key query param
        
    Returns:
        The valid API key
        
    Raises:
        HTTPException: If key is missing or invalid
    """
    settings = get_settings()
    
    # In a real app, you might look up keys in a DB to get user info/scopes
    # For MVP, we use a single secret key from config (or a hardcoded list)
    valid_keys = [settings.secret_key, "dev-test-key"]
    
    if api_key_header in valid_keys:
        return api_key_header
        
    if api_key_query in valid_keys:
        return api_key_query
        
    logger.warning(f"Failed auth attempt. Header: {api_key_header}, Query: {api_key_query}")
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "ApiKey"},
    )
