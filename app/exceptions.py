"""
Custom exceptions and error handling utilities for AI Chat Proxy.
Provides standardized error responses and exception classes.
"""
from typing import Any, Dict, Optional
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
import logging

from app.logging_config import get_logger

# Initialize logger
logger = get_logger(__name__)


class ProxyException(Exception):
    """Base exception for proxy-related errors."""
    
    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ModelNotFoundError(ProxyException):
    """Raised when a requested model is not found or not supported."""
    
    def __init__(self, model: str):
        super().__init__(
            message=f"Model '{model}' is not supported",
            status_code=status.HTTP_400_BAD_REQUEST,
            details={"model": model}
        )


class ProviderConfigurationError(ProxyException):
    """Raised when AI provider configuration is missing or invalid."""
    
    def __init__(self, provider: str):
        super().__init__(
            message=f"Provider '{provider}' is not properly configured",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details={"provider": provider}
        )


class RateLimitExceededError(ProxyException):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, client_ip: str):
        super().__init__(
            message="Rate limit exceeded",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details={"client_ip": client_ip}
        )


class UpstreamProviderError(ProxyException):
    """Raised when upstream AI provider returns an error."""
    
    def __init__(self, provider: str, status_code: int, message: str):
        super().__init__(
            message=f"Upstream provider '{provider}' error: {message}",
            status_code=status_code,
            details={"provider": provider, "upstream_status": status_code}
        )


def create_error_response(
    message: str,
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    details: Optional[Dict[str, Any]] = None,
    request: Optional[Request] = None
) -> JSONResponse:
    """
    Create a standardized error response.
    
    Args:
        message: Error message to return
        status_code: HTTP status code
        details: Optional additional error details
        request: Optional request object for logging
        
    Returns:
        JSONResponse with error details
    """
    error_content = {
        "error": True,
        "message": message,
        "status_code": status_code
    }
    
    if details:
        error_content["details"] = details
    
    # Log the error if request is provided
    if request:
        logger.error(
            f"HTTP {status_code}: {message}",
            extra={
                "status_code": status_code,
                "message": message,
                "details": details,
                "client_ip": getattr(request.client, 'host', 'unknown') if request.client else 'unknown',
                "endpoint": getattr(request.url, 'path', 'unknown'),
                "method": getattr(request, 'method', 'unknown')
            }
        )
    
    return JSONResponse(
        status_code=status_code,
        content=error_content
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Handle HTTPException instances with standardized format.
    
    Args:
        request: The FastAPI request object
        exc: The HTTPException instance
        
    Returns:
        JSONResponse with standardized error format
    """
    return create_error_response(
        message=str(exc.detail),
        status_code=exc.status_code,
        details=getattr(exc, 'details', None),
        request=request
    )


async def proxy_exception_handler(request: Request, exc: ProxyException) -> JSONResponse:
    """
    Handle ProxyException instances with standardized format.
    
    Args:
        request: The FastAPI request object
        exc: The ProxyException instance
        
    Returns:
        JSONResponse with standardized error format
    """
    return create_error_response(
        message=exc.message,
        status_code=exc.status_code,
        details=exc.details,
        request=request
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle all other unexpected exceptions.
    
    Args:
        request: The FastAPI request object
        exc: The Exception instance
        
    Returns:
        JSONResponse with standardized error format
    """
    logger.exception("Unhandled exception occurred")
    return create_error_response(
        message="Internal server error",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        details={"type": type(exc).__name__} if str(exc) else None,
        request=request
    )