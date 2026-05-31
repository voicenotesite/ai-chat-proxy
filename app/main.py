"""
AI Chat Proxy - Main Application Module
Enterprise-grade FastAPI application for proxying AI model requests.
"""
import json
import time
from pathlib import Path
from typing import Dict, Any
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
import httpx

# Import application modules
from app.config import (
    get_provider_for_model,
    get_available_models,
    get_total_model_count,
    FRONTEND_PATH
)
from app.logging_config import log_usage, get_logger
from app.exceptions import (
    ProxyException,
    ModelNotFoundError,
    ProviderConfigurationError,
    RateLimitExceededError,
    UpstreamProviderError,
    http_exception_handler,
    proxy_exception_handler,
    general_exception_handler
)

# Initialize logger
logger = get_logger(__name__)

# Create FastAPI application
app = FastAPI(title="AI Chat Proxy", version="2.0.0")

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add exception handlers
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(ProxyException, proxy_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# In-memory rate limiting (consider Redis for production)
from collections import defaultdict
import time as time_module

request_counts = defaultdict(list)
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX_REQUESTS = 60  # requests per window
LAST_CLEANUP = time_module.time()
CLEANUP_INTERVAL = 300  # Clean up old entries every 5 minutes


async def check_rate_limit(request: Request) -> None:
    """
    Check if the client has exceeded the rate limit.
    
    Args:
        request: The FastAPI request object
        
    Raises:
        RateLimitExceededError: If rate limit is exceeded
    """
    global LAST_CLEANUP
    
    client_ip = "unknown"
    if request.client:
        client_ip = request.client.host or "unknown"
    
    now = time_module.time()
    
    # Periodic cleanup of old entries to prevent memory leak
    if now - LAST_CLEANUP > CLEANUP_INTERVAL:
        cutoff_time = now - RATE_LIMIT_WINDOW
        for ip in list(request_counts.keys()):
            # Remove old requests
            request_counts[ip] = [req_time for req_time in request_counts[ip] if req_time > cutoff_time]
            # Remove IP entry if no recent requests
            if not request_counts[ip]:
                del request_counts[ip]
        LAST_CLEANUP = now
    
    # Clean old requests for this IP and check rate limit
    cutoff_time = now - RATE_LIMIT_WINDOW
    request_counts[client_ip] = [
        req_time for req_time in request_counts[client_ip]
        if req_time > cutoff_time
    ]
    
    # Check if limit exceeded
    if len(request_counts[client_ip]) >= RATE_LIMIT_MAX_REQUESTS:
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        raise RateLimitExceededError(client_ip=client_ip)
    
    # Add current request
    request_counts[client_ip].append(now)


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the frontend HTML page."""
    try:
        if FRONTEND_PATH.exists():
            frontend_html = FRONTEND_PATH.read_text(encoding="utf-8")
            return HTMLResponse(content=frontend_html)
    except Exception as e:
        logger.error(f"Failed to read frontend file: {e}")
        raise ProxyException(
            message="Frontend not available",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"error": str(e)}
        )
    
    # Fallback response if frontend cannot be loaded
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "ok", "message": "AI Chat Proxy API"}
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "providers": list(get_available_models().keys()),
        "models": get_total_model_count(),
    }


@app.api_route("/v1/chat/completions", methods=["GET", "POST"])
async def proxy_chat_completions(request: Request):
    """
    Proxy chat completion requests to AI model providers.
    
    Args:
        request: The FastAPI request object
        
    Returns:
        StreamingResponse or JSONResponse with the AI model's response
        
    Raises:
        ProxyException: For various error conditions
    """
    try:
        # Read request body
        body = await request.body()
        
        # Parse JSON body
        try:
            json_body = json.loads(body) if body else {}
        except json.JSONDecodeError:
            json_body = {}
        
        # Extract parameters
        model = json_body.get("model", "gpt-3.5-turbo")
        stream = json_body.get("stream", False)
        
        # Get provider configuration
        try:
            api_base, api_key = get_provider_for_model(model)
        except HTTPException as e:
            raise ModelNotFoundError(model=model)
        
        # Check if API key is configured
        if not api_key:
            # Extract provider from model to give better error
            from app.config import MODEL_TO_PROVIDER
            provider = MODEL_TO_PROVIDER.get(model, "unknown")
            raise ProviderConfigurationError(provider=provider)
        
        # Prepare headers for upstream request
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        # Handle streaming requests
        if stream:
            return StreamingResponse(
                _proxy_stream(api_base, headers, body),
                media_type="text/event-stream",
            )
        
        # Handle non-streaming requests
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{api_base}/chat/completions",
                headers=headers,
                content=body,
                timeout=60.0,
            )
            
            # Handle HTTP errors from upstream provider
            if resp.status_code >= 400:
                error_detail = "Unknown error"
                try:
                    error_data = resp.json()
                    error_detail = error_data.get("error", {}).get("message", str(error_data))
                except:
                    error_detail = resp.text or f"HTTP {resp.status_code}"
                
                raise UpstreamProviderError(
                    provider=model.split("/")[0] if "/" in model else model,
                    status_code=resp.status_code,
                    message=error_detail
                )
            
            # Parse successful response
            data = resp.json()
            
            # Log usage
            await log_usage(request, response_data=data)
            
            return JSONResponse(content=data, status_code=resp.status_code)
            
    except ProxyException:
        # Re-raise proxy exceptions as-is (they'll be handled by exception handlers)
        raise
    except httpx.RequestError as e:
        # Handle upstream request errors (network issues, timeouts, etc.)
        await log_usage(request, error=str(e))
        logger.error(f"Upstream request error: {e}")
        raise UpstreamProviderError(
            provider="unknown",
            status_code=status.HTTP_502_BAD_GATEWAY,
            message=f"Request error: {str(e)}"
        )
    except Exception as e:
        # Handle all other unexpected errors
        await log_usage(request, error=str(e))
        logger.exception("Unexpected error in proxy_chat_completions")
        raise ProxyException(
            message="Internal server error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"type": type(e).__name__, "error": str(e)}
        )


async def _proxy_stream(api_base: str, headers: dict, body: bytes):
    """
    Proxy streaming response from upstream AI provider.
    
    Args:
        api_base: Base URL of the AI provider API
        headers: HTTP headers for the request
        body: Request body to send
        
    Yields:
        Bytes chunks from the upstream response
    """
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST", f"{api_base}/chat/completions",
            headers=headers, content=body, timeout=60.0,
        ) as resp:
            # Check for HTTP errors in streaming response
            if resp.status_code >= 400:
                # For streaming, we need to read the error response
                error_content = b""
                async for chunk in resp.aiter_bytes():
                    error_content += chunk
                
                error_message = "Unknown error"
                try:
                    error_data = json.loads(error_content.decode())
                    error_message = error_data.get("error", {}).get("message", str(error_data))
                except:
                    error_message = error_content.decode() or f"HTTP {resp.status_code}"
                
                raise UpstreamProviderError(
                    provider="unknown",
                    status_code=resp.status_code,
                    message=error_message
                )
            
            # Stream successful response
            async for chunk in resp.aiter_bytes():
                yield chunk


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
