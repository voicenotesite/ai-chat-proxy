"""
Logging configuration module for AI Chat Proxy.
Provides centralized logging setup and usage logging functionality.
"""
import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Usage log path
USAGE_LOG_PATH = os.getenv("USAGE_LOG_PATH", "/tmp/usage.log")


def log_usage(
    request: Any,
    response_data: Optional[Dict[Any, Any]] = None,
    error: Optional[str] = None
) -> None:
    """
    Log usage information for monitoring and analytics.
    
    Args:
        request: The FastAPI request object
        response_data: Optional response data to log
        error: Optional error message if request failed
    """
    try:
        # Get client IP safely
        client_ip = "unknown"
        if hasattr(request, 'client') and request.client:
            client_ip = request.client.host or "unknown"
        
        # Create log entry
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ip": client_ip,
            "endpoint": getattr(request, 'url', {}).get('path', 'unknown'),
            "method": getattr(request, 'method', 'unknown'),
        }
        
        if error:
            log_entry["error"] = error
            
        if response_data:
            if isinstance(response_data, dict):
                if "model" in response_data:
                    log_entry["model"] = response_data["model"]
                if "usage" in response_data:
                    log_entry["usage"] = response_data["usage"]
        
        # Log to application logger
        logger.info(json.dumps(log_entry))
        
        # Log to file if path is configured
        if USAGE_LOG_PATH:
            try:
                with open(USAGE_LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry) + "\n")
            except Exception as e:
                logger.warning(f"Failed to write to usage log: {e}")
                
    except Exception as e:
        logger.error(f"Error in usage logging: {e}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)