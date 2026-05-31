"""
Configuration module for AI Chat Proxy.
Centralizes all configuration constants and settings.
"""
import os
from pathlib import Path
from typing import List, Dict, Any

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# API Configuration
GROQ_API_BASE = "https://api.groq.com/openai/v1"
MISTRAL_API_BASE = "https://api.mistral.ai/v1"
NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"

# Model configurations
GROQ_MODELS: List[str] = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "deepseek-r1-distill-llama-70b",
]

MISTRAL_MODELS: List[str] = [
    "open-mistral-nemo",
    "mistral-small-latest",
    "open-mistral-7b",
]

NVIDIA_MODELS: List[str] = [
    "meta/llama-3.3-70b-instruct",
    "mistralai/mistral-7b-instruct-v0.3",
    "nvidia/nemotron-4-340b-instruct",
]

# Rate limiting configuration
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "60"))

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
USAGE_LOG_PATH = os.getenv("USAGE_LOG_PATH", "/tmp/usage.log")

# Application metadata
APP_TITLE = "AI Chat Proxy"
APP_VERSION = "2.0.0"

# Provider mapping
PROVIDER_MAP: Dict[str, tuple] = {
    "groq": (GROQ_API_BASE, os.getenv("GROQ_API_KEY", "")),
    "mistral": (MISTRAL_API_BASE, os.getenv("MISTRAL_API_KEY", "")),
    "nvidia": (NVIDIA_API_BASE, os.getenv("NVIDIA_API_KEY", "")),
}

# Model to provider mapping
MODEL_TO_PROVIDER: Dict[str, str] = {
    model: "groq" for model in GROQ_MODELS
}
MODEL_TO_PROVIDER.update({
    model: "mistral" for model in MISTRAL_MODELS
})
MODEL_TO_PROVIDER.update({
    model: "nvidia" for model in NVIDIA_MODELS
})


def get_provider_for_model(model: str) -> tuple[str, str]:
    """
    Get API base URL and key for a given model.
    
    Args:
        model: The model identifier
        
    Returns:
        Tuple of (api_base, api_key)
        
    Raises:
        HTTPException: If model is not supported
    """
    provider = MODEL_TO_PROVIDER.get(model)
    if provider is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")
    
    return PROVIDER_MAP[provider]


def get_available_models() -> Dict[str, List[str]]:
    """
    Get all available models grouped by provider.
    
    Returns:
        Dictionary mapping provider names to lists of models
    """
    return {
        "groq": GROQ_MODELS,
        "mistral": MISTRAL_MODELS,
        "nvidia": NVIDIA_MODELS,
    }


def get_total_model_count() -> int:
    """
    Get total number of available models.
    
    Returns:
        Total count of models across all providers
    """
    return len(GROQ_MODELS) + len(MISTRAL_MODELS) + len(NVIDIA_MODELS)


# Frontend configuration
FRONTEND_PATH = Path(__file__).parent.parent / "frontend" / "index.html"