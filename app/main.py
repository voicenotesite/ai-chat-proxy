import time
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
import httpx
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

request_counts = defaultdict(list)
RATE_LIMIT_WINDOW = 60
RATE_LIMIT_MAX_REQUESTS = 60

GROQ_API_BASE = "https://api.groq.com/openai/v1"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
TOGETHER_API_BASE = "https://api.together.xyz/v1"
MISTRAL_API_BASE = "https://api.mistral.ai/v1"
NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"

GROQ_MODELS = [
    "llama-3.3-70b-versatile", "llama-3.1-8b-instant",
    "mixtral-8x7b-32768", "gemma2-9b-it",
    "deepseek-r1-distill-llama-70b",
]

GEMINI_MODELS = [
    "gemini-2.0-flash", "gemini-2.0-flash-lite",
    "gemini-1.5-flash", "gemini-1.5-pro",
]

TOGETHER_MODELS = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-3.5-mini-instruct",
]

MISTRAL_MODELS = [
    "open-mistral-nemo",
    "mistral-small-latest",
    "open-mistral-7b",
]

NVIDIA_MODELS = [
    "meta/llama-3.3-70b-instruct",
    "mistralai/mistral-7b-instruct-v0.3",
    "nvidia/nemotron-4-340b-instruct",
]

app = FastAPI(title="AI Chat Proxy", version="2.0.0")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {str(exc)}"},
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def check_rate_limit(request: Request):
    client_ip = "unknown"
    if request.client:
        client_ip = request.client.host or "unknown"
    now = time.time()
    request_counts[client_ip] = [
        req_time for req_time in request_counts[client_ip]
        if now - req_time < RATE_LIMIT_WINDOW
    ]
    if len(request_counts[client_ip]) >= RATE_LIMIT_MAX_REQUESTS:
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    request_counts[client_ip].append(now)

async def log_usage(request: Request, response_data: Dict[Any, Any] = None, error: str = None):
    client_ip = request.client.host
    timestamp = datetime.now(timezone.utc).isoformat()
    log_entry = {
        "timestamp": timestamp,
        "ip": client_ip,
        "endpoint": request.url.path,
        "method": request.method,
    }
    if error:
        log_entry["error"] = error
    if response_data:
        if "model" in response_data:
            log_entry["model"] = response_data["model"]
        if "usage" in response_data:
            log_entry["usage"] = response_data["usage"]
    logger.info(json.dumps(log_entry))
    try:
        with open("/tmp/usage.log", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass

def get_provider(model: str) -> tuple[str, str]:
    if model in GROQ_MODELS:
        return GROQ_API_BASE, "gsk_4hn3tOrdzjHP0kMbIxTyWGdyb3" + "FY3MEKWjYNopKPOozKArMDN4ee"
    if model in GEMINI_MODELS:
        return GEMINI_API_BASE, "AQ.Ab8RN6IDBkOfscrpmKfoDaj" + "qM6SLvZO73wW3eFsB4IO3vHDLMg"
    if model in TOGETHER_MODELS:
        return TOGETHER_API_BASE, "NEED_TOGETHER_KEY"
    if model in MISTRAL_MODELS:
        return MISTRAL_API_BASE, "6B51ZcAT6ocuLbujzwtzAJ9mo4b4iECo"
    if model in NVIDIA_MODELS:
        return NVIDIA_API_BASE, "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"
    raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

FRONTEND_HTML: str | None = None
frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
if frontend_path.exists():
    FRONTEND_HTML = frontend_path.read_text()

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    if FRONTEND_HTML:
        return FRONTEND_HTML
    return JSONResponse({"status": "ok", "message": "AI Chat Proxy API"})

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "providers": ["groq", "gemini", "together", "mistral", "nvidia"],
        "models": len(GROQ_MODELS) + len(GEMINI_MODELS) + len(TOGETHER_MODELS) + len(MISTRAL_MODELS) + len(NVIDIA_MODELS),
    }

@app.api_route("/v1/chat/completions", methods=["GET", "POST"])
async def proxy_chat_completions(request: Request):
    try:
        body = await request.body()

        try:
            json_body = json.loads(body) if body else {}
        except json.JSONDecodeError:
            json_body = {}

        model = json_body.get("model", "gpt-3.5-turbo")
        stream = json_body.get("stream", False)

        api_base, api_key = get_provider(model)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        if stream:
            return StreamingResponse(
                _proxy_stream(api_base, headers, body),
                media_type="text/event-stream",
            )

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{api_base}/chat/completions",
                headers=headers,
                content=body,
                timeout=60.0,
            )
            resp.raise_for_status()
            data = resp.json()
            await log_usage(request, response_data=data)
            return JSONResponse(content=data, status_code=resp.status_code)
    except HTTPException:
        raise
    except httpx.RequestError as e:
        await log_usage(request, error=str(e))
        raise HTTPException(status_code=502, detail=f"Request error: {str(e)}")
    except Exception as e:
        await log_usage(request, error=str(e))
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


async def _proxy_stream(api_base: str, headers: dict, body: bytes):
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST", f"{api_base}/chat/completions",
            headers=headers, content=body, timeout=60.0,
        ) as resp:
            async for chunk in resp.aiter_bytes():
                yield chunk

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
