# AI Chat Proxy

OpenAI/Anthropic proxy with streaming, rate limiting, and usage tracking.

## Features

- Proxy for OpenAI and Anthropic chat completions API
- Streaming response support
- Rate limiting per IP address
- Usage tracking and logging
- CORS enabled
- Health check endpoint

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python -m app.main
```

## Configuration

Set environment variables for API keys:
- `OPENAI_API_KEY` for OpenAI
- `ANTHROPIC_API_KEY` for Anthropic

## Endpoints

- `POST /v1/chat/completions` - Proxy to OpenAI chat completions
- `POST /v1/messages` - Proxy to Anthropic messages (if implemented)
- `GET /health` - Health check

## Rate Limiting

Configurable rate limits per IP address (default: 60 requests/minute).

## Usage Tracking

Logs requests to `usage.log` with timestamp, IP, model, and token usage.