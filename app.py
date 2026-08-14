"""
A Fountain from Famine — unified OpenAI-compatible proxy.

Routes requests to three free model providers based on model name:
  - NIM models (glm-5.2, minimax-m3, nemotron-3-ultra, inkling) -> NVIDIA NIM API
  - DeepSeek models (deepseek-v4-pro) -> DanyAPI reverse-engineered web client
  - Qwen models (qwen3.8-max, etc.) -> DanyAPI reverse-engineered web client

Single FastAPI app, single port, single auth key. Each provider manages its
own rate limits and account rotation independently.
"""

from __future__ import annotations

import os
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MASTER_KEY = os.environ.get("LITELLM_MASTER_KEY") or os.environ.get("FOUNTAIN_MASTER_KEY")

# NIM configuration
NIM_API_BASE = "https://integrate.api.nvidia.com/v1"
NIM_COOLDOWN_SECONDS = 65
NIM_MODEL_MAP = {
    "glm-5.2": "z-ai/glm-5.2",
    "minimax-m3": "minimaxai/minimax-m3",
    "nemotron-3-ultra": "nvidia/nemotron-3-ultra-550b-a55b",
    "inkling": "thinkingmachines/inkling",
}

# DanyAPI configuration
DANY_MODELS = {
    "deepseek-v4-pro": "deepseek",
    "deepseek-v4-flash": "deepseek",
    "deepseek-v4-vision": "deepseek",
    # Qwen models are fetched at startup, but we list defaults here
    "qwen3.8-max": "qwen",
    "qwen3.7-plus": "qwen",
    "qwen3.7-max": "qwen",
}


def _provider_for_model(model_name: str) -> str | None:
    """Determine which provider handles a given model name."""
    if model_name in NIM_MODEL_MAP:
        return "nim"
    if model_name.startswith("deepseek-"):
        return "deepseek"
    if model_name.startswith("qwen"):
        return "qwen"
    return None


# ---------------------------------------------------------------------------
# NIM Oscillator (from ccc_eos_nvidia_nim_oscillator, built with Claude)
# ---------------------------------------------------------------------------

def _load_nim_keys(env_var: str, prefix: str) -> list[dict]:
    raw = os.environ.get(env_var, "") or ""
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    return [{"label": f"{prefix}{i+1}", "api_key": k} for i, k in enumerate(keys)]


nim_accounts: list[dict] = []
nim_request_log: dict[str, deque] = {}
nim_cooldown_until: dict[str, float] = {}


def _nim_recent_count(label: str) -> int:
    now = time.time()
    dq = nim_request_log.get(label, deque())
    while dq and dq[0] < now - 60:
        dq.popleft()
    return len(dq)


def _nim_record_request(label: str) -> None:
    nim_request_log[label].append(time.time())


def _nim_ranked_accounts() -> list[dict]:
    now = time.time()
    available = [a for a in nim_accounts if nim_cooldown_until.get(a["label"], 0) <= now]
    if not available:
        available = sorted(nim_accounts, key=lambda a: nim_cooldown_until.get(a["label"], 0))
    return sorted(available, key=lambda a: _nim_recent_count(a["label"]))


class _NimRateLimited(Exception):
    pass


async def _nim_forward(account: dict, upstream_body: dict, is_stream: bool):
    label = account["label"]
    headers = {
        "Authorization": f"Bearer {account['api_key']}",
        "Content-Type": "application/json",
    }
    url = f"{NIM_API_BASE}/chat/completions"

    if is_stream:
        client = httpx.AsyncClient(timeout=120)
        req = client.build_request("POST", url, headers=headers, json=upstream_body)
        upstream = await client.send(req, stream=True)
        if upstream.status_code == 429:
            await upstream.aclose()
            await client.aclose()
            raise _NimRateLimited()
        if upstream.status_code >= 400:
            detail = await upstream.aread()
            await upstream.aclose()
            await client.aclose()
            raise httpx.HTTPError(f"NIM ({label}) returned {upstream.status_code}: {detail[:300]}")

        _nim_record_request(label)

        async def event_stream():
            try:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
            finally:
                await upstream.aclose()
                await client.aclose()

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(url, headers=headers, json=upstream_body)
        if resp.status_code == 429:
            raise _NimRateLimited()
        if resp.status_code >= 400:
            raise httpx.HTTPError(f"NIM ({label}) returned {resp.status_code}: {resp.text[:300]}")
        _nim_record_request(label)
        return JSONResponse(content=resp.json(), status_code=resp.status_code)


async def _nim_completion(body: dict) -> Any:
    requested_model = body.get("model")
    nim_model = NIM_MODEL_MAP.get(requested_model)
    if not nim_model:
        raise HTTPException(status_code=400, detail=f"Unknown NIM model '{requested_model}'")

    upstream_body = dict(body)
    upstream_body["model"] = nim_model
    is_stream = bool(body.get("stream"))

    last_error = None
    for account in _nim_ranked_accounts():
        if not account["api_key"]:
            continue
        try:
            result = await _nim_forward(account, upstream_body, is_stream)
            if result is not None:
                return result
        except _NimRateLimited:
            nim_cooldown_until[account["label"]] = time.time() + NIM_COOLDOWN_SECONDS
            last_error = "rate_limited"
            continue
        except httpx.HTTPError as exc:
            last_error = str(exc)
            continue

    raise HTTPException(
        status_code=503,
        detail=f"All NIM keys failed or are rate-limited (last error: {last_error})",
    )


# ---------------------------------------------------------------------------
# DanyAPI integration (from FANATFANATA/DanyAPI)
# ---------------------------------------------------------------------------

# We import DanyAPI lazily so the app can start even if DanyAPI credentials
# are not configured (NIM-only mode).
_dany_app = None
_dany_initialized = False


async def _init_dany():
    """Initialize the DanyAPI FastAPI sub-app if credentials are available."""
    global _dany_app, _dany_initialized
    has_deepseek = bool(os.environ.get("DEEPSEEK_TOKENS") or os.environ.get("DEEPSEEK_TOKEN") or os.environ.get("DEEPSEEK_EMAIL"))
    has_qwen = bool(os.environ.get("QWEN_TOKENS") or os.environ.get("QWEN_TOKEN") or os.environ.get("QWEN_EMAIL"))
    if not has_deepseek and not has_qwen:
        _dany_initialized = True
        return
    try:
        from danyapi.api.openai import app as dany_fastapi_app, lifespan as dany_lifespan
        # Enter the DanyAPI lifespan to initialize account pools
        # We use a dummy app to run the lifespan
        _dany_app = dany_fastapi_app
        # The lifespan will be managed by our unified lifespan below
        _dany_initialized = True
    except Exception as exc:
        import logging
        logging.getLogger("fountain").warning("DanyAPI init failed: %s", exc)
        _dany_initialized = True


# ---------------------------------------------------------------------------
# Unified app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize NIM accounts
    global nim_accounts, nim_request_log, nim_cooldown_until
    nim_accounts = _load_nim_keys("NIM_API_KEY_CCC", "ccc") + _load_nim_keys("NIM_API_KEY_EOS", "eos")
    nim_request_log = {a["label"]: deque() for a in nim_accounts}
    nim_cooldown_until = {a["label"]: 0.0 for a in nim_accounts}

    nim_count = len(nim_accounts)
    if nim_count:
        print(f"[fountain] NIM keys loaded: {nim_count}")
    else:
        print("[fountain] No NIM keys configured (NIM models will be unavailable)")

    # Initialize DanyAPI
    dany_ctx = None
    has_deepseek = bool(os.environ.get("DEEPSEEK_TOKENS") or os.environ.get("DEEPSEEK_EMAIL"))
    has_qwen = bool(os.environ.get("QWEN_TOKENS") or os.environ.get("QWEN_EMAIL"))
    if has_deepseek or has_qwen:
        try:
            from danyapi.api.openai import lifespan as dany_lifespan
            dany_ctx = dany_lifespan(app)
            await dany_ctx.__aenter__()
            print("[fountain] DanyAPI initialized (DeepSeek + Qwen)")
        except Exception as exc:
            print(f"[fountain] DanyAPI init failed: {exc}")
            dany_ctx = None
    else:
        print("[fountain] No DanyAPI credentials configured (DeepSeek/Qwen models will be unavailable)")

    yield

    # Cleanup
    if dany_ctx is not None:
        try:
            await dany_ctx.__aexit__(None, None, None)
        except Exception:
            pass


app = FastAPI(title="A Fountain from Famine", lifespan=lifespan)


def _check_auth(request: Request) -> None:
    auth = request.headers.get("authorization", "")
    token = auth[7:] if auth.lower().startswith("bearer ") else ""
    if not MASTER_KEY or token != MASTER_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "nim_keys": len(nim_accounts),
        "deepseek": bool(os.environ.get("DEEPSEEK_TOKENS") or os.environ.get("DEEPSEEK_EMAIL")),
        "qwen": bool(os.environ.get("QWEN_TOKENS") or os.environ.get("QWEN_EMAIL")),
    }


@app.get("/v1/models")
async def list_models(request: Request):
    _check_auth(request)
    models = []

    # NIM models
    for name in NIM_MODEL_MAP:
        models.append({"id": name, "object": "model", "owned_by": "nim-oscillator"})

    # DanyAPI models (DeepSeek + Qwen)
    has_deepseek = bool(os.environ.get("DEEPSEEK_TOKENS") or os.environ.get("DEEPSEEK_EMAIL"))
    has_qwen = bool(os.environ.get("QWEN_TOKENS") or os.environ.get("QWEN_EMAIL"))
    if has_deepseek:
        for name in ["deepseek-v4-pro", "deepseek-v4-flash", "deepseek-v4-vision"]:
            models.append({"id": name, "object": "model", "owned_by": "dany-deepseek"})
    if has_qwen:
        for name in ["qwen3.8-max", "qwen3.7-plus", "qwen3.7-max"]:
            models.append({"id": name, "object": "model", "owned_by": "dany-qwen"})

    return {"object": "list", "data": models}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    _check_auth(request)
    body = await request.json()
    model = body.get("model", "")
    provider = _provider_for_model(model)

    if provider is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model}'. Known: {list(NIM_MODEL_MAP)} + deepseek-* + qwen*",
        )

    if provider == "nim":
        if not nim_accounts:
            raise HTTPException(status_code=503, detail="NIM provider not configured")
        return await _nim_completion(body)

    if provider in ("deepseek", "qwen"):
        # Route through DanyAPI's handler
        if not hasattr(app.state, "pool") and not hasattr(app.state, "qwen_pool"):
            raise HTTPException(status_code=503, detail=f"{provider} provider not configured")
        # Import and call DanyAPI's chat completion handler
        from danyapi.api.openai import chat_completions as _dany_chat
        return await _dany_chat(request)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
