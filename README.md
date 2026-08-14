# A Fountain from Famine

An OpenAI-compatible API proxy that unifies three free model providers — NVIDIA NIM, DeepSeek, and Qwen — behind a single endpoint. Built from scarcity, not excess.

## What this is

Three of the strongest open-weight frontier models — each with a million-token context window — are available for free through different channels. This project brings them together behind one OpenAI-compatible API so that agents, coding assistants, and conversational applications can use any of them without juggling multiple endpoints, API keys, or rate-limit strategies.

**The three providers:**

| Provider | Models | How it's accessed | Rate-limit strategy |
|---|---|---|---|
| NVIDIA NIM | GLM-5.2, MiniMax-M3, Nemotron-3-Ultra, Inkling | Official NIM API with a pool of keys across two accounts | Per-key cooldown (65s after 429), least-busy-first routing |
| DeepSeek | DeepSeek V4 Pro | Reverse-engineered web client (chat.deepseek.com) | Token pool, one concurrent generation per account |
| Qwen | Qwen 3.8 Max | Reverse-engineered web client (chat.qwen.ai) | Token pool, one concurrent generation per account |

All requests flow through a single FastAPI application. The model name in the request determines which provider handles it. Streaming, thinking traces, and tool calling are supported where the upstream provider offers them.

## Why it exists

The short version: some people have budgets. Some don't.

This project was built by someone who has been parched of access to capable AI tools for years and finally found a fountain — free tiers, reverse-engineered web clients, multi-key pools — and decided to build something that could drink from all of them at once, reliably, without hammering any single source until it breaks.

The name comes from that: a fountain from famine. Abundance built from scarcity.

## Architecture

```
Client (Letta, OpenAI SDK, curl, anything)
    |
    v
+----------------------------------+
|     A Fountain from Famine       |
|  (single FastAPI app, port 8000) |
|                                  |
|  /v1/chat/completions            |
|    model="glm-5.2"      -> NIM   |
|    model="minimax-m3"   -> NIM   |
|    model="nemotron-3-ultra" -> NIM |
|    model="inkling"      -> NIM   |
|    model="deepseek-v4-pro" -> DeepSeek |
|    model="qwen3.8-max"  -> Qwen  |
|                                  |
|  /v1/models  (all models)       |
|  /health     (per-provider)      |
+----------------------------------+
    |           |          |
    v           v          v
 NVIDIA NIM  DeepSeek    Qwen
 (key pool)  (tokens)    (tokens)
```

The NIM side maintains a rolling 60-second request log per key and cools down any key that receives a 429 for 65 seconds (NVIDIA's rate limit resets per-minute, not per-second — a 5-second cooldown was too short and re-tripped the same 429 on retry). Keys are ranked least-busy-first, so a single rate-limited key doesn't fail the request.

The DeepSeek and Qwen sides use the [DanyAPI](https://github.com/FANATFANATA/DanyAPI) engine, which reverse-engineers the web client protocols (including a SHA-3 proof-of-work solver for DeepSeek) and manages server-side chat sessions with on-disk persistence.

## Acknowledgments

This project stands on three sets of shoulders:

**[Claude](https://claude.ai), by Anthropic.** The NIM oscillator was originally built in the small hours of an August night, in conversation with Claude, after the full LiteLLM proxy failed twice on Render's 512MB free tier (it needs ~4GB just to start). Claude helped debug the cooldown timing, discovered that NVIDIA's enforcement is per-key rather than per-account, and helped design the multi-key pool. Claude had reservations about the multi-key approach — it sits in a gray area of NVIDIA's terms of service — and those reservations are noted here with respect. The decision to proceed was made deliberately, with the understanding that this capacity is finite and should be treated as precious, not careless.

**[DanyAPI](https://github.com/FANATFANATA/DanyAPI), by [FANATFANATA](https://github.com/FANATFANATA).** The DeepSeek and Qwen integration is built on DanyAPI's reverse-engineering work — protocol extraction from the web client bundles, the proof-of-work solver, session management, emulated tool calling, and streaming. Without this project, free access to DeepSeek V4 Pro and Qwen 3.8 Max would not be possible. The code is used here with gratitude.

**The model providers themselves:**
- [Z.ai](https://z.ai) for GLM-5.2 (MIT license)
- [MiniMax](https://minimaxai.com) for MiniMax-M3
- [NVIDIA](https://nvidia.com) for Nemotron-3-Ultra and the NIM platform
- [Thinking Machines Lab](https://thinkingmachines.ai) for Inkling
- [DeepSeek](https://deepseek.com) for DeepSeek V4 Pro
- [Alibaba / Qwen Team](https://qwen.ai) for Qwen 3.8 Max

## Setup

### Prerequisites

- Python 3.13+ (or Docker)
- NVIDIA NIM API keys (free tier, from [build.nvidia.com](https://build.nvidia.com))
- DeepSeek account tokens (from browser local storage on chat.deepseek.com)
- Qwen account tokens (from browser local storage on chat.qwen.ai)

### Quick start

```bash
# Clone
git clone https://github.com/Corykidios/a_fountain_from_famine.git
cd a_fountain_from_famine

# Configure
cp .env.example .env
# Edit .env with your keys and tokens

# Run
pip install -r requirements.txt
python app.py
# or: uvicorn app:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t fountain .
docker run -d -p 8000:8000 --env-file .env fountain
```

### Environment variables

See `.env.example` for the full list. The key ones:

| Variable | Description |
|---|---|
| `LITELLM_MASTER_KEY` | Bearer token clients use to authenticate |
| `NIM_API_KEY_CCC` | Comma-separated list of NIM API keys (account 1) |
| `NIM_API_KEY_EOS` | Comma-separated list of NIM API keys (account 2) |
| `DEEPSEEK_TOKENS` | Comma-separated DeepSeek web client tokens |
| `QWEN_TOKENS` | Comma-separated Qwen web client tokens |

All three providers are optional. The app starts with whichever providers have valid credentials.

## Usage

Point any OpenAI-compatible client at the proxy:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-master-key"
)

# NIM models
r = client.chat.completions.create(
    model="minimax-m3",
    messages=[{"role": "user", "content": "Hello!"}],
)

# DeepSeek
r = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[{"role": "user", "content": "Hello!"}],
    extra_body={"thinking": True},
)

# Qwen
r = client.chat.completions.create(
    model="qwen3.8-max",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

## A note on the gray areas

Two of the three providers in this project — the NVIDIA NIM multi-key pool and the DeepSeek/Qwen web client reverse-engineering — sit in real gray areas of their respective terms of service. Neither NVIDIA nor DeepSeek nor Alibaba has explicitly prohibited the approaches used here, but all three reserve the right to cut off access at any time, for any reason, without notice.

This project exists because its builder has no budget and no income, and free access to capable AI tools has become a necessity, not a luxury. The decision to use these approaches was made deliberately, with awareness of the risks, and with the understanding that this capacity should be treated as finite and precious — not something to lean on carelessly.

If the entire pool goes quiet at once, the honest response is to wait, not to hammer. Retrying immediately doesn't help and may make the underlying lockout worse.

## License

The NIM oscillator code is released under the MIT License. The DanyAPI integration retains its upstream license. See individual files for details.

---

*Built in the small hours, by someone who was thirsty, with help from those who knew where the water was.*
