"""LLM entry point for all pipelines — provider-switchable, one change here.

Per context.md the primary LLM is Gemini; when its quota is exhausted the
same functions transparently serve OpenRouter (OpenAI-compatible). Every
caller keeps using `generate_structured` / `generate_text` unchanged.

Usage:
    from shared.llm_client import generate_structured

    parsed, usage = generate_structured(
        prompt=user_prompt,
        response_schema=ChunkExtraction,   # any Pydantic model
        system="You are a precise extraction engine.",
    )
    # parsed: ChunkExtraction instance (validated) — or None on hard failure
    # usage:  {"input_tokens", "output_tokens", "total_tokens"} for metrics

Env:
    LLM_PROVIDER      — "gemini" (default) | "openrouter"
    --- gemini ---
    GOOGLE_API_KEY / GEMINI_API_KEY / GOOGLE_API_KEY_2 / GEMINI_API_KEYS — key pool
    GEMINI_MODEL      — model id (default: gemini-3.8-flash)
    GEMINI_THINKING_BUDGET — thinking token budget (default: 0 = off)
    --- openrouter ---
    OPENROUTER_API_KEY — API key
    OPENROUTER_MODEL   — model id (default: nvidia/nemotron-3-super-120b-a12b:free)
"""

import json
import os
import re
import threading
import time

import requests
from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types

__all__ = ["DEFAULT_MODEL", "get_client", "generate_structured", "generate_text"]

# --- provider selection (the one switch) -------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()  # gemini | openrouter | groq

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
THINKING_BUDGET = int(os.getenv("GEMINI_THINKING_BUDGET", "0"))

OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free"
)
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "deepseek-ai/deepseek-v4.1-flash")
AZURE_MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-nano")
_AZURE_URL = None  # resolved per-call from AZURE_OPENAI_ENDPOINT
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# Effective model id for logging (agentic driver prints it).
if LLM_PROVIDER == "openrouter":
    DEFAULT_MODEL = OPENROUTER_MODEL
elif LLM_PROVIDER == "groq":
    DEFAULT_MODEL = GROQ_MODEL
elif LLM_PROVIDER == "azure":
    DEFAULT_MODEL = AZURE_MODEL

_client_pool: list[genai.Client] = []
_client_idx: int = 0
_pool_keys_sig: tuple[str, ...] = ()
_pool_lock = threading.Lock()


def _api_keys() -> tuple[str, ...]:
    """Collect available API keys (GEMINI_API_KEYS pool, then singles).

    Re-reads .env so keys added while a long run is in progress are
    discovered at the next rotation (existing process env wins).
    """
    load_dotenv()
    keys: list[str] = []
    raw_pool = os.getenv("GEMINI_API_KEYS", "")
    keys.extend(k.strip() for k in raw_pool.split(",") if k.strip())
    for name in ("GOOGLE_API_KEY", "GOOGLE_API_KEY_2", "GOOGLE_API_KEY_3", "GEMINI_API_KEY"):
        value = os.getenv(name)
        if value and value.strip() not in keys:
            keys.append(value.strip())
    return tuple(keys)


def get_client() -> genai.Client:
    """Return the current Gemini client from the key pool.

    The pool is rebuilt whenever the set of env keys changes, so adding a
    key to .env mid-run is picked up at the next quota-driven rotation.
    """
    global _client_pool, _client_idx, _pool_keys_sig
    with _pool_lock:
        keys = _api_keys()
        if not keys:
            raise RuntimeError(
                "Missing Gemini API key: set GOOGLE_API_KEY (or GEMINI_API_KEYS "
                "with multiple comma-separated keys for quota fallback) in .env."
            )
        if keys != _pool_keys_sig:
            _client_pool = [genai.Client(api_key=k) for k in keys]
            _client_idx = _client_idx % len(_client_pool)
            _pool_keys_sig = keys
        return _client_pool[_client_idx]


def _rotate_client(exc: Exception | None = None) -> None:
    """Switch to the next key in the pool (triggered by quota errors)."""
    global _client_pool, _client_idx, _pool_keys_sig
    with _pool_lock:
        keys = _api_keys()
        if keys != _pool_keys_sig:  # new key(s) appeared — rebuild and use them
            _client_pool = [genai.Client(api_key=k) for k in keys]
            _pool_keys_sig = keys
        if len(_client_pool) > 1:
            _client_idx = (_client_idx + 1) % len(_client_pool)


def _build_config(
    system: str | None,
    temperature: float,
    max_output_tokens: int | None,
    response_schema=None,
) -> types.GenerateContentConfig:
    kwargs: dict = {"temperature": temperature}
    if system:
        kwargs["system_instruction"] = system
    if max_output_tokens:
        kwargs["max_output_tokens"] = max_output_tokens
    if response_schema is not None:
        kwargs["response_mime_type"] = "application/json"
        kwargs["response_schema"] = response_schema
    if THINKING_BUDGET >= 0 and re.match(r"gemini-[23]", DEFAULT_MODEL):
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=THINKING_BUDGET)
    return types.GenerateContentConfig(**kwargs)


def _usage_dict(response) -> dict:
    """Extract token usage from Gemini usage_metadata for shared.metrics."""
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return {}
    return {
        "input_tokens": getattr(meta, "prompt_token_count", 0) or 0,
        "output_tokens": getattr(meta, "candidates_token_count", 0) or 0,
        "total_tokens": getattr(meta, "total_token_count", 0) or 0,
    }


def _retry_delay(exc: Exception, attempt: int, base: float) -> float:
    """Pick backoff delay: honor the server's 'retry in Ns' hint when present."""
    match = re.search(r"retry in ([0-9.]+)s", str(exc), re.IGNORECASE)
    if match:
        return min(float(match.group(1)) + 2.0, 120.0)
    return base * (2**attempt)


# --------------------------------------------------------------------------- #
# OpenRouter / Groq providers (OpenAI-compatible; structured output via json_schema)
# --------------------------------------------------------------------------- #

def _strict_json_schema(schema: dict) -> dict:
    """Make a Pydantic schema strict-mode compliant: `additionalProperties:false`
    on every object and every property listed in `required` — required by Groq
    strict mode, harmless for other OpenAI-compatible providers. Strict mode
    forces the model to emit all fields; Pydantic-side defaults remain."""
    if isinstance(schema, dict):
        if schema.get("type") == "object" or "properties" in schema:
            schema.setdefault("additionalProperties", False)
            props = schema.get("properties")
            if props:
                schema["required"] = list(props.keys())
        for value in schema.values():
            _strict_json_schema(value)
    elif isinstance(schema, list):
        for item in schema:
            _strict_json_schema(item)
    return schema


def _openai_compatible_call(
    url: str,
    api_key: str,
    messages: list[dict],
    response_schema=None,
    model: str | None = None,
    temperature: float = 0.0,
    max_output_tokens: int | None = None,
    max_tokens_field: str = "max_tokens",
    extra_body: dict | None = None,
    max_retries: int = 6,
    retry_base_delay: float = 2.0,
) -> tuple[str, dict]:
    """One OpenAI-compatible chat completion; returns (content, usage). Retries on 429/5xx."""
    body: dict = {
        "model": model,
        "messages": messages,
    }
    if temperature is not None:
        body["temperature"] = temperature
    if response_schema is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": response_schema.__name__,
                "strict": True,
                "schema": _strict_json_schema(response_schema.model_json_schema()),
            },
        }
    if max_output_tokens:
        body[max_tokens_field] = max_output_tokens
    if extra_body:
        body.update(extra_body)

    last_error: Exception | None = None
    for attempt in range(max_retries):
        start = time.perf_counter()
        try:
            resp = requests.post(
                url,
                json=body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=180,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = resp.headers.get("Retry-After")
                raise RuntimeError(
                    f"HTTP {resp.status_code}: {resp.text[:200]}"
                    + (f" (retry after {retry_after}s)" if retry_after else "")
                )
            resp.raise_for_status()
            data = resp.json()
            if "choices" not in data:
                raise RuntimeError(f"unexpected response: {json.dumps(data)[:200]}")
            usage_raw = data.get("usage", {}) or {}
            usage = {
                "input_tokens": usage_raw.get("prompt_tokens", 0) or 0,
                "output_tokens": usage_raw.get("completion_tokens", 0) or 0,
                "total_tokens": usage_raw.get("total_tokens", 0) or 0,
                "_latency_ms": round((time.perf_counter() - start) * 1000, 2),
            }
            content = data["choices"][0]["message"].get("content") or ""
            return content, usage
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(_retry_delay(exc, attempt, retry_base_delay))

    raise RuntimeError(
        f"OpenAI-compatible call failed after {max_retries} attempts: {last_error}"
    ) from last_error


def _is_quota_error(exc: Exception) -> bool:
    """True for rate/quota errors where a DIFFERENT provider might still work."""
    s = str(exc)
    return ("429" in s or "RESOURCE_EXHAUSTED" in s or "Rate limit" in s
            or "rate limit" in s or "quota" in s.lower())


def _gemini_structured(
    prompt: str,
    response_schema,
    system: str | None,
    model: str | None,
    temperature: float,
    max_output_tokens: int | None,
    max_retries: int,
    retry_base_delay: float,
):
    model_id = model or DEFAULT_MODEL
    config = _build_config(system, temperature, max_output_tokens, response_schema)

    last_error: Exception | None = None
    for attempt in range(max_retries):
        start = time.perf_counter()
        client = get_client()
        try:
            response = client.models.generate_content(
                model=model_id, contents=prompt, config=config,
            )
            usage = _usage_dict(response)
            usage["_latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
            parsed = getattr(response, "parsed", None)
            if parsed is None:
                text = (getattr(response, "text", None) or "").strip()
                if text.startswith("```"):
                    text = text.strip("`")
                    if text.startswith("json"):
                        text = text[4:]
                parsed = response_schema.model_validate_json(text)
            return parsed, usage
        except Exception as exc:
            last_error = exc
            if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                _rotate_client(exc)
            if attempt < max_retries - 1:
                time.sleep(_retry_delay(exc, attempt, retry_base_delay))

    raise RuntimeError(
        f"Gemini call failed after {max_retries} attempts: {last_error}"
    ) from last_error


def _openai_structured(
    cand: dict,
    prompt: str,
    response_schema,
    system: str | None,
    model: str | None,
    temperature: float,
    max_output_tokens: int | None,
    max_retries: int,
    retry_base_delay: float,
):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            content, usage = _openai_compatible_call(
                cand["url"], cand["key"], messages, response_schema,
                model or cand["model"], cand.get("temperature", temperature),
                max_output_tokens, "max_completion_tokens",
                extra_body=cand.get("extra_body"),
                max_retries=max_retries, retry_base_delay=retry_base_delay,
            )
            text = content.strip()
            if text.startswith("```"):
                text = text.strip("`")
                if text.startswith("json"):
                    text = text[4:]
            return response_schema.model_validate_json(text.strip()), usage
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(_retry_delay(exc, attempt, retry_base_delay))

    raise RuntimeError(
        f"{cand['provider']}:{cand['model']} structured call failed after "
        f"{max_retries} attempts: {last_error}"
    ) from last_error


_cand_idx: int = 0


def _provider_candidates() -> list[dict]:
    """Every usable provider/model candidate, built from env (re-read each call
    so keys/models added mid-run are picked up). LLM_PROVIDER=auto uses all;
    a specific value restricts to that provider only."""
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "auto").strip().lower() or "auto"

    cands: list[dict] = []
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if azure_key and azure_endpoint and provider in ("auto", "azure"):
        base = azure_endpoint.rstrip("/")
        url = base + "/chat/completions" if base.endswith("/openai/v1") else base + "/openai/v1/chat/completions"
        cands.append({"provider": "azure", "url": url,
                      "key": azure_key, "model": AZURE_MODEL,
                      "temperature": None,  # gpt-5 family rejects custom temperature
                      "extra_body": {"reasoning_effort": "low"}})
    gemini_keys = _api_keys()
    if gemini_keys and provider in ("auto", "gemini"):
        cands.append({"provider": "gemini"})
    if os.getenv("GROQ_API_KEY") and provider in ("auto", "groq"):
        for gm in [m.strip() for m in os.getenv("GROQ_MODEL", GROQ_MODEL).split(",") if m.strip()]:
            cands.append({"provider": "groq", "url": _GROQ_URL,
                          "key": os.getenv("GROQ_API_KEY"), "model": gm})
    if os.getenv("OPENROUTER_API_KEY") and provider in ("auto", "openrouter"):
        cands.append({"provider": "openrouter", "url": _OPENROUTER_URL,
                      "key": os.getenv("OPENROUTER_API_KEY"), "model": OPENROUTER_MODEL})
    if os.getenv("NVIDIA_API_KEY") and provider in ("auto", "nvidia"):
        cands.append({"provider": "nvidia", "url": _NVIDIA_URL,
                      "key": os.getenv("NVIDIA_API_KEY"), "model": NVIDIA_MODEL,
                      "extra_body": {"chat_template_kwargs": {"thinking": False}}})
    return cands


def _advance_candidate() -> None:
    global _cand_idx
    _cand_idx += 1


def generate_structured(
    prompt: str,
    response_schema,
    system: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    max_output_tokens: int | None = None,
    max_retries: int = 6,
    retry_base_delay: float = 2.0,
):
    """Call the LLM with a Pydantic `response_schema`; return (parsed, usage).

    Candidates come from _provider_candidates() (LLM_PROVIDER=auto uses every
    provider with keys: Gemini key pool, Groq models, OpenRouter). On a quota/
    rate error the call rotates to the next candidate; non-quota errors retry
    the same candidate. One entry point for all callers.
    """
    cands = _provider_candidates()
    if not cands:
        raise RuntimeError("No LLM provider configured: set GOOGLE_API_KEY, "
                           "GROQ_API_KEY, or OPENROUTER_API_KEY.")
    last_error: Exception | None = None
    for offset in range(len(cands)):
        cand = cands[(_cand_idx + offset) % len(cands)]
        try:
            if cand["provider"] == "gemini":
                return _gemini_structured(
                    prompt, response_schema, system, model, temperature,
                    max_output_tokens, max_retries, retry_base_delay,
                )
            return _openai_structured(
                cand, prompt, response_schema, system, model, temperature,
                max_output_tokens, max_retries, retry_base_delay,
            )
        except Exception as exc:
            last_error = exc
            if _is_quota_error(exc):
                print(f"[llm_client] quota on {cand['provider']}"
                      f"{'/' + cand.get('model', '') if cand.get('model') else ''}; rotating")
                _advance_candidate()
                continue
            raise
    raise RuntimeError(f"All LLM candidates failed: {last_error}")


def generate_text(
    prompt: str,
    system: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    max_output_tokens: int | None = None,
) -> tuple[str, dict]:
    """Plain text completion; returns (text, usage) — used for answer generation."""
    cands = _provider_candidates()
    if not cands:
        raise RuntimeError("No LLM provider configured: set GOOGLE_API_KEY, "
                           "GROQ_API_KEY, or OPENROUTER_API_KEY.")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last_error: Exception | None = None
    for offset in range(len(cands)):
        cand = cands[(_cand_idx + offset) % len(cands)]
        try:
            if cand["provider"] == "gemini":
                client = get_client()
                config = _build_config(system, temperature, max_output_tokens, None)
                response = client.models.generate_content(
                    model=model or DEFAULT_MODEL, contents=prompt, config=config
                )
                return response.text or "", _usage_dict(response)
            content, usage = _openai_compatible_call(
                cand["url"], cand["key"], messages, None, model or cand["model"],
                temperature, max_output_tokens, "max_completion_tokens",
            )
            return content, usage
        except Exception as exc:
            last_error = exc
            if _is_quota_error(exc):
                _advance_candidate()
                continue
            raise
    raise RuntimeError(f"All LLM candidates failed: {last_error}")


if __name__ == "__main__":
    # Smoke test: minimal structured call.
    from pydantic import BaseModel

    class Tiny(BaseModel):
        answer: str

    parsed, usage = generate_structured(
        "Reply with the single word: pong", response_schema=Tiny
    )
    print(f"parsed={parsed!r} usage={json.dumps(usage)}")
