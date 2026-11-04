"""Local provider adapter for the PAIEC autonomous loop.

Keys are loaded from the user's local credential files or environment only.
No credential value is logged or returned in errors.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
KEYS = Path(os.environ.get("PAIEC_LLM_KEYS", r"C:\Users\Saifuddin\.config\paiec\llm_keys.json"))
KAGGLE_ENV = Path(os.environ.get("PAIEC_KAGGLE_ENV", str(ROOT / ".secrets" / "proxy_current.env")))


def _dotenv(path: Path) -> dict[str, str]:
    out = {}
    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            if "=" in raw and not raw.lstrip().startswith("#"):
                k, v = raw.split("=", 1)
                out[k.strip()] = v.strip().strip('"')
    return out


def _secret(name: str) -> str:
    value = os.environ.get(name, "")
    if value:
        return value
    if KEYS.exists():
        data = json.loads(KEYS.read_text(encoding="utf-8"))
        value = data.get(name.lower(), "")
        if isinstance(value, str):
            return value
    return ""


def _route(provider: str) -> tuple[str, str, str]:
    p = provider.lower()
    if p == "kilo":
        return ("https://api.kilo.ai/api/gateway/chat/completions", _secret("KILO"),
                os.environ.get("KILO_MODEL", "nvidia/nemotron-3.5-super-vl:free"))
    if p == "nvidia":
        return ("https://integrate.api.nvidia.com/v1/chat/completions", _secret("NVIDIA"),
                os.environ.get("NVIDIA_MODEL", "ibm/granite-34b-code-instruct"))
    if p == "kaggle":
        e = _dotenv(KAGGLE_ENV)
        base = os.environ.get("MODEL_PROXY_URL", e.get("MODEL_PROXY_URL", "")).rstrip("/")
        if base.endswith("/openapi"):
            url = base + "/chat/completions"
        else:
            url = base + "/openapi/chat/completions"
        return (url, os.environ.get("MODEL_PROXY_API_KEY", e.get("MODEL_PROXY_API_KEY", "")),
                os.environ.get("KAGGLE_MODEL", e.get("LLM_DEFAULT", "google/gemini-3-flash-preview")))
    if p == "openrouter":
        return ("https://openrouter.ai/api/v1/chat/completions", _secret("OPENROUTER"),
                os.environ.get("OPENROUTER_MODEL", "qwen/qwen-2.5-coder-32b-instruct:free"))
    if p == "anthropic":
        return ("https://api.anthropic.com/v1/messages", os.environ.get("ANTHROPIC_API_KEY", _secret("ANTHROPIC")),
                os.environ.get("ANTHROPIC_MODEL", "claude-opus-5-5"))
    raise ValueError(f"unsupported provider: {provider}")


_HF = {}


def _hf_local(prompt: str, system: str, max_tokens: int, temperature: float) -> tuple[str, dict]:
    """Local Qwen2.5-Coder-7B-Instruct (4-bit NF4) from the HF cache; no network."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    model_id = os.environ.get("HF_LOCAL_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
    if "model" not in _HF:
        tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
        mdl = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant, device_map="auto", local_files_only=True)
        mdl.eval()
        _HF.update(tok=tok, model=mdl)
    tok, mdl = _HF["tok"], _HF["model"]
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt").to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(**inputs, max_new_tokens=max_tokens, do_sample=temperature > 0, temperature=max(temperature, 1e-3), top_p=0.9)
    reply = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return reply, {"provider": "hf_local", "model": model_id, "attempt": 1, "usage": {"input_tokens": int(inputs.input_ids.shape[1])}}


def complete(prompt: str, *, provider: str | None = None, system: str = "",
             max_tokens: int = 1200, temperature: float = 0.15) -> tuple[str, dict]:
    provider = provider or os.environ.get("LLM_PROVIDER", "kaggle")
    if provider.lower() == "hf_local":
        return _hf_local(prompt, system, max_tokens, temperature)
    url, key, model = _route(provider)
    if not key or not url:
        raise RuntimeError(f"credentials unavailable for provider={provider}")
    messages = ([{"role": "system", "content": system}] if system else [])
    messages.append({"role": "user", "content": prompt})
    body = {"model": model, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens, "stream": False}
    if provider.lower() == "kilo":
        body["reasoning"] = {"enabled": False}
    last = None
    for attempt, delay in enumerate((0, 5, 15, 30), 1):
        if delay:
            time.sleep(delay)
        try:
            request_body = dict(body)
            if attempt == 2:
                request_body["messages"] = messages[:1] + [{"role": "user", "content": prompt[:6000]}]
                request_body["max_tokens"] = min(max_tokens, 700)
            elif attempt == 3:
                request_body["messages"] = messages[:1] + [{"role": "user", "content":
                    "Return one minimal <CHANGE> JSON object. Use an exact existing line.\n" + prompt[:1800]}]
                request_body["max_tokens"] = min(max_tokens, 450)
            headers = {"Content-Type": "application/json"}
            if provider.lower() == "anthropic":
                headers.update({"x-api-key": key, "anthropic-version": os.environ.get("ANTHROPIC_VERSION", "2026-10-21")})
                request_body = {"model": model, "system": system, "messages": [{"role": "user", "content": prompt}], "max_tokens": request_body.get("max_tokens", max_tokens)}
            else:
                headers["Authorization"] = f"Bearer {key}"
            req = Request(url, data=json.dumps(request_body).encode(), headers=headers)
            with urlopen(req, timeout=int(os.environ.get("LLM_TIMEOUT", "180"))) as resp:
                raw = resp.read().decode(errors="replace").strip()
            if not raw:
                raise RuntimeError("provider returned empty body")
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError("provider returned non-JSON body") from exc
            if provider.lower() == "anthropic":
                blocks = data.get("content") or []
                text = "".join(str(b.get("text", "")) for b in blocks if isinstance(b, dict))
            else:
                choice = (data.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                text = msg.get("content") or msg.get("reasoning_content") or ""
            if not text.strip():
                raise RuntimeError("empty model response")
            usage = data.get("usage") or {}
            return text, {"provider": provider, "model": model,
                          "attempt": attempt, "usage": usage}
        except HTTPError as exc:
            last = RuntimeError(f"provider HTTP {exc.code}")
        except Exception as exc:
            last = exc
    raise RuntimeError(f"provider failed after 4 attempts: {last}")
