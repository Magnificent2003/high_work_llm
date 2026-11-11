"""Environment configuration. Import the optional model SDK only for real runs."""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm import VLMSettings

ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = ROOT / "prompts"

def load_env_file(path: Path) -> None:
    """Load a simple KEY=VALUE env file without printing secrets."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def env_optional(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return default if value is None or not value.strip() else value.strip()


def env_bool(name: str, default: bool = False) -> bool:
    value = env_optional(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def env_optional_int(name: str, default: int | None) -> int | None:
    value = env_optional(name)
    if value is None:
        return default
    if value.lower() in {"none", "null"}:
        return None
    return int(value)


def env_optional_float(name: str, default: float | None) -> float | None:
    value = env_optional(name)
    if value is None:
        return default
    if value.lower() in {"none", "null"}:
        return None
    return float(value)


def settings_from_env(temperature_override: float | None = None, max_tokens_override: int | None = None) -> VLMSettings:
    from llm import VLMSettings

    provider = env_optional("VLM_PROVIDER", env_optional("LLM_PROVIDER", "openai")).lower()
    if provider == "openai":
        api_key = env_optional("OPENAI_API_KEY")
        base_url = env_optional("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = env_optional("OPENAI_VLM_MODEL", "gpt-4o-mini")
    elif provider == "dashscope":
        api_key = env_optional("DASHSCOPE_API_KEY")
        base_url = env_optional("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        model = env_optional("DASHSCOPE_VLM_MODEL", "qwen-vl-plus")
    else:
        raise ValueError(f"Unsupported VLM_PROVIDER: {provider}")
    if not api_key:
        raise RuntimeError(f"No API key found for provider={provider!r}.")
    return VLMSettings(
        provider=provider,
        api_key=api_key,
        base_url=base_url or "",
        model=model or "",
        timeout_s=float(env_optional("VLM_TIMEOUT_S", env_optional("LLM_TIMEOUT_S", "120"))),
        temperature=(temperature_override if temperature_override is not None else env_optional_float("VLM_TEMPERATURE", 0.0)),
        max_tokens=(max_tokens_override if max_tokens_override is not None else env_optional_int("VLM_MAX_TOKENS", 1024)),
        trust_env=env_bool("VLM_TRUST_ENV", False),
        request_stream=env_bool("VLM_REQUEST_STREAM", False),
    )
