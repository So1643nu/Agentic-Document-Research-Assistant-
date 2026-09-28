"""Central place for reading settings from environment variables.

Keeping this in one module means nothing else in the codebase reaches
into os.environ directly, which makes it obvious where to look when
something is misconfigured.
"""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _bool_env(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or not val.strip():
        return default
    try:
        return int(val)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str
    model_name: str
    use_embeddings: bool
    chunk_size: int
    chunk_overlap: int
    top_k: int
    max_agent_steps: int


def load_settings() -> Settings:
    return Settings(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        model_name=os.getenv("MODEL_NAME", "claude-sonnet-5"),
        use_embeddings=_bool_env("USE_EMBEDDINGS", False),
        chunk_size=_int_env("CHUNK_SIZE", 800),
        chunk_overlap=_int_env("CHUNK_OVERLAP", 120),
        top_k=_int_env("TOP_K", 4),
        max_agent_steps=_int_env("MAX_AGENT_STEPS", 6),
    )
