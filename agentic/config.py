"""Agent budgets + settings (env-driven, per architecture doc §15.10)."""

import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "5"))
MAX_GRAPH_CALLS = int(os.getenv("AGENT_MAX_GRAPH_CALLS", "3"))
MAX_VECTOR_CALLS = int(os.getenv("AGENT_MAX_VECTOR_CALLS", "2"))
MAX_COMMUNITY_CALLS = int(os.getenv("AGENT_MAX_COMMUNITY_CALLS", "1"))

# Final context size
TOP_N_CHUNKS = int(os.getenv("AGENT_TOP_N_CHUNKS", "8"))
MAX_PATHS = int(os.getenv("AGENT_MAX_PATHS", "20"))

__all__ = [
    "MAX_STEPS",
    "MAX_GRAPH_CALLS",
    "MAX_VECTOR_CALLS",
    "MAX_COMMUNITY_CALLS",
    "TOP_N_CHUNKS",
    "MAX_PATHS",
]
