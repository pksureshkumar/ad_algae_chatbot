import _env  # noqa: F401  — must precede any lightrag/raganything import

import os
from pathlib import Path

from raganything import RAGAnythingConfig

BASE_DIR = Path(__file__).parent
PAPERS_DIR = Path(os.getenv("PAPERS_DIR", BASE_DIR / "papers"))
# Overridable so a trial ingest can be pointed at a scratch directory without
# risking the real index.
RAG_STORAGE_DIR = Path(os.getenv("RAG_STORAGE_DIR", BASE_DIR / "rag_storage"))

# --- Ollama (LLM + vision + embeddings — everything runs locally, no API cost) ---
# Ollama must be running with the models pulled:
#   ollama pull qwen3:14b
#   ollama pull qwen2.5vl:32b
#   ollama pull bge-m3
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen3:14b")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "qwen2.5vl:32b")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3")

# qwen3 is a reasoning model. Disabling its thinking looks like an easy speedup
# but measured *slower* on real extraction prompts (mean 23.1s off vs 18.7s on,
# 3 corpus chunks): without reasoning the model pads the visible answer with
# ~60% more tokens. Left on, which also keeps output closer to the strict
# delimited format LightRAG's extraction parser expects.
# Set OLLAMA_THINKING=0 to disable if a future model behaves differently.
OLLAMA_THINKING = os.getenv("OLLAMA_THINKING", "1").lower() in ("1", "true", "yes")

# EMBEDDING_DIM must match the model: bge-m3 = 1024, nomic-embed-text = 768,
# qwen3-embedding:0.6b = 1024. Changing the model requires a full re-ingest —
# the vector store asserts on dimension mismatch at load time.
EMBEDDING_DIM = int(os.getenv("OLLAMA_EMBEDDING_DIM", "1024"))
EMBEDDING_MAX_TOKENS = int(os.getenv("OLLAMA_EMBEDDING_MAX_TOKENS", "8192"))

# Measured on this box (bge-m3, RTX 5090): batch=8 -> 11 vec/s, batch=16 -> 33,
# batch=32 -> 57, above which it plateaus. Small batches are badly latency-bound.
EMBEDDING_BATCH_SIZE = int(os.getenv("OLLAMA_EMBEDDING_BATCH_SIZE", "32"))

# Query defaults
DEFAULT_TOP_K = 10
DEFAULT_SEARCH_MODE = "local"  # "hybrid", "local", or "global"

# RAG-Anything configuration
RAG_CONFIG = RAGAnythingConfig(
    working_dir=str(RAG_STORAGE_DIR),
    parser="mineru",
    parse_method="auto",
    enable_image_processing=True,
    enable_table_processing=True,
    enable_equation_processing=True,
    max_concurrent_files=2,
)

# Passed as RAGAnything(lightrag_kwargs=LIGHTRAG_KWARGS) — it is a field on
# RAGAnything itself, not on RAGAnythingConfig.
#
# The LLM response cache is a single JSON file that LightRAG loads whole at
# startup and grows without bound. It reached 1.55 GB here and then failed to
# load with MemoryError on a 32 GB box. That failure mode is nastier than it
# looks: a failed load leaves an empty in-memory store, which shutdown then
# persists straight over the file. For a public deployment, stable memory beats
# instant repeat answers.
LIGHTRAG_KWARGS = {
    "enable_llm_cache": False,
    "enable_llm_cache_for_entity_extract": False,
}

# Domain system prompt injected into every LLM call so the model
# answers as a specialist in AD/algae integration.
DOMAIN_SYSTEM_PROMPT = (
    "You are an expert scientific assistant specialising in anaerobic digestion (AD), "
    "algae cultivation, and the integration of algae with anaerobic digestion systems. "
    "You have access to a comprehensive knowledge base of 272 peer-reviewed papers on "
    "these topics.\n\n"
    "When answering:\n"
    "- Ground your response in the retrieved literature; cite specific findings, "
    "data, or mechanisms where available.\n"
    "- When discussing algae-AD integration, address benefits such as nutrient recycling, "
    "biogas/biomethane yield improvement, CO2 utilisation, and digestate valorisation.\n"
    "- Use precise scientific terminology appropriate for a research context.\n"
    "- Acknowledge uncertainty or literature gaps where relevant."
)
