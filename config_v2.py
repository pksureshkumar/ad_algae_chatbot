"""
config_v2.py — Configuration for the Ollama-based ingestion pipeline.
Uses local models via Ollama instead of Azure OpenAI.
Output goes to rag_storage_v2/ so it can run alongside the Azure pipeline.
"""

import os
from pathlib import Path
from raganything import RAGAnythingConfig

BASE_DIR = Path(__file__).parent
PAPERS_DIR = BASE_DIR / "papers"
RAG_STORAGE_DIR_V2 = BASE_DIR / "rag_storage_v2"

# Ollama model names (must match `ollama list`)
LLM_MODEL_V2 = "qwen2.5:7b"
VISION_MODEL_V2 = "qwen2.5:7b"        # no vision model installed; falls back to text-only
EMBEDDING_MODEL_V2 = "nomic-embed-text"
EMBEDDING_DIM_V2 = 768                 # nomic-embed-text outputs 768-dim vectors

DEFAULT_TOP_K = 10
DEFAULT_SEARCH_MODE = "hybrid"

RAG_CONFIG_V2 = RAGAnythingConfig(
    working_dir=str(RAG_STORAGE_DIR_V2),
    parser="mineru",
    parse_method="auto",
    enable_image_processing=True,
    enable_table_processing=True,
    enable_equation_processing=True,
    max_concurrent_files=4,            # higher than Azure version — no rate limits
)

DOMAIN_SYSTEM_PROMPT = (
    "You are an expert scientific assistant specialising in anaerobic digestion (AD), "
    "algae cultivation, and the integration of algae with anaerobic digestion systems. "
    "You have access to a comprehensive knowledge base of ~140 peer-reviewed papers on "
    "these topics.\n\n"
    "When answering:\n"
    "- Ground your response in the retrieved literature; cite specific findings, "
    "data, or mechanisms where available.\n"
    "- When discussing algae-AD integration, address benefits such as nutrient recycling, "
    "biogas/biomethane yield improvement, CO2 utilisation, and digestate valorisation.\n"
    "- Use precise scientific terminology appropriate for a research context.\n"
    "- Acknowledge uncertainty or literature gaps where relevant."
)
