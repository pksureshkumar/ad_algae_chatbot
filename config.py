import os
from pathlib import Path
from raganything import RAGAnythingConfig

BASE_DIR = Path(__file__).parent
PAPERS_DIR = BASE_DIR / "papers"
RAG_STORAGE_DIR = BASE_DIR / "rag_storage"

# Azure deployment names — must match what you created in Azure AI Studio.
# The values here are sensible defaults; override in .env if your deployment
# names differ.
LLM_MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT_LLM", "gpt-4o-mini")
VISION_MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT_VISION", "gpt-4o")
EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT_EMBEDDING", "text-embedding-3-large")
EMBEDDING_DIM = int(os.getenv("AZURE_OPENAI_EMBEDDING_DIM", "1536"))  # 1536 for text-embedding-3-small, 3072 for text-embedding-3-large

# Query defaults
DEFAULT_TOP_K = 10
DEFAULT_SEARCH_MODE = "hybrid"  # "hybrid", "local", or "global"

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

# Domain system prompt injected into every LLM call so the model
# answers as a specialist in AD/algae integration.
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
