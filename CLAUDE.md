# CLAUDE.md

## Project Overview

This is a research chatbot for **anaerobic digestion (AD), algae cultivation, and algae-AD integration**, built on top of [RAG-Anything](https://github.com/HKUDS/RAG-Anything). The knowledge base is ~141 peer-reviewed PDFs stored in `papers/`. RAG-Anything is chosen specifically because it parses tables and figures — not just text — which are critical for extracting data from scientific papers.

## Directory Structure

```
rag_anything/
├── papers/           # 141 PDFs (source knowledge base — do not modify)
├── rag_storage/      # Generated at runtime by RAG-Anything (index, graph, cache)
├── config.py         # All tunable settings: models, paths, RAGAnythingConfig, system prompt
├── models.py         # Async LLM, embedding, and vision model functions
├── ingest.py         # One-time pipeline: parse all PDFs and build the RAG index
├── chat.py           # Interactive multi-turn chatbot (reads from rag_storage/)
├── query.py          # Single-shot CLI query (reads from rag_storage/)
├── requirements.txt  # Python dependencies
├── .env              # Secret API keys (never commit)
└── .env.example      # Template for .env
```

## Setup

```bash
# 1. Install dependencies (Python 3.10+ required)
pip install -r requirements.txt

# 2. Copy env template and add your OpenAI key
cp .env.example .env
# Edit .env: set OPENAI_API_KEY=sk-...

# 3. On first run MinerU will download its parsing models (~several GB from HuggingFace)
#    Ensure internet access and enough disk space before running ingest.py
```

## Workflow

### Step 1 — Ingest (run once)
```bash
# Process all 141 PDFs and build the knowledge graph
python ingest.py

# Test with just the first 2 PDFs before committing to the full run
python ingest.py --test

# Reset progress and re-ingest everything from scratch
python ingest.py --reset
```

Ingestion progress is saved to `rag_storage/ingested_files.json` after each file, so it is safe to interrupt and resume. Failed files are logged but do not stop the run.

### Step 2 — Query
```bash
# Interactive chatbot
python chat.py

# Single question from the CLI
python query.py "What are the main benefits of co-digesting algae with organic waste?"

# Change retrieval mode (hybrid is default)
python query.py "..." --mode local
python query.py "..." --mode global
```

## Key Configuration (`config.py`)

All important knobs live in `config.py`. Change things there rather than editing individual scripts.

| Setting | Default | Notes |
|---|---|---|
| `LLM_MODEL` | `gpt-4o-mini` | Override via `LLM_MODEL` env var |
| `VISION_MODEL` | `gpt-4o` | Used for table/image captioning |
| `EMBEDDING_MODEL` | `text-embedding-3-large` | `EMBEDDING_DIM` must match |
| `DEFAULT_SEARCH_MODE` | `hybrid` | `hybrid` / `local` / `global` |
| `DEFAULT_TOP_K` | `10` | Chunks retrieved per query |
| `RAG_CONFIG.max_concurrent_files` | `2` | Lower if running out of memory |
| `DOMAIN_SYSTEM_PROMPT` | (AD/algae expert prompt) | Edit to change chatbot persona |

## Model Functions (`models.py`)

Three async callables are defined and wired into RAGAnything:

- **`llm_model_func`** — text generation (gpt-4o-mini). Injects `DOMAIN_SYSTEM_PROMPT` into every call so all responses stay on-topic.
- **`embedding_func`** — wrapped in LightRAG's `EmbeddingFunc` with `embedding_dim=3072`.
- **`vision_model_func`** — called by RAGAnything for every extracted figure/table. Accepts local file paths, base64 data URIs, or URLs.

## Search Modes

| Mode | What it retrieves |
|---|---|
| `hybrid` | Combines local entity-level and global graph-level retrieval (recommended) |
| `local` | Focused on specific entities and their immediate context |
| `global` | High-level themes and cross-document relationships |

## Important Notes

- **`rag_storage/` is generated data** — it can be deleted and rebuilt by re-running `ingest.py`. Do not manually edit files inside it.
- **`papers/` is read-only** — ingest.py never modifies PDFs.
- **MinerU model download** — happens automatically on the first `ingest.py` run. Models are cached in `~/.cache/huggingface/`. This can take time and requires several GB of disk space.
- **Ingestion time** — processing 141 PDFs with MinerU (which runs OCR and layout analysis) can take several hours. Use `--test` first to verify the pipeline works.
- **API costs** — ingestion calls the OpenAI vision API for every extracted figure and table. Monitor usage on large runs.
- **`.env` must never be committed** — it is listed in `.gitignore`.
