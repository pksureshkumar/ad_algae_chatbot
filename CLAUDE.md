# CLAUDE.md

## Project Overview

This is a research chatbot for **anaerobic digestion (AD), algae cultivation, and algae-AD integration**, built on top of [RAG-Anything](https://github.com/HKUDS/RAG-Anything). The knowledge base is 141 peer-reviewed PDFs stored in `papers/`. RAG-Anything is chosen specifically because it parses tables and figures — not just text — which are critical for extracting data from scientific papers.

## Directory Structure

```
ad_algae_chatbot/
├── papers/               # 141 PDFs (source knowledge base — do not modify)
├── rag_storage/          # Azure pipeline index (7.1 GB — do not edit manually)
├── rag_storage_v2/       # Ollama pipeline index (separate, can run in parallel)
├── output/               # Query results saved here (per-paper folders + batch .md files)
├── config.py             # Azure pipeline settings: models, paths, RAGAnythingConfig, system prompt
├── config_v2.py          # Ollama pipeline settings (qwen2.5:7b + nomic-embed-text)
├── models.py             # Azure async LLM, embedding, and vision model functions
├── models_v2.py          # Ollama async LLM, embedding, and vision model functions
├── ingest.py             # One-time pipeline: parse all PDFs → rag_storage/ (Azure)
├── ingest_v2.py          # One-time pipeline: parse all PDFs → rag_storage_v2/ (Ollama)
├── chat.py               # Interactive multi-turn chatbot (Azure, reads rag_storage/)
├── query.py              # Single-shot CLI query (Azure, reads rag_storage/)
├── batch_query.py        # Run all permutations of research questions, save to output/
├── requirements.txt      # Python dependencies
├── .env                  # Secret Azure credentials (never commit)
└── .env.example          # Template for .env
```

## Setup

```bash
# 1. Install dependencies (Python 3.10+ required)
pip install -r requirements.txt

# 2. Copy env template and add your Azure credentials
cp .env.example .env
# Edit .env: set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION

# 3. On first run MinerU will download its parsing models (~several GB from HuggingFace)
#    Ensure internet access and enough disk space before running ingest.py

# 4. For the Ollama pipeline (v2), install Ollama and pull models:
#    ollama pull qwen2.5:7b
#    ollama pull nomic-embed-text
```

## Running Python Scripts

**Always use the full conda env path** — `conda` is not on the shell PATH in this environment:

```bash
/c/Users/SunYufei/anaconda3/envs/rag_anything/python.exe <script.py>
```

## Workflow

### Step 1 — Ingest (run once, already complete)

Both pipelines have been fully ingested:
- **Azure** (`rag_storage/`): 141/141 papers indexed
- **Ollama v2** (`rag_storage_v2/`): test only (2/2); full 141-paper run not yet started

```bash
# Azure pipeline (uses rag_storage/)
python ingest.py
python ingest.py --test    # first 2 PDFs only
python ingest.py --reset   # clear progress and start over

# Ollama pipeline (uses rag_storage_v2/)
python ingest_v2.py
python ingest_v2.py --test
python ingest_v2.py --reset
```

Ingestion progress is saved to `rag_storage/ingested_files.json` (and `rag_storage_v2/ingested_files.json`) after each file, so it is safe to interrupt and resume.

**Important for v2:** Copy `rag_storage/kv_store_parse_cache.json` to `rag_storage_v2/` before running `ingest_v2.py` to skip MinerU re-parsing (reuses existing parse results).

### Step 2 — Query (Azure pipeline)

```bash
# Interactive chatbot
python chat.py

# Single question from the CLI
python query.py "What are the main benefits of co-digesting algae with organic waste?"

# Change retrieval mode (hybrid is default)
python query.py "..." --mode local
python query.py "..." --mode global
python query.py "..." --top-k 15
```

### Step 3 — Batch queries

```bash
# Run all 60 permutations (6 question types x 10 process variants)
# Results saved to output/batch_results_<timestamp>.md
python batch_query.py
```

Question types: methane yield, VFA yield, acetate yield, bioproduct yield, technoeconomic improvement, economic improvement.

Process variants: algal process, algal biochar, photosynthetic biocathode, bio-electrochemical systems, algal biogas upgrade, algal CO2 capture, photobioreactor, HRAP, high rate algal pond, biochar electrode.

## Key Configuration

### `config.py` (Azure pipeline)

| Setting | Default | Notes |
|---|---|---|
| `LLM_MODEL` | `gpt-4.1` | Azure deployment name |
| `VISION_MODEL` | `gpt-4.1` | Used for table/image captioning |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | `EMBEDDING_DIM` must match (1536) |
| `DEFAULT_SEARCH_MODE` | `hybrid` | `hybrid` / `local` / `global` |
| `DEFAULT_TOP_K` | `10` | Chunks retrieved per query |
| `RAG_CONFIG.max_concurrent_files` | `2` | Lower if running out of memory |
| `DOMAIN_SYSTEM_PROMPT` | (AD/algae expert prompt) | Edit to change chatbot persona |

### `config_v2.py` (Ollama pipeline)

| Setting | Default | Notes |
|---|---|---|
| `LLM_MODEL_V2` | `qwen2.5:7b` | Local Ollama model |
| `EMBEDDING_MODEL_V2` | `nomic-embed-text` | 768-dim vectors |
| `EMBEDDING_DIM_V2` | `768` | Must match nomic-embed-text output |
| `RAG_CONFIG_V2.max_concurrent_files` | `4` | No rate limits with local models |

## Model Functions

### `models.py` (Azure)
- **`llm_model_func`** — Azure OpenAI chat completions. Uses `load_dotenv(override=True)` to ensure `.env` overrides any system env vars.
- **`embedding_func`** — Azure embeddings, wrapped in LightRAG's `EmbeddingFunc` (dim=1536).
- **`vision_model_func`** — Azure GPT-4 vision. Handles local file paths, raw base64 strings, data URIs, and HTTP URLs.

### `models_v2.py` (Ollama)
- **`llm_model_func_v2`** — Ollama OpenAI-compatible API at `http://localhost:11434/v1`. Uses `asyncio.Semaphore(2)` to limit concurrent calls (prevents GPU timeout with qwen2.5:7b).
- **`embedding_func_v2`** — nomic-embed-text. Truncates input to 6000 chars to stay within the ~2048 token context limit.
- **`vision_model_func_v2`** — Falls back to text-only (no vision model installed).

## Search Modes

| Mode | What it retrieves |
|---|---|
| `hybrid` | Combines local entity-level and global graph-level retrieval (recommended) |
| `local` | Focused on specific entities and their immediate context |
| `global` | High-level themes and cross-document relationships |

## Known Issues and Fixes

### `vlm_enhanced=False` required on all `rag.aquery()` calls
RAGAnything's `aquery()` detects that a `vision_model_func` is provided and automatically routes through `aquery_vlm_enhanced`. That method calls `vision_model_func("", messages=messages)` — an empty prompt — when image paths appear in the retrieved context. Our `vision_model_func` ignores the `messages` kwarg and passes `""` to the LLM, producing "It appears your message is empty" responses.

**Fix:** Always pass `vlm_enhanced=False` to `rag.aquery()`:
```python
answer = await rag.aquery(query=query, mode=mode, vlm_enhanced=False)
```
This is already applied in `chat.py`, `query.py`, and `batch_query.py`.

### `load_dotenv(override=True)` in `models.py`
The system environment had `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT` swapped at the OS level. `override=True` ensures `.env` values always win over system env vars.

### `_ensure_lightrag_initialized()` before queries
LightRAG is initialized lazily. Always call `await rag._ensure_lightrag_initialized()` before any query to avoid `NoneType` errors on `rag.lightrag`.

### Ollama batch embedding 400 error
nomic-embed-text has a ~2048 token context limit. Large chunks cause `"input length exceeds context length"`. Fixed in `models_v2.py` by truncating all texts to 6000 characters before embedding.

### Ollama LLM worker timeout (360s)
LightRAG's default of 4 concurrent LLM workers overwhelms a single qwen2.5:7b GPU instance. Fixed with `asyncio.Semaphore(2)` in `models_v2.py`.

## Important Notes

- **`rag_storage/` is generated data** — 7.1 GB, can be rebuilt by re-running `ingest.py` (~1 day). Do not manually edit files inside it.
- **`papers/` is read-only** — ingest scripts never modify PDFs.
- **MinerU model download** — happens automatically on the first ingest run. Models cached in `~/.cache/huggingface/`. Requires several GB of disk space.
- **Azure API costs** — ingestion calls the Azure vision API for every extracted figure and table. Monitor usage on large runs.
- **`.env` must never be committed** — listed in `.gitignore`.
- **Transferring to another computer** — copy `rag_storage/` (7.1 GB), all `.py` files, `requirements.txt`, and `.env`. The `papers/` folder (468 MB) is only needed to re-ingest.
