# AD/Algae Chatbot

## Project Overview

This is a research chatbot for **anaerobic digestion (AD), algae cultivation, and algae-AD integration**, built on top of [RAG-Anything](https://github.com/HKUDS/RAG-Anything). RAG-Anything is chosen specifically because it parses tables and figures — not just text — which are critical for extracting data from scientific papers.

**The pipeline runs entirely on local Ollama — no cloud API keys, no per-token billing.** The LLM, the vision model, and the embedding model are all served from `localhost:11434`. The chatbot can be deployed to open access without incurring any cloud cost.

### Corpus: the index holds more papers than `papers/` does

`rag_storage/` was built on a different machine (`D:\pavan_chatbot\...`) from a `papers/` **plus** an `additional_papers/` folder, covering **272 papers**. Only 155 of those PDFs are present on this machine, and 1 PDF here (`1-s2.0-S1385894722038062-main.pdf`) was never indexed.

**Consequence: never run `ingest.py --reset` on this machine.** It would rebuild from the 155 local PDFs and permanently discard 118 papers' worth of indexed knowledge. To rebuild the full corpus you would first need to recover the missing PDFs from the original `D:` drive. Use `reembed.py` instead for anything embedding-related.

## Directory Structure

```
ad_algae_chatbot/
├── papers/               # 155 PDFs locally (the index covers 272 — see above)
├── rag_storage/          # Vector index + knowledge graph (do not edit manually)
├── reembed.py            # Swap the index to a new embedding model (hours, not weeks)
├── output/               # Query results (batch_results_final.md + per-paper MinerU folders)
├── config.py             # All settings: models, paths, RAGAnythingConfig, system prompt
├── models.py             # Async LLM, embedding, and vision model functions (all Ollama)
├── ingest.py             # One-time pipeline: parse all PDFs → rag_storage/
├── chat.py               # Interactive multi-turn chatbot (reads rag_storage/)
├── query.py              # Single-shot CLI query (reads rag_storage/)
├── app.py                # FastAPI web UI (reads rag_storage/)
├── batch_query.py        # Run all 60 permutations of research questions → output/
├── requirements.txt      # Python dependencies
├── .env                  # Local overrides (optional — defaults in config.py work as-is)
└── .env.example          # Template for .env
```

## Setup

```bash
# 1. Install dependencies (Python 3.10+ required)
pip install -r requirements.txt

# 2. Start Ollama and pull the three models
ollama pull qwen3:14b       # LLM — entity extraction + answering
ollama pull qwen2.5vl:32b   # vision — tables/figures during ingestion
ollama pull bge-m3          # embeddings — 1024-dim, 8192-token context

# 3. (Optional) copy the env template to override model choices
cp .env.example .env

# 4. On first run MinerU will download its parsing models (~several GB from HuggingFace)
#    Ensure internet access and enough disk space before running ingest.py
```

No API keys are required. If Ollama listens somewhere other than `localhost:11434`, set `OLLAMA_BASE_URL`.

## Workflow

### Step 1 — Ingest (already done; see the corpus warning above)

```bash
python ingest.py               # index any PDFs in papers/ not already indexed
python ingest.py --test        # first 2 PDFs only (verify pipeline before full run)
python ingest.py --reset       # DESTRUCTIVE here — would drop the corpus to 155 papers
```

`--reset` **renames** `rag_storage/` to `rag_storage_backup_<timestamp>/` rather than deleting it, then builds fresh.

Ingestion requires MinerU, which needs **Python ≤3.13** — use the `ad_algae` conda env, not the system Python 3.14. MinerU's CLI must be on PATH or `MineruParser.check_installation()` returns False:

```bash
conda activate ad_algae && python ingest.py
```

### Step 1b — Changing the embedding model

Changing `OLLAMA_EMBEDDING_MODEL` invalidates every stored vector — different model, different vector space, different dimension. **Use `reembed.py`, not `ingest.py --reset`:**

```bash
python reembed.py --dry-run    # report record counts and a time estimate
python reembed.py              # re-embed all three stores (~4 h for 806k vectors)
python reembed.py --store chunks
```

It regenerates vectors from the `content` already stored in the index, so the knowledge graph and all 272 papers survive. It rewrites `.npy`, `.pkl` **and** `.json` together (`query.py` and `batch_query.py` read the JSON directly, so a stale JSON would be a landmine), moves the originals to `rag_storage/_backup_azure_1536/`, and skips stores already marked `*.reembedded`.

Ingestion progress is saved to `rag_storage/ingested_files.json` after each file — safe to interrupt and resume. Failed files are logged but do not stop the run.

### Step 2 — Query

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

**Question types:** methane yield, VFA yield, acetate yield, bioproduct yield, technoeconomic improvement, economic improvement.

**Process variants:** algal process, algal biochar, photosynthetic biocathode, bio-electrochemical systems, algal biogas upgrade, algal CO2 capture, photobioreactor, HRAP, high rate algal pond, biochar electrode.

Results are saved as clean Markdown (answers + paper references only, no raw chunk excerpts). The completed run is at `output/batch_results_final.md`.

## Key Configuration (`config.py`)

| Setting | Default | Notes |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama's OpenAI-compatible endpoint |
| `OLLAMA_LLM_MODEL` | `qwen3:14b` | Entity extraction and answering |
| `OLLAMA_VISION_MODEL` | `qwen2.5vl:32b` | Table/image captioning during ingestion |
| `OLLAMA_EMBEDDING_MODEL` | `bge-m3` | Changing this requires `ingest.py --reset` |
| `EMBEDDING_DIM` | `1024` | Must match the embedding model exactly |
| `OLLAMA_THINKING` | `1` | qwen3 reasoning. Turning it *off* measured slower, not faster |
| `DEFAULT_SEARCH_MODE` | `local` | `hybrid` / `local` / `global` |
| `DEFAULT_TOP_K` | `10` | Chunks retrieved per query |
| `RAG_CONFIG.max_concurrent_files` | `2` | Lower if running out of memory |
| `DOMAIN_SYSTEM_PROMPT` | (AD/algae expert prompt) | Edit to change chatbot persona |

## Model Functions (`models.py`)

All three call the same local Ollama server through its OpenAI-compatible API.

- **`llm_model_func`** — chat completions via `qwen3:14b`, temperature 0.1. Prepends `DOMAIN_SYSTEM_PROMPT` to every call.
- **`embedding_func`** — `bge-m3` embeddings wrapped in LightRAG's `EmbeddingFunc` (dim=1024). Sends fixed-size sub-batches because Ollama stalls on large batches of long chunks, re-sorts responses by index since Ollama does not guarantee ordering, and hard-fails on a dimension mismatch rather than writing corrupt vectors.
- **`vision_model_func`** — `qwen2.5vl:32b` for ingestion. Handles local file paths, raw base64 strings, data URIs, and HTTP URLs; falls back to text-only if an image cannot be loaded.

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
Already applied in `chat.py`, `query.py`, and `batch_query.py`.

### Embedding model and index are coupled
`rag_storage/` stores raw vectors, not text-to-vector mappings. Swapping `OLLAMA_EMBEDDING_MODEL` without `ingest.py --reset` either crashes on the dimension assert in `nano_vectordb` or — if the dimensions happen to match — silently returns nonsense, because query vectors land in a different space than the indexed ones.

### `_ensure_lightrag_initialized()` before queries
LightRAG is initialized lazily. Always call `await rag._ensure_lightrag_initialized()` before any query to avoid `NoneType` errors on `rag.lightrag`.

## Important Notes

- **`rag_storage/` is generated data** — rebuildable by re-running `ingest.py`. Do not manually edit files inside it. Expect ~20 GB and a multi-day run on local hardware.
- **`papers/` is read-only** — ingest.py never modifies PDFs.
- **MinerU model download** — happens automatically on the first `ingest.py` run. Models cached in `~/.cache/huggingface/`. Requires several GB of disk space.
- **No API costs** — the whole pipeline is local. The tradeoff is wall-clock time: a hosted API ingests in hours, local Ollama in days.
- **Keep Ollama's models resident** — set `OLLAMA_KEEP_ALIVE=-1` before a long ingest so the LLM, vision, and embedding models are not repeatedly evicted and reloaded from disk.
- **`.env` must never be committed** — listed in `.gitignore`.

## License and scope

The code in this repository is released under the MIT License (see `LICENSE`).

The licence covers the **software only**. Three things it does not and cannot
cover:

- **The source papers.** The corpus consists of third-party peer-reviewed
  articles, most behind paywalls. They are not redistributed here, and no
  permission to use them is granted by this licence. `papers_metadata.json` and
  `eval/corpus_paper_list.csv` list the DOIs so the papers can be obtained
  legally from their publishers.
- **Text copied out of those papers.** Retrieved chunk bodies and the pinned
  verbatim passages used as ground truth are excluded from this repository for
  that reason. `rag_benchmarking/runs_public/` and
  `rag_benchmarking/data/ground_truth_public.jsonl` contain the same records
  with the copied text removed — enough to audit every reported score, not
  enough to reconstruct the sources. Anyone holding the corpus can regenerate
  the full files with `build_dataset.py` and `run_arms.py`.
- **The built index.** `rag_storage/` is derived from the papers and is not
  published here.

Upstream components carry their own licences, including RAG-Anything, LightRAG,
MinerU, and the Ollama models (`qwen3`, `qwen2.5vl`, `bge-m3`, `gemma3`,
`phi4`). Check each before redistributing a deployment.
