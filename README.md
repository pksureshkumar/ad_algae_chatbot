# AD/Algae RAG Chatbot

A retrieval-augmented chatbot over 282 peer-reviewed papers on **anaerobic digestion (AD),
algae cultivation, and algae–AD integration**, together with the quantitative benchmark used to
validate it.

Built on [RAG-Anything](https://github.com/HKUDS/RAG-Anything), which parses **tables and figures**
rather than text alone. That matters in this field: roughly three quarters of the numeric values
printed in tables in this corpus never appear in the surrounding prose, so a text-only pipeline
cannot reach them.

**Everything runs on local [Ollama](https://ollama.com).** The LLM, vision model and embedding
model are all served from `localhost:11434`. There are no cloud API keys and no per-token billing,
so the system can be deployed at zero marginal cost.

This repository supplements a manuscript under review. The source papers are not redistributed
here — see [License and scope](#license-and-scope).

---

## Where to find things

| I want to… | Go to |
|---|---|
| **See the benchmark results** | [`rag_benchmarking/results/benchmark_results.xlsx`](rag_benchmarking/results/benchmark_results.xlsx) — every metric, every arm, all questions and answers |
| Read the results as text | [`rag_benchmarking/results/REPORT.md`](rag_benchmarking/results/REPORT.md) |
| Understand how the benchmark works | [`rag_benchmarking/README.md`](rag_benchmarking/README.md) |
| See the questions and ground truth | [`rag_benchmarking/data/ground_truth_public.jsonl`](rag_benchmarking/data/ground_truth_public.jsonl) |
| Run the chatbot | [Running the chatbot](#running-the-chatbot) |
| Rebuild the index from papers | [`ingest.py`](ingest.py), and [Ingestion](#ingestion) |
| See which papers are in the corpus | [`papers_metadata.json`](papers_metadata.json), [`rag_benchmarking/corpus_paper_list.csv`](rag_benchmarking/corpus_paper_list.csv) |
| Change models or settings | [`config.py`](config.py) |

## Benchmark results at a glance

16 questions, each pinned to a specific numeric value in a specific paper. Six arms. Full detail,
caveats and paired tests in the workbook above.

| arm | gold chunk retrieved | gold values recovered (of 53) | faithfulness |
|---|---|---|---|
| **multimodal RAG** (this system) | **14/16** | **38** | **0.979** |
| text-only RAG (identical, tables/figures withheld) | 6/16 | 9 | 0.788 |
| `qwen3:14b`, no retrieval | n/a | 1 | n/a |
| `gemma3:27b`, no retrieval | n/a | 2 | n/a |
| GPT-6 Astra, no retrieval | n/a | 0 | n/a |
| GPT-6 Astra + web search | n/a | 37 | n/a |

Multimodal versus text-only: p = 0.0078 on gold-chunk retrieval, p < 0.0001 on values (McNemar
exact). Against a frontier model with live web access the difference is not detectable (p = 1.00) —
a 14B model running locally over a curated, citable corpus performs comparably.

Two metrics in the workbook are flagged and should not be read at face value: `answer_relevancy`
rewards confident wrong answers and ranks the weakest arm highest, and the `non_llm_context_*`
metrics sit near zero for a mechanical reason (string similarity between a short pinned passage and
a full retrieved chunk). Those reasons are written into the spreadsheet itself.

---

## Repository layout

Python modules stay flat at the repository root on purpose: they import each other by bare name
(`import config`, `from models import ...`) and resolve data paths relative to the working
directory, so moving them into packages breaks both.

```
├── app.py                  FastAPI web UI — the primary interface
├── chat.py / query.py      interactive and single-shot CLI
├── batch_query.py          60 permutations of research questions
│
├── config.py               all settings: models, paths, system prompt
├── models.py               async Ollama callables (LLM, embeddings, vision)
├── _env.py                 env defaults that MUST precede any LightRAG import
├── fast_storage.py         loads vdb_*.npy/.pkl instead of parsing multi-GB JSON
├── citations.py            maps retrieved chunks to papers for inline citations
├── papers_metadata.json    282 records: filename, title, authors, year, DOI
│
├── ingest.py               parse PDFs → rag_storage/   (resumable)
├── reembed.py              swap the index to a new embedding model
├── migrate_storage.py      vdb_*.json → .npy + .pkl for fast startup
├── repair_missing_vectors.py  embed chunks that ingestion left unvectorised
│
├── enrich_metadata.py      backfill authors from CrossRef
├── expand_metadata.py      find DOIs for papers missing from metadata
├── extract_database.py     extract structured numerical data from the index
├── generate_eval_report.py build the corpus-audit PDF
│
├── static/index.html       single-file web UI served by app.py
│
└── rag_benchmarking/       the benchmark — see its own README
    ├── harness/            dataset build, screening, retrieval, scoring, export
    ├── data/               the question set (public variant)
    ├── runs_public/        every arm's retrieval and answers, chunk text removed
    ├── results/            scores, RAGAS output, REPORT.md, the workbook
    ├── pilot/              the five machine-proposed questions and the pilot run
    ├── design/             earlier evaluation design and corpus audit
    └── gpt_*_arm/          prompts and collected answers for the manual arms
```

Not in this repository: `papers/` (the source PDFs), `rag_storage/` (the built index, ~14 GB), and
the manuscript. All are gitignored with the reason stated inline.

---

## Setup

Python 3.10+ (MinerU requires ≤3.13, so stay in that range if you intend to ingest).

```bash
pip install -r requirements.txt

ollama pull qwen3:14b       # LLM — entity extraction and answering
ollama pull qwen2.5vl:32b   # vision — tables and figures, ingestion only
ollama pull bge-m3          # embeddings — 1024-dim, 8192-token context
```

No API keys are required. Set `OLLAMA_BASE_URL` if Ollama is not on `localhost:11434`.

A query-only deployment does **not** need `qwen2.5vl:32b` — the vision model is used during
ingestion only. That drops the VRAM requirement from roughly 32 GB to about 11 GB.

## Running the chatbot

```bash
uvicorn app:app --reload --port 8000     # then open http://localhost:8000
python chat.py                           # interactive CLI
python query.py "What are the benefits of co-digesting algae with AD?"
python query.py "..." --mode local --top-k 15
```

The index holds 808,353 vectors (29,599 chunks, 167,876 entities, 610,878 relationships) and
takes a few minutes to load even with `fast_storage`.
`app.py` does this once, in its lifespan handler.

| mode | retrieves |
|---|---|
| `local` | specific entities and immediate context — **default** |
| `global` | high-level themes and cross-document relationships |
| `hybrid` | both |

## Ingestion

Requires the source PDFs, which are not distributed here. `papers_metadata.json` lists every DOI so
the corpus can be obtained from the publishers.

```bash
python ingest.py                             # index any PDFs in papers/ not already indexed
PAPERS_DIR=/path/to/new python ingest.py     # index a different directory only
```

Progress is saved after each file, so a run is interruptible and resumable. Expect roughly 45
minutes per paper on a single consumer GPU: MinerU parsing, then vision captioning of tables and
figures, then LLM entity extraction.

**Verify the index afterwards.** Ingestion can report success for every file while leaving chunks
unvectorised — present in the knowledge base and unreachable by retrieval. That happened here to
five papers, 1,329 chunks. `python repair_missing_vectors.py --dry-run` reports any such gap, and
the same script repairs it.

## Running the benchmark

```bash
cd rag_benchmarking/harness
python build_dataset.py      # spreadsheet → data/ground_truth.jsonl
python screen.py --write     # screen for text leakage, assign strata
python run_arms.py           # run the local arms
python score.py --run ../runs/multimodal.jsonl
python report.py             # consolidated REPORT.md
python export_xlsx.py        # the workbook
```

RAGAS scoring runs in a **separate environment** — see the header of
`rag_benchmarking/harness/ragas_eval.py`, which documents two install traps in ragas 0.4.3. Do not
install ragas alongside the chatbot's dependencies.

---

## Things worth knowing before changing anything

**Import order is load-bearing.** `fast_storage` (or `_env`) must be imported before anything that
pulls in LightRAG. `app.py` and `chat.py` do this on line 1. Moving it silently breaks UTF-8
output, the tiktoken cache pin and the rerank default.

**`vlm_enhanced=False` is required on every `rag.aquery()` call.** Otherwise RAG-Anything routes
through a vision path that calls the vision model with an empty prompt whenever image paths appear
in retrieved context.

**Call `await rag._ensure_lightrag_initialized()` before any query.** LightRAG is initialised
lazily inside RAG-Anything.

**The embedding model and the index are coupled.** `rag_storage/` holds raw vectors, not
text→vector mappings. Changing `OLLAMA_EMBEDDING_MODEL` without re-embedding either trips the
dimension assertion or, if the dimensions happen to match, silently returns nonsense. Use
`reembed.py`.

**`vdb_*.npy` must be at least as new as `vdb_*.json`**, or `fast_storage` silently falls back to
the slow JSON path. `reembed.py` and `migrate_storage.py` both touch the `.npy` last.

**The LLM response cache is disabled deliberately.** It is a single JSON file loaded whole at
startup that grows without bound; it reached 1.55 GB here and then failed with `MemoryError`, and a
failed load persists an empty store back over the file.

---

## License and scope

The code in this repository is released under the MIT License (see [`LICENSE`](LICENSE)).

The licence covers the **software only**. Three things it does not and cannot cover:

- **The source papers.** The corpus is third-party peer-reviewed work, most of it behind paywalls.
  It is not redistributed here and this licence grants no rights over it. `papers_metadata.json`
  and `rag_benchmarking/corpus_paper_list.csv` list the DOIs so the papers can be obtained legally
  from their publishers.
- **Text copied out of those papers.** Retrieved chunk bodies and the pinned verbatim passages used
  as ground truth are excluded for that reason. `rag_benchmarking/runs_public/` and
  `rag_benchmarking/data/ground_truth_public.jsonl` carry the same records with the copied text
  removed — enough to audit every reported score, not enough to reconstruct the sources. Anyone
  holding the corpus can regenerate the full files with `build_dataset.py` and `run_arms.py`.
- **The built index.** `rag_storage/` is derived from the papers and is not published here.

Upstream components carry their own licences, including RAG-Anything, LightRAG, MinerU, and the
Ollama models (`qwen3`, `qwen2.5vl`, `bge-m3`, `gemma3`, `phi4`). Check each before redistributing
a deployment.
