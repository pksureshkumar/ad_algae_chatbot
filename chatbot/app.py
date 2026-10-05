"""
app.py — FastAPI web server for the AD/Algae research chatbot.

Usage:
    uvicorn app:app --reload --port 8000

Then open http://localhost:8000 in your browser.
"""

import _bootstrap  # noqa: F401  puts the repository root and core/ on sys.path
from _bootstrap import STATIC_DIR
import fast_storage  # must be first — patches NanoVectorDB before LightRAG loads  # noqa: F401

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from lightrag import QueryParam
from pydantic import BaseModel
from raganything import RAGAnything

from citations import build_citation_context, extract_unique_filenames
from config import (DEFAULT_SEARCH_MODE, DEFAULT_TOP_K, LIGHTRAG_KWARGS,
                    RAG_CONFIG, RAG_STORAGE_DIR)
from models import embedding_func, llm_model_func, vision_model_func

_rag: RAGAnything | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _rag
    logging.basicConfig(level=logging.WARNING)

    if not RAG_STORAGE_DIR.exists() or not any(RAG_STORAGE_DIR.iterdir()):
        raise RuntimeError(
            "rag_storage/ is empty or missing. "
            "Run `python ingest.py` first to build the knowledge base."
        )

    _rag = RAGAnything(
        config=RAG_CONFIG,
        lightrag_kwargs=LIGHTRAG_KWARGS,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        vision_model_func=vision_model_func,
    )
    print("Loading knowledge base into memory (this may take a few minutes)...")
    await _rag._ensure_lightrag_initialized()
    print("✓ Ready — open http://localhost:8000 in your browser.\n")
    yield
    await _rag.finalize_storages()


app = FastAPI(title="AD & Algae Research Chatbot", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str
    mode: str = DEFAULT_SEARCH_MODE
    top_k: int = DEFAULT_TOP_K


class ReferenceItem(BaseModel):
    ref_num: int
    filename: str
    title: str
    authors: str
    year: str
    doi: str
    excerpts: list[str]


class ChatResponse(BaseModel):
    answer: str
    references: list[ReferenceItem]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    mode = req.mode if req.mode in ("hybrid", "local", "global") else DEFAULT_SEARCH_MODE
    top_k = max(1, min(req.top_k, 50))

    await _rag._ensure_lightrag_initialized()

    # Retrieve sources first so we can inject citation context into the query.
    param = QueryParam(mode=mode, top_k=top_k)
    sources_data = await _rag.lightrag.aquery_data(req.query, param=param)

    chunks = (sources_data or {}).get("data", {}).get("chunks", [])
    filenames = extract_unique_filenames(chunks)
    instruction, refs = build_citation_context(filenames)

    augmented_query = f"{req.query}\n\n{instruction}" if instruction else req.query
    answer = await _rag.aquery(query=augmented_query, mode=mode, vlm_enhanced=False)

    # Build per-file excerpts for the reference list
    excerpt_map: dict[str, list[str]] = {r["filename"]: [] for r in refs}
    for chunk in chunks:
        raw = chunk.get("file_path") or ""
        fname = raw.replace("\\", "/").split("/")[-1]
        if fname in excerpt_map:
            content = chunk.get("content", "").strip()
            excerpt = " ".join(content.split())[:300]
            if len(content) > 300:
                excerpt += "…"
            excerpt_map[fname].append(excerpt)

    references = [
        ReferenceItem(
            ref_num=r["ref_num"],
            filename=r["filename"],
            title=r["title"],
            authors=r["authors"],
            year=r["year"],
            doi=r["doi"],
            excerpts=excerpt_map.get(r["filename"], []),
        )
        for r in refs
    ]

    return ChatResponse(answer=answer, references=references)
