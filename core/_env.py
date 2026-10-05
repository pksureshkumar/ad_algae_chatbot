"""
_env.py — Environment defaults that must be set before LightRAG is imported.

Both config.py and fast_storage.py import this first. It has to live in its own
module because LightRAG reads these at *import* time: `QueryParam.enable_rerank`
is a dataclass field whose default is evaluated when `lightrag.base` loads, and
fast_storage imports raganything (and therefore LightRAG) before config.py gets
a chance to run. Setting them inside config.py alone is silently too late.

Import this before anything that pulls in lightrag/raganything.
"""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]   # repository root, not core/

# Windows consoles default to cp1252, which cannot encode the check marks, arrows
# and em-dashes used in this project's status output. That is not cosmetic: the
# UnicodeEncodeError propagates out of print(), and in app.py it is raised inside
# the FastAPI lifespan — so `uvicorn app:app` dies at startup, after already
# spending a minute loading the index. Force UTF-8 and never fail on output.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # already wrapped, or not a text stream
        pass

# LightRAG's tokenizer calls tiktoken, which downloads its BPE vocabulary from a
# public CDN (openaipublic.blob.core.windows.net) on first use and caches it
# under %TEMP% by default. That is an anonymous file download, not a billable
# API call, but %TEMP% gets cleared and a deployed instance should not need
# outbound internet at startup — so pin the cache in-tree.
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(BASE_DIR / ".tiktoken_cache"))

# LightRAG enables reranking by default, then warns on every query that no
# reranker is configured. We deliberately run without one: the rerank bindings
# it would reach for (Cohere, Jina) are paid cloud APIs, and this deployment
# must stay free and fully local.
os.environ.setdefault("RERANK_BY_DEFAULT", "false")
