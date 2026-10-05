"""
fast_storage.py — Monkey-patches NanoVectorDB to load from fast binary files.

Import this module before any LightRAG/RAGAnything imports.
If the fast files (.npy + .pkl) don't exist or are stale, it transparently
falls back to the original slow JSON loading.
"""

import _env  # noqa: F401  — must precede the raganything import below

import pickle
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import nano_vectordb.dbs as _nvdb
import numpy as np

# mineru (PDF parser) requires Python <3.14 and is only needed for ingest.py.
# Stub it out so the query-only app loads on any Python version.
for _mod in (
    "lightrag.mineru_parser",
    "lightrag.modalprocessors",
    "mineru",
    "mineru.api",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# MineruParser.check_installation() runs a subprocess to verify mineru is
# installed — it returns False on Python 3.14, blocking all queries.
# Since the index is already built, ingestion never runs here; safe to stub.
try:
    from raganything.parser import MineruParser as _MineruParser
    _MineruParser.check_installation = lambda self: True
except Exception:
    pass

_original_load = _nvdb.load_storage


def _fast_load(file_name: str):
    json_path = Path(file_name)
    npy_path = json_path.with_suffix(".npy")
    pkl_path = json_path.with_suffix(".pkl")

    # Use fast files only if they exist and are at least as new as the JSON.
    if (
        npy_path.exists()
        and pkl_path.exists()
        and json_path.exists()
        and npy_path.stat().st_mtime >= json_path.stat().st_mtime
    ):
        t0 = time.time()
        matrix = np.load(str(npy_path))
        with open(pkl_path, "rb") as f:
            meta = pickle.load(f)
        elapsed = time.time() - t0
        n = matrix.shape[0]
        print(
            f"  [fast_storage] Loaded {json_path.stem}: "
            f"{n:,} vectors in {elapsed:.2f}s"
        )
        return {
            "embedding_dim": meta["embedding_dim"],
            "data": meta["data"],
            "matrix": matrix,
        }

    # Fallback: original slow JSON path.
    print(
        f"  [fast_storage] Fast files not found for {json_path.stem}, "
        "falling back to JSON (run migrate_storage.py to speed this up)."
    )
    return _original_load(file_name)


# Apply the patch.
_nvdb.load_storage = _fast_load
