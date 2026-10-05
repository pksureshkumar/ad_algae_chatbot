"""Shared path resolution.

Every harness script resolves its paths from __file__ rather than the working
directory, so they behave the same whether run from the project root or from
inside rag_benchmarking/.
"""
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
BENCH = HARNESS.parent
ROOT = BENCH.parent

DATA = BENCH / "data"
RUNS = BENCH / "runs"
RESULTS = BENCH / "results"
GROUND_TRUTH_XLSX = BENCH / "ground_truth_final.xlsx"
DATASET = DATA / "ground_truth.jsonl"

RAG_STORAGE = ROOT / "rag_storage"
CHUNKS_KV = RAG_STORAGE / "kv_store_text_chunks.json"
VDB_NPY = RAG_STORAGE / "vdb_chunks.npy"
VDB_PKL = RAG_STORAGE / "vdb_chunks.pkl"
PAPERS_METADATA = ROOT / "papers_metadata.json"

for d in (DATA, RUNS, RESULTS):
    d.mkdir(parents=True, exist_ok=True)


def add_root_to_path():
    """Make the project's config/models importable from harness scripts.

    core/ is added as well as the root: the modules in it import each other by
    bare name, so `import config` only resolves if core/ is itself on the path.
    """
    for p in (ROOT, ROOT / "core"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
