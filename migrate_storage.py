"""
migrate_storage.py — One-time migration for faster startup.

Converts NanoVectorDB JSON files (slow to parse) into:
  - vdb_*.npy   — binary float32 matrix (loads in seconds instead of minutes)
  - vdb_*.pkl   — metadata pickle (much faster than JSON)

Run once:
    python migrate_storage.py

After migration, chat.py and app.py will automatically use the fast files.
The original .json files are NOT deleted — they remain as the authoritative backup.
"""

import base64
import json
import pickle
import time
from pathlib import Path

import numpy as np

RAG_STORAGE = Path("rag_storage")
VDB_FILES = ["vdb_chunks.json", "vdb_entities.json", "vdb_relationships.json"]


def migrate(json_path: Path) -> None:
    stem = json_path.stem
    npy_out = json_path.with_suffix(".npy")
    pkl_out = json_path.with_suffix(".pkl")

    if npy_out.exists() and pkl_out.exists():
        npy_newer = npy_out.stat().st_mtime >= json_path.stat().st_mtime
        if npy_newer:
            print(f"  {json_path.name}: already migrated and up to date, skipping.")
            return
        print(f"  {json_path.name}: source JSON is newer — re-migrating.")

    size_gb = json_path.stat().st_size / 1e9
    print(f"\n[{stem}] Reading {size_gb:.2f} GB JSON... ", end="", flush=True)
    t0 = time.time()

    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)

    t1 = time.time()
    print(f"done in {t1 - t0:.1f}s")

    dim = raw["embedding_dim"]
    matrix = np.frombuffer(base64.b64decode(raw["matrix"]), dtype=np.float32).reshape(-1, dim)
    n_vectors = matrix.shape[0]
    print(f"  Vectors: {n_vectors:,}  |  dim: {dim}")

    # Save binary matrix
    print(f"  Saving matrix → {npy_out.name} ... ", end="", flush=True)
    t2 = time.time()
    np.save(npy_out, matrix)
    t3 = time.time()
    print(f"done in {t3 - t2:.1f}s  ({npy_out.stat().st_size / 1e9:.2f} GB)")

    # Save metadata (data list only, no matrix) as pickle
    print(f"  Saving metadata → {pkl_out.name} ... ", end="", flush=True)
    t4 = time.time()
    meta = {"embedding_dim": dim, "data": raw["data"]}
    with open(pkl_out, "wb") as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
    t5 = time.time()
    print(f"done in {t5 - t4:.1f}s  ({pkl_out.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    print("=" * 60)
    print("  NanoVectorDB → Fast Storage Migration")
    print("=" * 60)

    if not RAG_STORAGE.exists():
        print("ERROR: rag_storage/ not found. Run from the project root.")
        return

    total_start = time.time()
    for fname in VDB_FILES:
        src = RAG_STORAGE / fname
        if not src.exists():
            print(f"\n  {fname}: not found, skipping.")
            continue
        migrate(src)

    elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"  Migration complete in {elapsed / 60:.1f} min.")
    print("  chat.py and app.py will now start much faster.")
    print("=" * 60)


if __name__ == "__main__":
    main()
