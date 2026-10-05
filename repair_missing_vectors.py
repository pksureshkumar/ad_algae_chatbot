"""Embed chunks that exist in kv_store_text_chunks.json but not in vdb_chunks.

Why this is needed: the 29 Sep ingestion of 10 papers reported "Done" for every
file and exited 0, but 1,329 chunks across five of those papers were written to
the chunk store without ever being embedded into the vector store. Those papers
were therefore present in the knowledge base and completely unsearchable, which
silently scored benchmark questions as retrieval failures.

The record format is matched exactly to what an upsert produces:
  - matrix rows are L2-normalised float32, base64 of the raw buffer
  - each record also carries its own copy of the embedding in "vector"
    (float16 -> zlib -> base64), which LightRAG reads via get_vectors_by_ids()
  - .json, .npy and .pkl are all rewritten, and .npy is touched last so
    fast_storage's mtime check does not fall back to the slow JSON path

Usage:
    python repair_missing_vectors.py --dry-run
    python repair_missing_vectors.py
"""
import argparse
import asyncio
import base64
import json
import pickle
import shutil
import time
import zlib
from pathlib import Path

import numpy as np

import _env  # noqa: F401
from models import embedding_func

STORAGE = Path(__file__).parent / "rag_storage"
CHUNKS_KV = STORAGE / "kv_store_text_chunks.json"
VDB_JSON = STORAGE / "vdb_chunks.json"
VDB_NPY = STORAGE / "vdb_chunks.npy"
VDB_PKL = STORAGE / "vdb_chunks.pkl"
BATCH = 32


def encode_record_vector(vec):
    """float32 unit vector -> the compressed form stored per record."""
    return base64.b64encode(zlib.compress(vec.astype(np.float16).tobytes())).decode()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    chunks = json.loads(CHUNKS_KV.read_text(encoding="utf-8"))
    vdb = json.loads(VDB_JSON.read_text(encoding="utf-8"))
    dim = vdb["embedding_dim"]
    have = {r["__id__"] for r in vdb["data"]}
    missing = [(cid, c) for cid, c in chunks.items() if cid not in have]

    by_file = {}
    for _, c in missing:
        f = c.get("file_path") or "?"
        by_file[f] = by_file.get(f, 0) + 1
    print(f"chunk store {len(chunks)} | vector store {len(have)} | missing {len(missing)}")
    for f, n in sorted(by_file.items(), key=lambda kv: -kv[1]):
        print(f"   {n:5}  {f}")
    if not missing:
        print("nothing to repair")
        return
    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    for p in (VDB_JSON, VDB_NPY, VDB_PKL):
        shutil.copy(p, p.with_suffix(p.suffix + ".prerepair"))
    print(f"\nbacked up to *{VDB_JSON.suffix}.prerepair etc.")

    texts = [c.get("content") or "" for _, c in missing]
    vectors = []
    t0 = time.time()
    for i in range(0, len(texts), BATCH):
        got = await embedding_func(texts[i:i + BATCH])
        vectors.extend(got)
        done = min(i + BATCH, len(texts))
        print(f"  embedded {done}/{len(texts)}  ({done / max(time.time() - t0, 1e-9):.0f}/s)",
              flush=True)

    new = np.asarray(vectors, dtype=np.float32)
    if new.shape != (len(missing), dim):
        raise SystemExit(f"bad embedding shape {new.shape}, expected {(len(missing), dim)}")
    new /= np.linalg.norm(new, axis=1, keepdims=True) + 1e-12

    old = np.frombuffer(base64.b64decode(vdb["matrix"]), dtype=np.float32).reshape(-1, dim)
    if old.shape[0] != len(vdb["data"]):
        raise SystemExit("existing matrix rows do not match record count; aborting")
    matrix = np.vstack([old, new])

    now = int(time.time())
    for (cid, c), vec in zip(missing, new):
        vdb["data"].append({
            "__id__": cid,
            "__created_at__": now,
            "content": c.get("content") or "",
            "full_doc_id": c.get("full_doc_id") or "",
            "file_path": c.get("file_path") or "",
            "vector": encode_record_vector(vec),
        })
    vdb["matrix"] = base64.b64encode(matrix.astype(np.float32).tobytes()).decode()

    VDB_JSON.write_text(json.dumps(vdb), encoding="utf-8")
    VDB_PKL.write_bytes(pickle.dumps({"embedding_dim": dim, "data": vdb["data"]},
                                     protocol=pickle.HIGHEST_PROTOCOL))
    np.save(VDB_NPY, matrix)
    VDB_NPY.touch()  # must be newest, or fast_storage falls back to JSON

    print(f"\nvector store now {matrix.shape[0]} rows / {len(vdb['data'])} records")
    print("rewrote vdb_chunks.json, .pkl, .npy")


if __name__ == "__main__":
    asyncio.run(main())
