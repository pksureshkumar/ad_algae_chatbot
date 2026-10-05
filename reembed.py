"""
reembed.py — Re-embed the existing index with the local Ollama model.

Replaces every vector in rag_storage/vdb_*.{json,npy,pkl} with one produced by
OLLAMA_EMBEDDING_MODEL, leaving all other content untouched. The knowledge graph
(entities, relations, descriptions, chunk text) is *not* rebuilt — only the
vectors change, so the corpus survives intact.

Why this exists instead of `ingest.py --reset`:
  - The index covers 282 papers, but only 157 of those PDFs exist locally.
    A re-ingest would silently shrink the corpus to whatever is in papers/.
  - Re-embedding takes hours; re-ingesting takes weeks.

Run:
    python reembed.py                # all three stores
    python reembed.py --store chunks # one store
    python reembed.py --dry-run      # report what would happen, embed nothing

Safe to interrupt: each store is written to temp files and swapped in only once
complete, and finished stores are skipped on the next run.
"""

import argparse
import asyncio
import base64
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

import httpx
import numpy as np
from openai import (
    AsyncOpenAI, APIConnectionError, APITimeoutError, BadRequestError,
    InternalServerError,
)

from config import (
    RAG_STORAGE_DIR, OLLAMA_BASE_URL, OLLAMA_EMBEDDING_MODEL,
    EMBEDDING_DIM, EMBEDDING_BATCH_SIZE,
)

STORES = ["vdb_chunks", "vdb_entities", "vdb_relationships"]
BACKUP_DIR = RAG_STORAGE_DIR / "_backup_azure_1536"

# Concurrency 4 measured marginally faster, but Ollama opens an internal
# connection per tokenize call and Windows ran out of sockets partway through
# a long run (WSAENOBUFS). 3 keeps the socket churn survivable.
CONCURRENCY = 3

# Rows between checkpoints. A crash costs at most this much work.
WINDOW = 20_000

_client: AsyncOpenAI | None = None


def client() -> AsyncOpenAI:
    global _client
    if _client is None:
        # An explicit keep-alive pool stops us adding our own connection churn
        # on top of Ollama's.
        http = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=CONCURRENCY * 2,
                                max_keepalive_connections=CONCURRENCY * 2,
                                keepalive_expiry=300.0),
            timeout=httpx.Timeout(600.0, connect=30.0),
        )
        _client = AsyncOpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama",
                              timeout=600.0, max_retries=0, http_client=http)
    return _client


def _is_transient(exc: Exception) -> bool:
    """Socket/connection failures worth waiting out.

    Ollama surfaces its own internal failures as HTTP 400, so a 400 is not
    automatically a permanent input problem — 'dial tcp ... buffer space' is a
    exhausted-socket condition that clears once TIME_WAIT drains.
    """
    if isinstance(exc, (APIConnectionError, APITimeoutError, InternalServerError)):
        return True
    msg = str(exc).lower()
    return any(s in msg for s in
               ("dial tcp", "buffer space", "socket", "eof", "connection reset",
                "timeout", "temporarily unavailable"))


def encode_vector(vec: np.ndarray) -> str:
    """Match LightRAG's per-record vector encoding exactly.

    Every record carries a *second* copy of its embedding in a "vector" field
    (float16 -> zlib -> base64), independent of the nano_vectordb matrix.
    LightRAG reads it via get_vectors_by_ids() to re-rank entity- and
    relation-related chunks. Miss it and the matrix says 1024-dim while these
    say 1536, so every similarity raises and retrieval silently degrades to the
    cruder WEIGHT fallback.
    """
    import zlib
    return base64.b64encode(
        zlib.compress(vec.astype(np.float16).tobytes())
    ).decode("utf-8")


def refresh_vector_fields(data: list, matrix: np.ndarray, label: str) -> None:
    for i, rec in enumerate(data):
        if "vector" in rec:
            rec["vector"] = encode_vector(matrix[i])
        if "__vector__" in rec:  # transient in-memory field; never persist it
            del rec["__vector__"]
        if i and i % 100_000 == 0:
            print(f"    {label}: re-encoded {i:,} vector fields", flush=True)


def load_meta(store: str) -> dict:
    """Read records from the .pkl — far cheaper than the multi-GB .json."""
    pkl = RAG_STORAGE_DIR / f"{store}.pkl"
    if not pkl.exists():
        raise SystemExit(
            f"{pkl.name} not found. Run migrate_storage.py first — reembed.py "
            "reads the pickle to avoid parsing a 7.7 GB JSON."
        )
    with open(pkl, "rb") as f:
        return pickle.load(f)


_truncated = 0  # how many records had to be shortened to fit the context


async def embed_batch(texts: list[str]) -> np.ndarray:
    """One embedding call, retrying through transient socket exhaustion.

    Backoff runs long on purpose: Windows needs a couple of minutes to drain
    TIME_WAIT sockets, so retrying quickly just fails again.
    """
    delays = [2, 5, 15, 30, 60, 120, 120, 180]
    for attempt, delay in enumerate([*delays, None]):
        try:
            resp = await client().embeddings.create(
                model=OLLAMA_EMBEDDING_MODEL, input=texts, encoding_format="float",
            )
            # Ollama does not guarantee response order matches input order.
            ordered = sorted(resp.data, key=lambda d: d.index)
            return np.asarray([d.embedding for d in ordered], dtype=np.float32)
        except BadRequestError as e:
            # Ollama reports its own infrastructure failures as HTTP 400 with a
            # free-form message ('dial tcp ...', 'wsarecv: connection forcibly
            # closed', ...). Enumerating those strings proved fragile, so invert
            # it: the only genuinely permanent 400 here is context overflow,
            # which embed_adaptive handles by shortening the text.
            if "context length" in str(e).lower():
                raise
            if delay is None:
                raise
            print(f"      transient error, retry {attempt + 1} in {delay}s: "
                  f"{str(e)[:110]}", flush=True)
            await asyncio.sleep(delay)
        except Exception as e:
            if delay is None or not _is_transient(e):
                raise
            print(f"      transient error, retry {attempt + 1} in {delay}s: "
                  f"{str(e)[:110]}", flush=True)
            await asyncio.sleep(delay)
    raise RuntimeError("unreachable")


async def embed_adaptive(texts: list[str]) -> np.ndarray:
    """Embed a batch, coping with items that overflow the model context.

    This corpus is dense with LaTeX ($\\mathrm{CO}_{2}$ and friends), which
    tokenises far worse than its character count suggests, so a fixed character
    cap would either truncate thousands of good records or still overflow.
    Instead: bisect the batch to isolate the offender, then shorten only that
    record, keeping as much of it as fits.
    """
    global _truncated
    try:
        return await embed_batch(texts)
    except BadRequestError as e:
        if "context length" not in str(e).lower():
            raise
    if len(texts) > 1:
        mid = len(texts) // 2
        left, right = await asyncio.gather(
            embed_adaptive(texts[:mid]), embed_adaptive(texts[mid:])
        )
        return np.vstack([left, right])

    # Single oversized record: keep the largest prefix the model will accept.
    text = texts[0]
    for ratio in (0.75, 0.5, 0.35, 0.25, 0.15, 0.08):
        try:
            vec = await embed_batch([text[:max(1, int(len(text) * ratio))]])
            _truncated += 1
            return vec
        except BadRequestError:
            continue
    raise RuntimeError(f"Could not embed a {len(text)}-char record even at 8%.")


async def embed_into(matrix: np.memmap, texts: list[str], label: str,
                     start_row: int, checkpoint) -> None:
    """Fill `matrix` from `start_row` on. Bounded memory: vectors go to disk.

    Rows are processed in sequential windows so progress is a simple prefix and
    can be checkpointed; batches inside a window still run concurrently.
    """
    total = len(texts)
    t0 = time.time()
    done_at_start = start_row

    for win_lo in range(start_row, total, WINDOW):
        win_hi = min(win_lo + WINDOW, total)
        slices = [(i, min(i + EMBEDDING_BATCH_SIZE, win_hi))
                  for i in range(win_lo, win_hi, EMBEDDING_BATCH_SIZE)]
        sem = asyncio.Semaphore(CONCURRENCY)
        done = win_lo

        async def worker(lo: int, hi: int):
            nonlocal done
            async with sem:
                vecs = await embed_adaptive(texts[lo:hi])
                if vecs.shape != (hi - lo, EMBEDDING_DIM):
                    raise RuntimeError(
                        f"{label}: expected {(hi-lo, EMBEDDING_DIM)} got {vecs.shape}. "
                        "Check EMBEDDING_DIM matches the model."
                    )
                # nano_vectordb normalises on load and on query; store normalised
                # so the file matches what an upsert would have written.
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                matrix[lo:hi] = vecs / np.where(norms == 0, 1, norms)
                done += hi - lo
                if done % 5000 < EMBEDDING_BATCH_SIZE or done == win_hi:
                    rate = (done - done_at_start) / max(time.time() - t0, 1e-9)
                    eta = (total - done) / max(rate, 1e-9) / 60
                    print(f"    {label}: {done:,}/{total:,} ({100*done/total:5.1f}%) "
                          f"{rate:5.1f} vec/s  ETA {eta:5.1f} min", flush=True)

        await asyncio.gather(*(worker(lo, hi) for lo, hi in slices))
        matrix.flush()
        checkpoint(win_hi)


def write_json(path: Path, data: list, matrix: np.memmap) -> None:
    """Stream the NanoVectorDB JSON so peak memory stays bounded.

    query.py and batch_query.py load the JSON directly (they do not import
    fast_storage), so it must agree with the .npy rather than be left stale.
    """
    # 3 rows * dim * 4 bytes is divisible by 3, so base64 chunks concatenate
    # without padding artefacts.
    rows_per_chunk = 3 * 1024
    with open(path, "w", encoding="utf-8") as f:
        f.write(f'{{"embedding_dim": {EMBEDDING_DIM}, "data": [')
        for i, rec in enumerate(data):
            if i:
                f.write(",")
            f.write(json.dumps(rec, ensure_ascii=False))
        f.write('], "matrix": "')
        for lo in range(0, matrix.shape[0], rows_per_chunk):
            block = np.ascontiguousarray(matrix[lo:lo + rows_per_chunk])
            f.write(base64.b64encode(block.tobytes()).decode())
        f.write('"}')


def backup_originals(store: str) -> None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for ext in (".json", ".npy", ".pkl"):
        src = RAG_STORAGE_DIR / f"{store}{ext}"
        dst = BACKUP_DIR / f"{store}{ext}"
        if src.exists() and not dst.exists():
            print(f"    backing up {src.name} -> {dst.relative_to(RAG_STORAGE_DIR)}")
            shutil.move(str(src), str(dst))


async def process(store: str, dry_run: bool) -> None:
    print(f"\n[{store}]")
    done_marker = RAG_STORAGE_DIR / f"{store}.reembedded"
    if done_marker.exists():
        print("    already re-embedded, skipping.")
        return

    meta = load_meta(store)
    data = meta["data"]
    n = len(data)
    old_dim = meta.get("embedding_dim")
    print(f"    {n:,} records | old dim {old_dim} -> new dim {EMBEDDING_DIM}")

    missing = sum(1 for d in data if not d.get("content"))
    if missing:
        print(f"    WARNING: {missing:,} records have empty content; "
              "they will get a zero-information vector.")

    if dry_run:
        print(f"    dry run — would embed {n:,} texts "
              f"(~{n/57/3600:.2f} h at 57 vec/s)")
        return

    tmp_npy = RAG_STORAGE_DIR / f"{store}.npy.tmp"
    prog_file = RAG_STORAGE_DIR / f"{store}.reembed_progress.json"
    signature = {"model": OLLAMA_EMBEDDING_MODEL, "dim": EMBEDDING_DIM, "n": n}

    # Resume only if the partial file was produced by this same model/shape.
    start_row = 0
    if tmp_npy.exists() and prog_file.exists():
        saved = json.loads(prog_file.read_text())
        if {k: saved.get(k) for k in signature} == signature:
            start_row = int(saved.get("rows_done", 0))
            print(f"    resuming at row {start_row:,} "
                  f"({100*start_row/n:.1f}% already embedded)")
        else:
            print("    partial file does not match current settings, restarting.")
            tmp_npy.unlink(missing_ok=True)

    if start_row and tmp_npy.exists():
        matrix = np.lib.format.open_memmap(tmp_npy, mode="r+")
    else:
        start_row = 0
        matrix = np.lib.format.open_memmap(
            tmp_npy, mode="w+", dtype=np.float32, shape=(n, EMBEDDING_DIM)
        )

    def checkpoint(rows_done: int) -> None:
        prog_file.write_text(json.dumps({**signature, "rows_done": rows_done}))

    texts = [d.get("content") or "" for d in data]

    # Drop the parsed records for the duration of the embedding phase. For
    # relationships that is ~3 GB of dicts versus ~200 MB of strings, and the
    # box only has a few GB free — Ollama's runner sharing this machine is the
    # likeliest reason connections were being dropped mid-run.
    del data, meta

    await embed_into(matrix, texts, store, start_row, checkpoint)
    matrix.flush()
    del texts

    # Reload for the write phase; the original .pkl is still untouched here.
    meta = load_meta(store)
    data = meta["data"]
    if len(data) != n:
        raise RuntimeError(f"{store}: record count changed underneath us "
                           f"({len(data)} vs {n}).")

    backup_originals(store)

    print("    re-encoding per-record vector fields ...", flush=True)
    refresh_vector_fields(data, matrix, store)

    print("    writing .pkl ...", flush=True)
    with open(RAG_STORAGE_DIR / f"{store}.pkl", "wb") as f:
        pickle.dump({"embedding_dim": EMBEDDING_DIM, "data": data},
                    f, protocol=pickle.HIGHEST_PROTOCOL)

    print("    writing .json ...", flush=True)
    write_json(RAG_STORAGE_DIR / f"{store}.json", data, matrix)

    del data
    matrix.flush()
    del matrix
    tmp_npy.replace(RAG_STORAGE_DIR / f"{store}.npy")
    prog_file.unlink(missing_ok=True)

    # .npy must be at least as new as .json or fast_storage ignores it.
    npy = RAG_STORAGE_DIR / f"{store}.npy"
    npy.touch()
    done_marker.write_text(f"{OLLAMA_EMBEDDING_MODEL} dim={EMBEDDING_DIM}\n")
    print(f"    done: {n:,} vectors"
          + (f" ({_truncated:,} shortened to fit context)" if _truncated else ""))


def repair_vectors(store: str) -> None:
    """Rewrite only the per-record "vector" fields from the existing .npy.

    For recovering from a re-embed that updated the matrix but left the
    compressed per-record copies stale. No embedding calls — pure CPU.
    """
    print(f"\n[{store}] repairing vector fields")
    matrix = np.load(RAG_STORAGE_DIR / f"{store}.npy", mmap_mode="r")
    meta = load_meta(store)
    data = meta["data"]
    if len(data) != matrix.shape[0]:
        raise RuntimeError(f"{store}: {len(data)} records vs {matrix.shape[0]} rows.")
    if meta.get("embedding_dim") != matrix.shape[1]:
        raise RuntimeError(f"{store}: pkl dim {meta.get('embedding_dim')} "
                           f"vs matrix dim {matrix.shape[1]}.")
    print(f"    {len(data):,} records | dim {matrix.shape[1]}")

    refresh_vector_fields(data, matrix, store)

    print("    writing .pkl ...", flush=True)
    with open(RAG_STORAGE_DIR / f"{store}.pkl", "wb") as f:
        pickle.dump({"embedding_dim": int(matrix.shape[1]), "data": data},
                    f, protocol=pickle.HIGHEST_PROTOCOL)
    print("    writing .json ...", flush=True)
    write_json(RAG_STORAGE_DIR / f"{store}.json", data, matrix)
    del data, meta
    (RAG_STORAGE_DIR / f"{store}.npy").touch()  # keep .npy newer than .json
    print("    done")


async def main(stores: list[str], dry_run: bool) -> None:
    print("=" * 68)
    print(f"  Re-embedding with {OLLAMA_EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  storage: {RAG_STORAGE_DIR}")
    print("=" * 68)

    t0 = time.time()
    for store in stores:
        await process(store, dry_run)
    print(f"\n{'=' * 68}")
    print(f"  Finished in {(time.time() - t0) / 60:.1f} min")
    if not dry_run:
        print(f"  Originals moved to {BACKUP_DIR.name}/ — delete once verified.")
    print("=" * 68)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Re-embed the index with a local model.")
    ap.add_argument("--store", choices=[s.replace("vdb_", "") for s in STORES],
                    help="Re-embed only one store (default: all).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report sizes and estimates without embedding.")
    ap.add_argument("--repair-vectors", action="store_true",
                    help="Rebuild only the per-record 'vector' fields from the "
                         "existing .npy. No embedding calls.")
    a = ap.parse_args()
    chosen = [f"vdb_{a.store}"] if a.store else STORES
    try:
        if a.repair_vectors:
            t0 = time.time()
            for s in chosen:
                repair_vectors(s)
            print(f"\nRepair finished in {(time.time() - t0) / 60:.1f} min")
            sys.exit(0)
        asyncio.run(main(chosen, a.dry_run))
    except KeyboardInterrupt:
        print("\nInterrupted — completed stores are kept, partial work discarded.")
        sys.exit(130)
