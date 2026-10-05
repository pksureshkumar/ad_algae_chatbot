"""Remove duplicated papers from the index.

Eight papers were ingested twice, each under two filenames - typically a
descriptive name from the original March/April 2026 build and an Elsevier
PII-style name added weeks later. Every pair has an identical chunk count and
identical title, and the two copies share no chunk ids, so each occupies an
independent set of chunks, entities and relationships. The duplicates compete
with each other in every top-k retrieval and inflate the corpus count.

Six of the eight pairs were invisible to a DOI-based duplicate check, because one
copy in each pair had no DOI recorded at the time; two more only became visible
after seven wrongly-assigned DOIs were corrected. Matching on normalised title as
well as DOI is what surfaces them.

Deletion goes through LightRAG's `adelete_by_doc_id`, which removes the document,
its chunks and its vectors, and reconciles entities and relationships that
referenced them. Hand-editing the stores would leave entity descriptions pointing
at chunks that no longer exist.

The copy kept in each pair is the earlier-ingested one, which is also the one the
rest of the project references (`insights_microalgaebased_technologies.pdf`, for
instance, is the source for two benchmark questions).

A full copy of rag_storage/ must exist before running this. The index cannot be
rebuilt: 117 of its source PDFs no longer exist anywhere.

Usage:
    python deduplicate_corpus.py --dry-run
    python deduplicate_corpus.py
"""
import _bootstrap  # noqa: F401  puts the project root on sys.path
import argparse
import asyncio
import json
import os
import shutil
from pathlib import Path

import fast_storage  # noqa: F401  must precede any LightRAG import

from raganything import RAGAnything

from config import RAG_CONFIG, RAG_STORAGE_DIR
from models import llm_model_func, embedding_func, vision_model_func

from _bootstrap import ROOT
METADATA = ROOT / "papers_metadata.json"
BACKUP = ROOT / "rag_storage_predup_backup"

# keep -> delete
PAIRS = [
    ("ad_food_waste_coupled.pdf", "1-s2.0-S001623612201403X-main.pdf"),
    ("breaking_barriers_largescale_microalgae.pdf", "1-s2.0-S0360319925051365-main.pdf"),
    ("digestate_dilution_shapes_carb.pdf", "1-s2.0-S2211926425004011-main.pdf"),
    ("insights_microalgaebased_technologies.pdf", "1-s2.0-S0960852425005577-main.pdf"),
    ("methane_production_enhancement_tetraselmis.pdf", "1-s2.0-S2352186423004741-main.pdf"),
    ("upgrading_algae_waste_3d.pdf", "1-s2.0-S1385894724077404-main.pdf"),
    ("1-s2.0-S0960852424017048-main.pdf", "1-s2.0-S0960852424017048-main (1).pdf"),
    ("temporal-dynamics-and-contribution-of-phage-community-to-the-prevalence-of-"
     "antibiotic-resistance-genes-in-a-full-scale.pdf",
     "temporal-dynamics-and-contribution-of-phage-community-to-the-prevalence-of-"
     "antibiotic-resistance-genes-in-a-full-scale (1).pdf"),
]


def doc_ids_by_filename():
    status = json.loads((RAG_STORAGE_DIR / "kv_store_doc_status.json").read_text(encoding="utf-8"))
    out = {}
    for doc_id, v in status.items():
        name = os.path.basename((v.get("file_path") or "").replace("\\", "/"))
        if name:
            out[name] = (doc_id, v.get("chunks_count"))
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not BACKUP.exists():
        raise SystemExit(f"refusing to run: no index backup at {BACKUP}")

    index = doc_ids_by_filename()
    plan = []
    for keep, drop in PAIRS:
        k, d = index.get(keep), index.get(drop)
        if d is None:
            print(f"  SKIP  {drop[:56]}  (already absent)")
            continue
        if k is None:
            print(f"  SKIP  {drop[:56]}  (the copy to KEEP is missing - not deleting)")
            continue
        if k[1] != d[1]:
            print(f"  WARN  chunk counts differ ({k[1]} vs {d[1]}) for\n"
                  f"          keep {keep[:60]}\n          drop {drop[:60]}")
        plan.append((keep, drop, d[0]))

    print(f"\n{len(plan)} documents to delete:")
    for keep, drop, doc_id in plan:
        print(f"   drop {drop[:62]:64} {doc_id}")
        print(f"   keep {keep[:62]}")
    if args.dry_run:
        print("\n--dry-run: nothing deleted")
        return
    if not plan:
        print("nothing to do")
        return

    rag = RAGAnything(config=RAG_CONFIG, llm_model_func=llm_model_func,
                      embedding_func=embedding_func, vision_model_func=vision_model_func)
    await rag._ensure_lightrag_initialized()

    ok, failed = [], []
    for keep, drop, doc_id in plan:
        print(f"\ndeleting {drop[:62]} ...", flush=True)
        try:
            result = await rag.lightrag.adelete_by_doc_id(doc_id)
            status = getattr(result, "status", None) or getattr(result, "message", result)
            print(f"  -> {status}")
            ok.append(drop)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            failed.append(drop)

    await rag.lightrag.finalize_storages()

    if ok:
        meta = json.loads(METADATA.read_text(encoding="utf-8"))
        before = len(meta)
        meta = [m for m in meta if m["filename"] not in set(ok)]
        shutil.copy(METADATA, METADATA.with_suffix(".json.pre-dedup"))
        METADATA.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"\npapers_metadata.json: {before} -> {len(meta)} records")

    print(f"\ndeleted {len(ok)}, failed {len(failed)}")
    if failed:
        print("  failed:", failed)
    print("\nNEXT: python migrate_storage.py   (rebuild the .npy/.pkl fast-load pair)")


if __name__ == "__main__":
    asyncio.run(main())
