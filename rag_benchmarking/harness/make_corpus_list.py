"""Regenerate corpus_paper_list.csv from the index and papers_metadata.json.

A hand-generated listing goes stale the moment papers are ingested, which is
exactly what happened: the first version was written at 272 papers and did not
include the ten added in Sep 2026. Deriving it from the index means it cannot
drift.

Columns beyond the metadata:
    pdf_on_hand           whether the PDF exists in papers/ on this machine
    chunks_indexed        chunk count from kv_store_doc_status
    tables_figures_parsed whether the vision model processed the document

Written with a UTF-8 BOM so Excel on Windows renders accented author names.

Usage:  python make_corpus_list.py
"""
import csv
import json
import os

from paths import BENCH, PAPERS_METADATA, RAG_STORAGE, ROOT

OUT = BENCH / "corpus_paper_list.csv"
FIELDS = ["title", "authors", "year", "doi", "pdf_filename", "pdf_on_hand",
          "chunks_indexed", "tables_figures_parsed"]


def main():
    status = json.loads((RAG_STORAGE / "kv_store_doc_status.json").read_text(encoding="utf-8"))
    by_file = {}
    for v in status.values():
        name = os.path.basename((v.get("file_path") or "").replace("\\", "/"))
        if name:
            by_file[name] = v

    meta = json.loads(PAPERS_METADATA.read_text(encoding="utf-8"))
    papers_dir = ROOT / "papers"
    local = {f for f in os.listdir(papers_dir)} if papers_dir.exists() else set()

    rows = []
    for m in meta:
        fn = m["filename"]
        d = by_file.get(fn, {})
        rows.append({
            "title": m.get("title") or "(title missing - see filename)",
            "authors": m.get("authors") or "",
            "year": m.get("year") or "",
            "doi": m.get("doi") or "",
            "pdf_filename": fn,
            "pdf_on_hand": "yes" if fn in local else "no",
            "chunks_indexed": d.get("chunks_count", ""),
            "tables_figures_parsed": "yes" if str(d.get("multimodal_processed")) == "True" else "no",
        })

    rows.sort(key=lambda r: (-(int(r["year"]) if str(r["year"]).isdigit() else 0),
                             r["title"].lower()))

    with OUT.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    missing = [r["pdf_filename"] for r in rows if r["pdf_filename"] not in by_file]
    print(f"wrote {OUT}  ({len(rows)} papers)")
    print(f"  indexed      : {sum(1 for r in rows if r['chunks_indexed'] != '')}")
    print(f"  PDF on hand  : {sum(1 for r in rows if r['pdf_on_hand'] == 'yes')}")
    print(f"  with a DOI   : {sum(1 for r in rows if r['doi'])}")
    if missing:
        print(f"  IN METADATA BUT NOT INDEXED: {missing}")


if __name__ == "__main__":
    main()
