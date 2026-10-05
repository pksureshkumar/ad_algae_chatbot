"""Convert the reviewed ground-truth spreadsheet into a RAGAS-shaped dataset.

Field names match ragas SingleTurnSample (v0.4.3) so the scoring step can load
rows directly:

    user_input            <- question
    reference             <- ideal_answer
    reference_contexts    <- verbatim_passage (+ reference_context_2)
    reference_context_ids <- indexed filename of the source paper

Everything else is carried as metadata. `gold_values` is extracted here rather
than at scoring time so the numbers that define a correct answer are fixed with
the dataset and cannot drift between runs.

Usage:  python rag_benchmarking/harness/build_dataset.py [--include-flagged]
"""
import argparse
import json
import re

import openpyxl

from paths import GROUND_TRUTH_XLSX, DATASET, PAPERS_METADATA

# Values that carry meaning for scoring: decimals, or integers of 3+ digits.
# Two-digit integers match far too loosely against arbitrary text.
NUMERIC = re.compile(r"\d[\d,]*\.\d+|\d[\d,]{2,}")
# Bare years would otherwise be scored as gold values.
YEARISH = re.compile(r"^(19|20)\d{2}$")


def normalise_doi(value):
    if not value:
        return ""
    v = str(value).strip().lower()
    v = re.sub(r"^https?://(dx\.)?doi\.org/", "", v)
    return v.strip()


def extract_gold_values(text):
    out = []
    for raw in NUMERIC.findall(str(text or "")):
        clean = raw.replace(",", "").rstrip(".")
        if YEARISH.match(clean) or len(clean.replace(".", "")) < 3:
            continue
        if clean not in out:
            out.append(clean)
    return out


def load_doi_to_filename():
    records = json.loads(PAPERS_METADATA.read_text(encoding="utf-8"))
    return {normalise_doi(r.get("doi")): r["filename"] for r in records if r.get("doi")}


def load_pilot(start_index):
    """Merge the pilot question set.

    These five were constructed mechanically from table-exclusive values and
    verified against the index before the co-authors' set existed. They live in
    pilot/pilot_goldset.json rather than the spreadsheet, so they have to be
    merged explicitly -- leaving them out silently costs five questions in the
    table-exclusive stratum, which is the thinnest one.

    Their provenance differs from the rest: machine-proposed, not authored by a
    domain expert. Recorded as annotator "PILOT" so that distinction survives
    into the results and can be stated in the methods.
    """
    from paths import BENCH, ROOT
    pilot_path = BENCH / "pilot" / "pilot_goldset.json"
    if not pilot_path.exists():
        print(f"  (no pilot set at {pilot_path}, skipping)")
        return []

    chunks = json.loads((ROOT / "rag_storage" / "kv_store_text_chunks.json")
                        .read_text(encoding="utf-8"))
    out = []
    for n, p in enumerate(json.loads(pilot_path.read_text(encoding="utf-8"))):
        gold_chunk = chunks.get(p.get("gold_chunk_id"), {})
        out.append({
            "id": f"Q{start_index + n:02d}",
            "sheet_row": None,
            "user_input": p["q"],
            "reference": p["ideal"],
            "reference_contexts": [gold_chunk.get("content", "")[:4000]],
            "reference_context_ids": [p["pdf"]],
            "gold_values": p["gold"],
            "source_doi": "",
            "source_filename": p["pdf"],
            "indexed": True,
            "evidence_type": "table",
            "access": "",
            "year": None,
            "annotator": "PILOT",
            "page": None,
            "location_in_paper": "",
            "status": "OK",
            "stratum": None,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-flagged", action="store_true",
                    help="also emit rows marked NEEDS AUTHOR or MARGINAL")
    args = ap.parse_args()

    doi2file = load_doi_to_filename()
    ws = openpyxl.load_workbook(GROUND_TRUTH_XLSX)["Questions"]
    header = [c.value for c in ws[1]]

    rows, skipped = [], []
    for n, values in enumerate(ws.iter_rows(min_row=3, values_only=True), start=3):
        row = dict(zip(header, values))
        if not str(row.get("question") or "").strip():
            continue
        if str(row.get("annotator")) == "EXAMPLE":
            continue

        status = str(row.get("status") or "OK").upper()
        if status in {"NEEDS AUTHOR", "MARGINAL"} and not args.include_flagged:
            skipped.append((n, status, str(row.get("question"))[:60]))
            continue

        doi = normalise_doi(row.get("source_doi"))
        filename = doi2file.get(doi)

        contexts = [str(row.get("verbatim_passage") or "").strip()]
        extra = str(row.get("reference_context_2") or "").strip()
        if extra:
            contexts.append(extra)
        contexts = [c for c in contexts if c and c != "#VALUE!"]

        rows.append({
            "id": f"Q{len(rows) + 1:02d}",
            "sheet_row": n,
            "user_input": str(row.get("question")).strip(),
            "reference": str(row.get("ideal_answer") or "").strip(),
            "reference_contexts": contexts,
            "reference_context_ids": [filename] if filename else [],
            "gold_values": extract_gold_values(row.get("ideal_answer")),
            "source_doi": doi,
            "source_filename": filename,
            "indexed": bool(filename),
            "evidence_type": str(row.get("evidence_type") or "").strip().lower(),
            "access": str(row.get("Access") or "").strip(),
            "year": row.get("Year"),
            "annotator": str(row.get("annotator") or "").strip(),
            "page": row.get("page"),
            "location_in_paper": str(row.get("location_in_paper") or "").strip(),
            "status": status,
            # filled in by screen.py
            "stratum": None,
        })

    rows.extend(load_pilot(start_index=len(rows) + 1))

    with DATASET.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {DATASET}  ({len(rows)} questions)")
    n_indexed = sum(r["indexed"] for r in rows)
    print(f"  source paper indexed : {n_indexed}/{len(rows)}")
    print(f"  with pinned contexts : {sum(bool(r['reference_contexts']) for r in rows)}")
    print(f"  with gold values     : {sum(bool(r['gold_values']) for r in rows)}")
    by_type = {}
    for r in rows:
        by_type[r["evidence_type"]] = by_type.get(r["evidence_type"], 0) + 1
    print(f"  evidence_type        : {by_type}")
    if skipped:
        print(f"\n  skipped {len(skipped)} flagged rows (use --include-flagged to keep):")
        for n, status, q in skipped:
            print(f"    row {n} [{status}] {q}")
    missing = [r["id"] for r in rows if not r["indexed"]]
    if missing:
        print(f"\n  NOT YET INDEXED (cannot be run until ingestion completes): {missing}")


if __name__ == "__main__":
    main()
