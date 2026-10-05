"""Screen ground-truth questions for the two defects that silently ruin a run.

LEAKAGE -- a value that also appears in the source paper's body text cannot
discriminate multimodal retrieval from a text-only baseline, because the
text-only arm can reach it. Measured across the corpus, roughly 25% of table
values leak this way, so this is common rather than exceptional. Leaked
questions are not deleted: they are reassigned to the text-available stratum,
which is the control group.

UNIQUENESS -- if the same value appears in several papers, a system can answer
correctly from the wrong source and be scored a retrieval failure, or answer
from the right source for the wrong reason. Matching is done on word boundaries
so that "11.9" does not match inside "311.95"; an earlier version of this check
without boundaries reported 38 papers for a value that appears in one.

Writes stratum assignments back into data/ground_truth.jsonl.

Usage:  python screen.py [--write]
"""
import argparse
import json
import re
from collections import defaultdict

from paths import DATASET, CHUNKS_KV, RESULTS
from retrieval import is_multimodal_content

BODY_TYPES = {"None", "list", "aside_text", "equation", "ref_text", "page_footnote"}
TABLE_TYPES = {"table", "image", "chart"}   # "chart" is MinerU 3.4.4 only


# Separators may appear between any two digits: MinerU renders table cells as
# "103,791,051" and also as "2 473". Allowing the separator only before the
# decimal point (an earlier version of this function) silently failed to match
# any grouped number, which looked like the value being absent from the index.
SEP = r"[,\s  ]?"


def boundary_pattern(value):
    """Match the value as a standalone number, tolerating grouping separators."""
    digits = value.replace(",", "")
    if "." in digits:
        whole, frac = digits.split(".", 1)
        body = SEP.join(map(re.escape, whole)) + r"[.,]" + SEP.join(map(re.escape, frac))
    else:
        body = SEP.join(map(re.escape, digits))
    return re.compile(r"(?<![\d.,])" + body + r"(?![\d.,]*\d)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="write stratum assignments back into the dataset")
    args = ap.parse_args()

    rows = [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()]
    chunks = json.loads(CHUNKS_KV.read_text(encoding="utf-8"))

    by_file = defaultdict(lambda: {"body": [], "table": []})
    per_file_text = defaultdict(list)
    for c in chunks.values():
        f = c.get("file_path") or ""
        t = str(c.get("original_type"))
        content = c.get("content") or ""
        per_file_text[f].append(content)
        # A chunk carrying parsed table markup counts as table-derived whatever
        # its type label says; see MULTIMODAL_MARKERS in retrieval.py. Without
        # this, table content that MinerU also wrote into a plain-text chunk is
        # scored as "leaked to body text" and the question is wrongly moved out
        # of the table-exclusive stratum.
        if t in TABLE_TYPES or is_multimodal_content(content):
            by_file[f]["table"].append(content)
        elif t in BODY_TYPES:
            by_file[f]["body"].append(content)

    report = []
    for row in rows:
        src = row.get("source_filename")
        if not src:
            row["stratum"] = "pending-ingestion"
            report.append((row["id"], "-", "-", "source paper not indexed yet"))
            continue

        body = " ".join(by_file[src]["body"])
        table = " ".join(by_file[src]["table"])
        leaked, in_table, elsewhere = [], [], {}
        for v in row.get("gold_values", []):
            pat = boundary_pattern(v)
            if pat.search(body):
                leaked.append(v)
            if pat.search(table):
                in_table.append(v)
            n = sum(1 for f, texts in per_file_text.items()
                    if f != src and any(pat.search(t) for t in texts))
            if n:
                elsewhere[v] = n

        n_gold = len(row.get("gold_values", []))
        if row.get("evidence_type") == "text":
            stratum = "text-available"
        elif leaked and len(leaked) == n_gold:
            stratum = "text-available"      # every value reachable without tables
        elif leaked:
            stratum = "mixed"
        else:
            stratum = "table-exclusive"
        row["stratum"] = stratum

        notes = []
        if leaked:
            notes.append(f"leaked to body text: {leaked}")
        if not in_table and row.get("evidence_type") == "table":
            notes.append("NOT FOUND in this paper's table chunks")
        shared = {v: n for v, n in elsewhere.items() if n >= 3}
        if shared:
            notes.append(f"value also in other papers: {shared}")
        report.append((row["id"], stratum, f"{len(leaked)}/{n_gold}",
                       "; ".join(notes) or "clean"))

    print(f"{'id':6}{'stratum':18}{'leak':7}notes")
    for r in report:
        print(f"{r[0]:6}{r[1]:18}{r[2]:7}{r[3]}")

    counts = defaultdict(int)
    for row in rows:
        counts[row["stratum"]] += 1
    print("\nstrata:", dict(counts))

    (RESULTS / "screening.json").write_text(
        json.dumps([dict(zip(("id", "stratum", "leak", "notes"), r)) for r in report],
                   indent=2), encoding="utf-8")

    if args.write:
        with DATASET.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote strata back to {DATASET}")
    else:
        print("\n(dry run — pass --write to record strata in the dataset)")


if __name__ == "__main__":
    main()
