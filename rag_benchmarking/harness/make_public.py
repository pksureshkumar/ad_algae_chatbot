"""Produce publication-safe copies of the run files.

The repository is public. runs/*.jsonl embeds every retrieved chunk in full --
1.4 MB of verbatim text drawn from papers, most of them paywalled. That is
redistribution, not incidental quotation, so the published copies keep
everything needed to verify scoring (chunk id, source paper, rank, similarity,
chunk type, and the model's answer) and drop the chunk bodies.

Anyone with the corpus can rebuild the full files by re-running the arms; anyone
without it can still check that every reported score follows from the recorded
retrieval and answers.

Usage:  python make_public.py
"""
import json

from paths import RUNS, BENCH

PUBLIC = BENCH / "runs_public"
NOTE = ("chunk text and pinned verbatim passages removed for publication "
        "(paywalled sources); rerun build_dataset.py and run_arms.py against "
        "the corpus to regenerate")


def main():
    PUBLIC.mkdir(exist_ok=True)
    total_removed = 0
    for src in sorted(RUNS.glob("*.jsonl")):
        out, removed = [], 0
        for line in src.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            for h in r.get("retrieved", []):
                removed += len(h.pop("content", "") or "")
                h["content_removed"] = True
            # reference_contexts are the pinned verbatim passages copied out of
            # the source papers. Short individually, but 28 kB per file and
            # repeated across six arms, and most sources are paywalled. The
            # paraphrased `reference` answer and the numeric `gold_values` carry
            # the same information for interpreting a score without reproducing
            # the papers' text.
            for v in (r.pop("reference_contexts", None) or []):
                removed += len(v)
            r["reference_contexts_removed"] = True
            r["_note"] = NOTE
            out.append(r)
        dst = PUBLIC / src.name
        with dst.open("w", encoding="utf-8") as fh:
            for r in out:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        total_removed += removed
        print(f"  {src.name:28} {len(out):3} rows, {removed/1000:7.1f} kB chunk text removed")
    print(f"\nwrote {PUBLIC}  (total {total_removed/1e6:.2f} MB removed)")


if __name__ == "__main__":
    main()
