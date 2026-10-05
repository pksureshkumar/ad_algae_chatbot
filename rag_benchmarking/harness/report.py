"""Consolidate every scored arm into one report.

Usage:  python report.py
"""
import json
from collections import defaultdict
from math import comb

from paths import DATASET, RESULTS, RUNS

ORDER = ["multimodal", "text_only", "no_retrieval", "gemma3_no_retrieval",
         "gpt_strict", "gpt_websearch"]
STRATA = ["table-exclusive", "mixed", "text-available"]


def load(arm):
    p = RESULTS / f"{arm}.scored.jsonl"
    if not p.exists():
        return None
    return {s["id"]: s for s in map(json.loads, p.read_text(encoding="utf-8").splitlines())}


def mcnemar(a, b, key="gold_chunk_retrieved"):
    """Exact two-sided McNemar on paired binary outcomes."""
    ids = sorted(set(a) & set(b))
    only_a = sum(1 for i in ids if a[i][key] and not b[i][key])
    only_b = sum(1 for i in ids if not a[i][key] and b[i][key])
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 1.0
    p = sum(comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2 ** n * 2
    return only_a, only_b, min(p, 1.0)


def main():
    rows = {r["id"]: r for r in map(json.loads, DATASET.read_text(encoding="utf-8").splitlines())}
    arms = {a: d for a in ORDER if (d := load(a))}
    if not arms:
        raise SystemExit("no scored runs found — run score.py first")

    L = ["# Benchmark results", "",
         f"{len(rows)} questions, each pinned to a specific value in a specific paper.",
         "All local arms use qwen3:14b at temperature 0.1 unless noted.", "",
         "## Overall", "",
         "| arm | gold chunk retrieved | values correct | mean accuracy | fabrications |",
         "|---|---|---|---|---|"]

    for a, d in arms.items():
        n = len(d)
        cf = all(s["context_free"] for s in d.values())
        chunk = "n/a" if cf else f"{sum(bool(s['gold_chunk_retrieved']) for s in d.values())}/{n}"
        vals = sum(len(s["substantiated"]) for s in d.values())
        tot = sum(s["n_gold_values"] for s in d.values())
        acc = sum(s["accuracy"] for s in d.values() if s["accuracy"] is not None) / n
        fab = "n/a" if cf else sum(len(s["fabricated"]) for s in d.values())
        L.append(f"| `{a}` | {chunk} | {vals}/{tot} | {acc:.3f} | {fab} |")

    L += ["", "## By stratum", "",
          "The text-available stratum is the control group: values there also appear in",
          "body text, so a text-only baseline can reach them. Equivalent performance in",
          "that stratum, with divergence elsewhere, is what shows the advantage is",
          "specific to table-derived content rather than general retrieval superiority.",
          "",
          "| stratum | n | arm | gold chunk | values correct |", "|---|---|---|---|---|"]

    by_stratum = defaultdict(list)
    for qid, r in rows.items():
        by_stratum[r["stratum"]].append(qid)

    for st in STRATA:
        ids = by_stratum.get(st, [])
        if not ids:
            continue
        for a, d in arms.items():
            sel = [d[i] for i in ids if i in d]
            if not sel:
                continue
            cf = all(s["context_free"] for s in sel)
            chunk = "n/a" if cf else f"{sum(bool(s['gold_chunk_retrieved']) for s in sel)}/{len(sel)}"
            vals = sum(len(s["substantiated"]) for s in sel)
            tot = sum(s["n_gold_values"] for s in sel)
            L.append(f"| {st} | {len(ids)} | `{a}` | {chunk} | {vals}/{tot} |")

    L += ["", "## Paired tests", ""]
    if "multimodal" in arms and "text_only" in arms:
        b, c, p = mcnemar(arms["multimodal"], arms["text_only"])
        L += [f"**multimodal vs text_only**, gold-chunk retrieval: {b} questions favour",
              f"multimodal, {c} favour text-only, {b + c} discordant pairs, "
              f"McNemar exact two-sided **p = {p:.5f}**.", ""]

    L += ["## Caveats", "",
          "- Known-item retrieval, not exhaustive recall: each question pins one passage",
          "  we know contains the answer. This bounds retrieval failure from below; it does",
          "  not measure everything the system missed across the corpus.",
          "- Context-free arms answer from parametric knowledge, so retrieval metrics and",
          "  substantiation are undefined for them, not zero.",
          "- Questions are deliberately specific and mostly table-derived. Ungrounded",
          "  accuracy would be higher on general questions; this is not a general",
          "  capability assessment.",
          "- Three questions (Q01, Q05, Q07) miss the source paper in both retrieval arms.",
          "  They do not identify which of 282 similar papers they refer to.",
          ""]

    out = RESULTS / "REPORT.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
