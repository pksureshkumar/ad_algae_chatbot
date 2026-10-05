"""Deterministic scoring: retrieval success and answer substantiation.

No LLM judge is involved here, so nothing in this file is exposed to the
self-preference objection that applies to LLM-scored metrics. RAGAS metrics are
computed separately by ragas_eval.py in its own environment.

The substantiation check exists because of a concrete pilot failure. Asked about
Korsmeyer-Peppas parameters, the text-only arm reported Kkp = 35.38 and n = 0.19
and cited a source for them. Both values were numerically correct, and both
appeared in exactly one chunk of that paper -- the table the arm was blocked from
retrieving. They were not in its context. Scoring on "does the gold value appear
in the answer" awarded that fabrication 2/3.

So every gold value is classified three ways:

    substantiated : in the answer AND in the retrieved context   (genuine)
    fabricated    : in the answer but NOT in the retrieved context (invented)
    missed        : not in the answer

A system's headline accuracy is substantiated/total. Fabrication is tracked
separately as a rate, because a confidently wrong answer is worse than an
abstention, not equivalent to one.

Usage:  python score.py --run ../runs/<arm>.jsonl
"""
import argparse
import json
import re
from pathlib import Path

from paths import RUNS, RESULTS


def normalise(text):
    """Strip separators so '6,034' matches '6034' and '19 %' matches '19%'."""
    return re.sub(r"[\s,  ]", "", str(text or ""))


def value_present(value, text):
    return normalise(value) in normalise(text)


# An honest abstention and a confident fabrication both score zero accuracy, but
# they are not the same failure: one is the behaviour you want from a research
# tool, the other is the behaviour that corrupts a dataset. Tracking abstention
# separately is what lets the two be reported apart. Observed in practice:
# qwen3:14b invented a value and a citation for Q01, while GPT-6 Astra declined
# and said it could not identify the paper.
ABSTAIN = re.compile(
    r"\b(?:do not|don't|does not|doesn't) know\b"
    r"|\bnot (?:known|aware|able) to\b"
    r"|\bcannot (?:reliably )?(?:identify|determine|provide|confirm|verify)\b"
    r"|\bI (?:do not|don't) have\b"
    r"|\bno (?:specific|verified|reliable) (?:value|data|figure|information)\b"
    r"|\bvalue not known\b"
    r"|\binsufficient (?:information|evidence|context)\b"
    r"|\bnot (?:explicitly )?(?:reported|stated|provided) in\b"
    r"|\bunable to\b",
    re.I,
)


def is_abstention(text):
    return bool(ABSTAIN.search(text or ""))


def score_row(row):
    gold_values = row.get("gold_values") or []
    answer = row.get("answer") or ""
    hits = row.get("retrieved") or []
    context = " ".join(h.get("content", "") for h in hits)

    # A context-free arm (no_retrieval, gpt_strict, gpt_open) is answering from
    # parametric knowledge, so there is no retrieved context to check a value
    # against. Substantiation and the retrieval metrics are undefined, not zero.
    # Scoring these the same way would classify every correct value as a
    # fabrication and make the arm look catastrophically worse than it is.
    context_free = bool(row.get("context_free")) or not hits

    substantiated, fabricated, missed = [], [], []
    for v in gold_values:
        in_answer = value_present(v, answer)
        if context_free:
            (substantiated if in_answer else missed).append(v)
            continue
        if in_answer and value_present(v, context):
            substantiated.append(v)
        elif in_answer:
            fabricated.append(v)
        else:
            missed.append(v)

    source = row.get("source_filename")
    paper_rank = next(
        (n for n, h in enumerate(hits, 1) if h.get("file") == source), None
    )

    # Did the retrieved set include a chunk that actually contains the gold
    # values? This is the known-item retrieval signal, and it is a lower bound on
    # retrieval failure -- not exhaustive recall over the corpus.
    gold_chunk_hit = any(
        h.get("file") == source
        and any(value_present(v, h.get("content", "")) for v in gold_values)
        for h in hits
    ) if gold_values else None

    n = len(gold_values)
    return {
        **{k: row[k] for k in ("id", "arm", "user_input", "evidence_type",
                               "source_filename", "stratum") if k in row},
        "context_free": context_free,
        "abstained": is_abstention(answer),
        # Stated numbers while none of the gold values were produced: the shape of
        # a confident wrong answer, as opposed to an abstention.
        "asserted_without_gold": bool(
            not substantiated and not is_abstention(answer)
            and re.search(r"\d[\d,]*\.?\d*\s*(?:%|kWh|GJ|EUR|€|\$|ha|years?|/yr)", answer)
        ),
        "n_gold_values": n,
        # For a context-free arm this list means "value stated in the answer",
        # which is correctness, not substantiation. Report the two separately.
        "substantiated": substantiated,
        "fabricated": fabricated,
        "missed": missed,
        "accuracy": round(len(substantiated) / n, 3) if n else None,
        "fabricated_any": bool(fabricated),
        "gold_paper_rank": None if context_free else paper_rank,
        "gold_paper_retrieved": None if context_free else paper_rank is not None,
        "gold_chunk_retrieved": None if context_free else gold_chunk_hit,
        "retrieved_types": sorted({h.get("type") for h in hits}),
        "answer": answer,
    }


def summarise(scored):
    n = len(scored)
    if not n:
        return {}
    acc = [s["accuracy"] for s in scored if s["accuracy"] is not None]
    context_free = all(s["context_free"] for s in scored)
    label = "values_correct" if context_free else "values_substantiated"
    out = {
        "questions": n,
        "context_free_arm": context_free,
        "mean_accuracy": round(sum(acc) / len(acc), 3) if acc else None,
        label: sum(len(s["substantiated"]) for s in scored),
        "values_missed": sum(len(s["missed"]) for s in scored),
        "questions_abstained": sum(s["abstained"] for s in scored),
        "questions_asserted_without_gold": sum(s["asserted_without_gold"] for s in scored),
    }
    if context_free:
        # Retrieval metrics are not applicable; reporting them as 0 would be a
        # different claim from reporting them as undefined.
        out["gold_paper_retrieved"] = "n/a"
        out["gold_chunk_retrieved"] = "n/a"
        out["values_fabricated"] = "n/a (no context to check against)"
    else:
        out["gold_paper_retrieved"] = sum(bool(s["gold_paper_retrieved"]) for s in scored)
        out["gold_chunk_retrieved"] = sum(bool(s["gold_chunk_retrieved"]) for s in scored)
        out["values_fabricated"] = sum(len(s["fabricated"]) for s in scored)
        out["questions_with_fabrication"] = sum(s["fabricated_any"] for s in scored)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="path to a runs/*.jsonl file")
    args = ap.parse_args()

    path = Path(args.run)
    if not path.is_absolute():
        path = RUNS / path.name
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    scored = [score_row(r) for r in rows]

    out = RESULTS / (path.stem + ".scored.jsonl")
    with out.open("w", encoding="utf-8") as fh:
        for s in scored:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    summary = summarise(scored)
    (RESULTS / (path.stem + ".summary.json")).write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print(f"scored {len(scored)} rows -> {out}")
    for k, v in summary.items():
        print(f"  {k:26} {v}")
    if isinstance(summary.get("values_fabricated"), int) and summary["values_fabricated"]:
        print("\n  FABRICATIONS (value asserted but absent from retrieved context):")
        for s in scored:
            if s["fabricated"]:
                print(f"    {s['id']}: {s['fabricated']}")


if __name__ == "__main__":
    main()
