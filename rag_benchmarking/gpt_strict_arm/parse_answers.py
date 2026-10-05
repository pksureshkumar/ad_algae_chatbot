"""Convert a collected answers file into runs/<arm>.jsonl for scoring.

Usage:
    python parse_answers.py --arm gpt_websearch --answers answers_websearch.md \
        --model "GPT-6 Astra (web search enabled)"
    python parse_answers.py --arm gpt_strict --answers answers_nosearch.md \
        --model "GPT-6 Astra"
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
from paths import DATASET, RUNS  # noqa: E402

# Resolve relative to this file so the script keeps working if the
# containing folder is renamed (gpt_arm -> gpt_websearch_arm).
HERE = Path(__file__).resolve().parent

# Web-search contamination gate. The first collection pass ran with search
# enabled without anyone noticing: 15 of 16 answers carried live URLs, including
# a "?error=cookies_not_supported" query string that only a browser produces.
# That arm scored 37/53 while the local ungrounded models scored 1/53 and 2/53 --
# not a capability difference, but retrieval. Labelling such a run as "no
# retrieval" would repeat the flaw that weakened the v1 Perplexity comparison.
WEB_MARKERS = re.compile(r"https?://|\bdoi\.org/|cookies_not_supported", re.I)

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="gpt_strict")
ap.add_argument("--answers", default="answers.md",
                help="filename inside gpt_arm/ to read")
ap.add_argument("--model", default="", help="model name shown in the interface")
args = ap.parse_args()

text = (HERE / args.answers).read_text(encoding="utf-8")
answers = {}
for qid, body in re.findall(r"^## (Q\d+)\b.*?\n+```(.*?)```", text, re.S | re.M):
    if body.strip():
        answers[qid] = body.strip()

rows = [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()]
out, missing = [], []
for r in rows:
    a = answers.get(r["id"], "")
    if not a:
        missing.append(r["id"])
        continue
    out.append({**r, "arm": args.arm, "top_k": None, "context_free": True,
                "model": args.model or "unknown (web interface)",
                "answer": a, "retrieved": [], "seconds": None})

path = RUNS / f"{args.arm}.jsonl"
with path.open("w", encoding="utf-8") as fh:
    for r in out:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

print("wrote %s  (%d/%d questions)" % (path, len(out), len(rows)))
if missing:
    print("  still empty: " + ", ".join(missing))

flagged = [r["id"] for r in out if WEB_MARKERS.search(r["answer"])]
if flagged:
    print("")
    print("  WARNING: %d/%d answers contain URLs or DOI links:" % (len(flagged), len(out)))
    print("    " + ", ".join(flagged))
    print("  These look web-retrieved. If search was meant to be OFF this run is")
    print("  not an ungrounded baseline -- relabel it (--arm gpt_websearch) or")
    print("  recollect with browsing disabled.")
else:
    print("  no URLs found - consistent with search being off")

if not args.model:
    print("  NOTE: no --model given; record the interface model name for the methods")
