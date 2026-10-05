"""Correct seven papers that shared one wrong DOI, and refresh their metadata.

Seven distinct papers all carried `10.2139/ssrn.4417034` and the "title"
"Contents Lists Available at Sciencedirect". That is `expand_metadata.py` failing
to extract a title, then matching CrossRef against a page header and landing on
an unrelated SSRN preprint. A wrong DOI is worse than a blank one: it points a
reader at the wrong paper.

Each replacement was established from the Elsevier PII in the filename and
confirmed through CrossRef's `filter=alternative-id:<PII>`, which returns the
record Elsevier registered that PII against. That is a stronger check than a
bibliographic title search, because it matches on the publisher's own identifier
rather than on text similarity.

Titles, authors and years are overwritten here (not just filled) because the
existing values are the same header artefact.

Usage:
    python fix_bad_dois.py --dry-run
    python fix_bad_dois.py
"""
import _bootstrap  # noqa: F401  puts the project root on sys.path
import argparse
import json
import shutil
import time
import urllib.parse
import urllib.request
from pathlib import Path

from _bootstrap import PAPERS_METADATA as METADATA
UA = {"User-Agent": "ad-algae-chatbot/1.0 (mailto:tanglab462@gmail.com)"}
WRONG_DOI = "10.2139/ssrn.4417034"
BAD_TITLE = "contents lists available at sciencedirect"

CORRECTED = {
    "1-s2.0-S0960852417304339-main.pdf": "10.1016/j.biortech.2017.03.151",
    "1-s2.0-S0960852424017048-main (1).pdf": "10.1016/j.biortech.2024.132000",
    "1-s2.0-S0960852425005577-main.pdf": "10.1016/j.biortech.2025.132591",
    "1-s2.0-S0960852425014890-main.pdf": "10.1016/j.biortech.2025.133522",
    "1-s2.0-S0960852426001458-main.pdf": "10.1016/j.biortech.2026.134064",
    "1-s2.0-S2211926425004011-main.pdf": "10.1016/j.algal.2025.104290",
    "1-s2.0-S2589014X26000861-main.pdf": "10.1016/j.biteb.2026.102628",
}


def crossref(doi):
    try:
        req = urllib.request.Request(
            "https://api.crossref.org/works/" + urllib.parse.quote(doi), headers=UA)
        return json.load(urllib.request.urlopen(req, timeout=30))["message"]
    except Exception:
        return None


def summarise(msg):
    title = (msg.get("title") or [""])[0]
    authors = msg.get("author") or []
    names = [f"{a.get('family', '')} {(a.get('given') or '')[:1]}.".strip() for a in authors[:3]]
    joined = ", ".join(n for n in names if n)
    if len(authors) > 3:
        joined += " et al."
    year = ""
    for key in ("published-print", "published-online", "issued", "created"):
        parts = (msg.get(key) or {}).get("date-parts")
        if parts and parts[0] and parts[0][0]:
            year = str(parts[0][0])
            break
    return title, joined, year


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    meta = json.loads(METADATA.read_text(encoding="utf-8"))
    by_file = {m["filename"]: m for m in meta}

    for fn, doi in CORRECTED.items():
        rec = by_file.get(fn)
        if rec is None:
            print(f"  SKIP  {fn}  (not in metadata)")
            continue
        old = (rec.get("doi") or "").strip()
        if old and old.lower() != WRONG_DOI:
            print(f"  SKIP  {fn}  (has {old}, not the known-bad DOI)")
            continue

        msg = crossref(doi)
        rec["doi"] = doi
        if msg:
            title, authors, year = summarise(msg)
            if title:
                rec["title"] = title
            if authors:
                rec["authors"] = authors
            if year:
                rec["year"] = year
        print(f"  FIX   {fn[:42]:44} {old or '(blank)'} -> {doi}")
        if msg:
            print(f"        {rec.get('year')}  {rec.get('title', '')[:68]}")
        time.sleep(0.4)

    left = [m["filename"] for m in meta
            if (m.get("doi") or "").lower() == WRONG_DOI
            or BAD_TITLE in (m.get("title") or "").lower()]
    print(f"\nrecords still carrying the bad DOI or header-as-title: {len(left)}")
    for f in left:
        print(f"    {f}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return
    shutil.copy(METADATA, METADATA.with_suffix(".json.pre-doi-fix"))
    METADATA.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {METADATA}")


if __name__ == "__main__":
    main()
