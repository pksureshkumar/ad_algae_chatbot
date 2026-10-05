"""Fill in DOIs for the 13 indexed papers that had none.

Those papers were in the corpus but not citable: the published DOI list is how a
reader obtains the sources, so a gap there is a gap in the data-availability
statement.

How each DOI was established, and why that is trustworthy:

* Encoded in the filename, corroborated by the paper's own text
  (10.22034/gjesm.2025.04.05) - the only one CrossRef does not hold.
* Springer filenames carrying the DOI suffix (s11356-*, s002849900267),
  confirmed by retrieving the record and matching the title.
* Elsevier PII filenames (1-s2.0-S*): the candidate DOI was taken from the
  paper's own text or a CrossRef title search, then confirmed by checking that
  the PII appears in CrossRef's `alternative-id` for that DOI. Scanning chunk
  text alone is not sufficient, because a paper's bibliography contains other
  papers' DOIs - the PII check is what distinguishes the two.
* Two remaining papers were matched on an exact title comparison against
  CrossRef using their opening text.

Usage:
    python backfill_missing_dois.py --dry-run
    python backfill_missing_dois.py
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

RESOLVED = {
    "1-s2.0-S0032959201002308-main.pdf": "10.1016/s0032-9592(01)00230-8",
    "1-s2.0-S004313540100522X-main.pdf": "10.1016/s0043-1354(01)00522-x",
    "1-s2.0-S096014812500758X-main.pdf": "10.1016/j.renene.2025.123096",
    "1-s2.0-S1385894724077404-main.pdf": "10.1016/j.cej.2024.156249",
    "1-s2.0-S1385894725097578-main.pdf": "10.1016/j.cej.2025.168915",
    "1-s2.0-S2213343725038710-main.pdf": "10.1016/j.jece.2025.119175",
    "1-s2.0-S2666351122000158-main.pdf": "10.1016/j.sintl.2022.100170",
    "10.22034_gjesm.2025.04.05.pdf": "10.22034/gjesm.2025.04.05",
    "enhancing_methane_food_waste_fermentate.pdf": "10.1186/s13068-017-0994-7",
    "exploring_sustainable_pathways_wastewater.pdf": "10.21608/ijisd.2024.298056.1066",
    "s002849900267.pdf": "10.1007/s002849900267",
    "s11356-016-6224-1.pdf": "10.1007/s11356-016-6224-1",
    "s11356-018-1989-z.pdf": "10.1007/s11356-018-1989-z",
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

    changed = 0
    for fn, doi in RESOLVED.items():
        rec = by_file.get(fn)
        if rec is None:
            print(f"  SKIP  {fn}  (not in metadata)")
            continue
        if (rec.get("doi") or "").strip():
            print(f"  SKIP  {fn}  (already has {rec['doi']})")
            continue

        msg = crossref(doi)
        title, authors, year = summarise(msg) if msg else ("", "", "")
        rec["doi"] = doi
        if title and not (rec.get("title") or "").strip():
            rec["title"] = title
        if authors and not (rec.get("authors") or "").strip():
            rec["authors"] = authors
        if year and not (rec.get("year") or "").strip():
            rec["year"] = year
        changed += 1
        note = "" if msg else "  (not in CrossRef; DOI from filename + paper text)"
        print(f"  SET   {fn[:44]:46} {doi}{note}")
        if title:
            print(f"        {year}  {title[:70]}")
        time.sleep(0.4)

    remaining = [m["filename"] for m in meta if not (m.get("doi") or "").strip()]
    print(f"\n{changed} records updated; {len(remaining)} still without a DOI")
    if remaining:
        for f in remaining:
            print(f"    {f}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return
    shutil.copy(METADATA, METADATA.with_suffix(".json.pre-doi-backfill"))
    METADATA.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {METADATA}")


if __name__ == "__main__":
    main()
