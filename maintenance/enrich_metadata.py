"""
enrich_metadata.py — Add authors to papers_metadata.json via the CrossRef API.

Fetches author lists for every entry that has a DOI but no authors field.
Safe to re-run: already-enriched entries are skipped.

Usage:
    python enrich_metadata.py              # enrich all missing author fields
    python enrich_metadata.py --dry-run    # preview what would be updated
"""

import _bootstrap  # noqa: F401  puts the project root on sys.path
import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# Windows console defaults to cp1252; author names can contain Unicode.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from _bootstrap import PAPERS_METADATA as META_PATH
CROSSREF_BASE = "https://api.crossref.org/works/"
USER_AGENT = "AD-Algae-Chatbot/1.0 (mailto:pavan@wustl.edu)"
DELAY_S = 0.25  # polite rate — CrossRef asks for < 50 req/s


def fetch_authors(doi: str) -> str | None:
    """Return a formatted author string from CrossRef, or None on failure."""
    url = CROSSREF_BASE + urllib.parse.quote(doi, safe="")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        author_list = data.get("message", {}).get("author", [])
        if not author_list:
            return None
        names = []
        for a in author_list:
            family = a.get("family", "")
            given = a.get("given", "")
            if family and given:
                names.append(f"{family} {given[0]}.")
            elif family:
                names.append(family)
        if not names:
            return None
        if len(names) > 3:
            return ", ".join(names[:3]) + " et al."
        return ", ".join(names)
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}")
        return None
    except Exception as e:
        print(f"error: {e}")
        return None


def main(dry_run: bool = False) -> None:
    records = json.loads(META_PATH.read_text(encoding="utf-8"))

    needs_fetch = [r for r in records if r.get("doi") and not r.get("authors")]
    already_done = sum(1 for r in records if r.get("authors"))
    no_doi = sum(1 for r in records if not r.get("doi"))

    print(f"Total entries  : {len(records)}")
    print(f"Already have authors: {already_done}")
    print(f"No DOI (skipped)    : {no_doi}")
    print(f"To fetch             : {len(needs_fetch)}")

    if dry_run:
        print("\n[dry-run] No changes written.")
        return

    updated = failed = 0
    for i, r in enumerate(needs_fetch, 1):
        print(f"[{i}/{len(needs_fetch)}] {r['doi'][:60]} ... ", end="", flush=True)
        authors = fetch_authors(r["doi"])
        if authors:
            r["authors"] = authors
            print(authors)
            updated += 1
        else:
            print("not found")
            failed += 1
        # Save incrementally every 10 entries so progress survives crashes.
        if i % 10 == 0:
            META_PATH.write_text(
                json.dumps(records, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        time.sleep(DELAY_S)

    META_PATH.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nDone. {updated} updated, {failed} failed → {META_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Preview only, no writes.")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
