"""
expand_metadata.py — Find DOIs and metadata for papers in rag_storage
that are missing from papers_metadata.json.

Two-pass strategy per paper:
  1. Extract DOI directly from chunk text (MinerU often captures it from headers/footers)
  2. If no embedded DOI, extract a candidate title and search CrossRef by title

Safe to re-run: already-registered filenames are skipped.

Usage:
    python expand_metadata.py              # process all missing papers
    python expand_metadata.py --dry-run    # preview what's missing, no writes
"""

import _bootstrap  # noqa: F401  puts the project root on sys.path
import argparse
import difflib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from _bootstrap import PAPERS_METADATA as META_PATH
from _bootstrap import RAG_STORAGE
CHUNKS_PATH = RAG_STORAGE / "kv_store_text_chunks.json"
CROSSREF_BASE = "https://api.crossref.org/works"
USER_AGENT = "AD-Algae-Chatbot/1.0 (mailto:pavan@wustl.edu)"
DELAY_S = 0.3
TITLE_MATCH_THRESHOLD = 0.72  # minimum similarity to accept a CrossRef title match

# Matches a DOI at word boundary; strips trailing punctuation afterwards.
_DOI_RE = re.compile(r"\b(10\.\d{4,}/\S+)", re.IGNORECASE)
_TRAILING_PUNCT = re.compile(r"[.,;:)\]>\"']+$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_doi(doi: str) -> str:
    return _TRAILING_PUNCT.sub("", doi).strip()


def extract_doi_from_chunks(chunks: list[dict]) -> str | None:
    """Return the most-frequently-occurring DOI found in raw chunk text."""
    counts: Counter = Counter()
    for chunk in chunks:
        for raw in _DOI_RE.findall(chunk.get("content", "")):
            counts[_clean_doi(raw)] += 1
    if counts:
        return counts.most_common(1)[0][0]
    return None


def extract_title_from_chunks(chunks: list[dict]) -> str | None:
    """
    Best-effort title extraction from the first ~30 chunks.
    Looks for MinerU header blocks first, then falls back to plain text lines.
    """
    # Pass 1: header blocks that look like titles
    for chunk in chunks[:30]:
        content = chunk.get("content", "")
        if "Header Content Analysis" in content:
            m = re.search(r"'text':\s*'([^']{20,})'", content)
            if m:
                text = m.group(1).strip()
                # Skip journal volume lines like "Bioresource Technology 391 (2024)"
                if not re.match(r"^[A-Za-z]+ (Technology|Journal|Letters|Research|Science|Review)", text):
                    if not re.search(r"^\d", text) and len(text) < 250:
                        return text

    # Pass 2: first meaningful plain-text line in early chunks
    for chunk in chunks[:6]:
        content = chunk.get("content", "").strip()
        for line in content.splitlines():
            line = line.strip()
            if 25 < len(line) < 220 and not line[0].isdigit() and line.count(" ") > 3:
                return line

    return None


def _crossref_get(url: str) -> dict | None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code != 404:
            print(f"  HTTP {e.code}")
        return None
    except Exception as e:
        print(f"  request error: {e}")
        return None


def search_crossref_by_title(title: str) -> tuple[str, str, float] | None:
    """Search CrossRef by title. Returns (doi, matched_title, score) or None."""
    params = urllib.parse.urlencode({
        "query.title": title,
        "rows": "5",
        "select": "DOI,title,author,published",
    })
    data = _crossref_get(f"{CROSSREF_BASE}?{params}")
    if not data:
        return None
    items = data.get("message", {}).get("items", [])
    title_lower = title.lower()
    best_item, best_score = None, 0.0
    for item in items:
        for t in item.get("title", []):
            score = difflib.SequenceMatcher(None, title_lower, t.lower()).ratio()
            if score > best_score:
                best_score, best_item = score, item
    if best_item and best_score >= TITLE_MATCH_THRESHOLD:
        return best_item["DOI"], (best_item.get("title") or [""])[0], best_score
    return None


def fetch_full_metadata(doi: str) -> dict:
    """Fetch title, authors, year from CrossRef for a known DOI."""
    data = _crossref_get(f"{CROSSREF_BASE}/{urllib.parse.quote(doi, safe='')}")
    if not data:
        return {}
    msg = data.get("message", {})

    titles = msg.get("title", [])
    title = titles[0] if titles else ""

    raw_authors = msg.get("author", [])
    names = []
    for a in raw_authors:
        family = a.get("family", "")
        given = a.get("given", "")
        if family and given:
            names.append(f"{family} {given[0]}.")
        elif family:
            names.append(family)
    if len(names) > 3:
        authors = ", ".join(names[:3]) + " et al."
    elif names:
        authors = ", ".join(names)
    else:
        authors = ""

    date_parts = msg.get("published", {}).get("date-parts", [[""]])[0]
    year = str(date_parts[0]) if date_parts else ""

    return {"title": title, "authors": authors, "year": year}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(dry_run: bool = False) -> None:
    meta_records: list[dict] = json.loads(META_PATH.read_text(encoding="utf-8"))
    existing = {r["filename"] for r in meta_records}

    print("Loading chunks (this may take a moment)...")
    all_chunks: dict = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

    # Group chunks by paper filename, sorted by chunk position for title extraction
    by_paper: dict[str, list[dict]] = {}
    for chunk in all_chunks.values():
        fp = chunk.get("file_path", "")
        fname = fp.replace("\\", "/").split("/")[-1]
        if fname:
            by_paper.setdefault(fname, []).append(chunk)

    missing = sorted(set(by_paper.keys()) - existing)

    print(f"Papers in index   : {len(by_paper)}")
    print(f"In metadata       : {len(existing)}")
    print(f"Missing           : {len(missing)}")

    if dry_run:
        print("\n[dry-run] First 15 missing filenames:")
        for f in missing[:15]:
            print(f"  {f}")
        return

    found_doi = 0
    no_doi = 0

    for i, fname in enumerate(missing, 1):
        print(f"\n[{i}/{len(missing)}] {fname}")
        chunks = by_paper[fname]
        entry: dict = {"filename": fname, "title": "", "doi": None, "year": "", "authors": ""}

        # --- Strategy 1: DOI embedded in chunk content ---
        doi = extract_doi_from_chunks(chunks)
        if doi:
            print(f"  DOI in content: {doi}")
            meta = fetch_full_metadata(doi)
            time.sleep(DELAY_S)
            if meta.get("title"):
                entry.update({"doi": doi, **meta})
                print(f"  title : {meta['title'][:70]}")
                print(f"  year  : {meta.get('year')} | authors: {meta.get('authors','')[:50]}")
                meta_records.append(entry)
                found_doi += 1
                if found_doi % 10 == 0:
                    META_PATH.write_text(json.dumps(meta_records, indent=2, ensure_ascii=False), encoding="utf-8")
                continue

        # --- Strategy 2: title extraction → CrossRef search ---
        title = extract_title_from_chunks(chunks)
        if not title:
            title = fname.replace(".pdf", "").replace("_", " ").replace("-", " ")
        print(f"  title search: {title[:65]}")

        result = search_crossref_by_title(title)
        time.sleep(DELAY_S)

        if result:
            doi, matched_title, score = result
            print(f"  match ({score:.2f}): {matched_title[:65]}")
            meta = fetch_full_metadata(doi)
            time.sleep(DELAY_S)
            entry.update({"doi": doi, **meta})
            print(f"  DOI: {doi}")
            found_doi += 1
        else:
            print("  no confident match — saving filename only")
            no_doi += 1

        meta_records.append(entry)

        if (i % 10) == 0:
            META_PATH.write_text(json.dumps(meta_records, indent=2, ensure_ascii=False), encoding="utf-8")

    META_PATH.write_text(json.dumps(meta_records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone. {found_doi} DOIs found, {no_doi} unmatched → {META_PATH}")
    print(f"Total metadata entries: {len(meta_records)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
