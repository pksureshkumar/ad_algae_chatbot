"""
citations.py — Paper metadata lookup and citation context for RAG responses.
"""

import json
from pathlib import Path

_META_PATH = Path(__file__).resolve().parents[1] / "papers_metadata.json"
_meta_by_filename: dict[str, dict] | None = None


def _load() -> dict[str, dict]:
    global _meta_by_filename
    if _meta_by_filename is None:
        try:
            records = json.loads(_META_PATH.read_text(encoding="utf-8"))
            _meta_by_filename = {r["filename"]: r for r in records}
        except Exception:
            _meta_by_filename = {}
    return _meta_by_filename


def lookup(filename: str) -> dict:
    return _load().get(filename, {})


def extract_unique_filenames(chunks: list[dict]) -> list[str]:
    """Return unique paper filenames from retrieved chunks, preserving relevance order."""
    seen: set[str] = set()
    result: list[str] = []
    for chunk in chunks:
        raw = chunk.get("file_path") or ""
        fname = raw.replace("\\", "/").split("/")[-1]
        if fname and fname not in seen:
            seen.add(fname)
            result.append(fname)
    return result


def build_citation_context(filenames: list[str]) -> tuple[str, list[dict]]:
    """
    Build (instruction_string, refs_list) from a list of source filenames.

    instruction_string is appended to the user query so the LLM can cite inline.
    refs_list is returned in the API response for the frontend to render.
    """
    refs: list[dict] = []
    for i, fname in enumerate(filenames, 1):
        meta = lookup(fname)
        refs.append({
            "ref_num": i,
            "filename": fname,
            "title": meta.get("title") or fname,
            "authors": meta.get("authors") or "",
            "year": meta.get("year") or "",
            "doi": meta.get("doi") or "",
        })

    if not refs:
        return "", []

    lines = [
        "Cite specific findings or data inline using [1], [2], etc. "
        "from these retrieved source papers:"
    ]
    for r in refs:
        if r["doi"]:
            line = f"[{r['ref_num']}] {r['title']} ({r['year']}) DOI:{r['doi']}"
        else:
            parts = [f"[{r['ref_num']}] {r['title']}"]
            if r["authors"]:
                parts.append(r["authors"])
            if r["year"]:
                parts.append(r["year"])
            line = " — ".join(parts)
        lines.append(line)

    return "\n".join(lines), refs
