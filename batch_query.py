"""
batch_query.py — Run all permutations of research questions and save to output/.

Initializes RAGAnything once, then loops through all queries.
"""

import asyncio
import logging
import textwrap
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from lightrag import QueryParam
from raganything import RAGAnything

from config import RAG_CONFIG, DEFAULT_TOP_K, DEFAULT_SEARCH_MODE
from models import llm_model_func, embedding_func, vision_model_func

EXCERPT_LEN = 300
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Question templates (use {process} as placeholder) ---
TEMPLATES = [
    "Give me methane yield improvement with {process} combined with anaerobic digestion",
    "Give me VFA yield improvement with {process} combined with anaerobic digestion",
    "Give me acetate yield improvement with {process} combined with anaerobic digestion",
    "Give me bioproduct yield improvement with {process} combined with anaerobic digestion",
    "Give me technoeconomic improvement in anaerobic digestion when combined with {process}",
    "Give me economic improvement in anaerobic digestion when combined with {process}",
]

PROCESSES = [
    "algal process",
    "algal biochar",
    "photosynthetic biocathode",
    "bio-electrochemical systems",
    "algal biogas upgrade",
    "algal CO2 capture",
    "photobioreactor",
    "HRAP",
    "high rate algal pond",
    "biochar electrode",
]


def format_sources(sources: dict) -> str:
    chunks = sources.get("data", {}).get("chunks", [])
    if not chunks:
        return ""
    by_file = defaultdict(list)
    for chunk in chunks:
        fname = (chunk.get("file_path") or "").split("/")[-1]
        if fname:
            by_file[fname].append(chunk.get("content", "").strip())
    lines = ["\n**Sources:**"]
    for i, (fname, contents) in enumerate(by_file.items(), 1):
        lines.append(f"\n  [{i}] {fname}")
        for content in contents:
            excerpt = " ".join(content.split())[:EXCERPT_LEN]
            if len(content) > EXCERPT_LEN:
                excerpt += "..."
            for line in textwrap.wrap(excerpt, width=80, initial_indent="      ", subsequent_indent="      "):
                lines.append(line)
    return "\n".join(lines)


async def main():
    logging.basicConfig(level=logging.WARNING)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"batch_results_{timestamp}.md"

    rag = RAGAnything(
        config=RAG_CONFIG,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        vision_model_func=vision_model_func,
    )
    await rag._ensure_lightrag_initialized()

    queries = [
        (t.format(process=p), t, p)
        for t in TEMPLATES
        for p in PROCESSES
    ]
    total = len(queries)
    print(f"Running {total} queries -> {out_path}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# AD/Algae Research Batch Query Results\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total queries: {total}\n\n")
        f.write("---\n\n")

        for i, (query, template, process) in enumerate(queries, 1):
            print(f"[{i}/{total}] {query}")
            f.write(f"## Q{i}: {query}\n\n")
            f.flush()

            try:
                param = QueryParam(mode=DEFAULT_SEARCH_MODE, top_k=DEFAULT_TOP_K)
                answer = await rag.aquery(query=query, mode=DEFAULT_SEARCH_MODE, vlm_enhanced=False)
                sources = await rag.lightrag.aquery_data(query, param=param)
                f.write(answer)
                f.write("\n")
                f.write(format_sources(sources))
                f.write("\n\n---\n\n")
                print(f"  Done.")
            except Exception as e:
                f.write(f"[Error: {e}]\n\n---\n\n")
                print(f"  Error: {e}")

    await rag.finalize_storages()
    print(f"\nAll done. Results saved to: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
