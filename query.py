"""
query.py — Single-shot CLI query against the AD/Algae knowledge base.

Usage:
    python query.py "What are the main benefits of co-digesting algae with AD?"
    python query.py "Compare methane yields in different algae-AD studies" --mode global
    python query.py "..." --mode local --top-k 15
"""

import sys
import asyncio
import logging
import argparse
import textwrap
from collections import defaultdict

from lightrag import QueryParam
from raganything import RAGAnything

from config import RAG_CONFIG, DEFAULT_TOP_K, DEFAULT_SEARCH_MODE
from models import llm_model_func, embedding_func, vision_model_func

EXCERPT_LEN = 300  # characters shown per chunk excerpt


def format_sources(sources: dict) -> str:
    """Format retrieved chunks grouped by source file with short excerpts."""
    chunks = sources.get("data", {}).get("chunks", [])
    if not chunks:
        return ""

    # Group chunks by filename, preserving first-seen order.
    by_file = defaultdict(list)
    for chunk in chunks:
        fname = (chunk.get("file_path") or "").split("/")[-1]
        if fname:
            by_file[fname].append(chunk.get("content", "").strip())

    lines = ["\nSources:"]
    for i, (fname, contents) in enumerate(by_file.items(), 1):
        lines.append(f"\n  [{i}] {fname}")
        for content in contents:
            excerpt = " ".join(content.split())[:EXCERPT_LEN]
            if len(content) > EXCERPT_LEN:
                excerpt += "..."
            for line in textwrap.wrap(excerpt, width=76, initial_indent="      ", subsequent_indent="      "):
                lines.append(line)
    return "\n".join(lines)


async def main(query: str, search_mode: str, top_k: int):
    logging.basicConfig(level=logging.WARNING)

    rag = RAGAnything(
        config=RAG_CONFIG,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        vision_model_func=vision_model_func,
    )

    param = QueryParam(mode=search_mode, top_k=top_k)

    answer, sources = await asyncio.gather(
        rag.aquery(query=query, top_k=top_k, search_mode=search_mode),
        rag.lightrag.aquery_data(query, param=param),
    )

    print(answer)
    print(format_sources(sources))

    await rag.finalize_storages()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query the AD/Algae research knowledge base."
    )
    parser.add_argument("query", help="Your research question.")
    parser.add_argument(
        "--mode",
        choices=["hybrid", "local", "global"],
        default=DEFAULT_SEARCH_MODE,
        help=f"Retrieval mode (default: {DEFAULT_SEARCH_MODE}).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        dest="top_k",
        help=f"Number of chunks to retrieve (default: {DEFAULT_TOP_K}).",
    )
    args = parser.parse_args()

    if not args.query.strip():
        print("Error: query cannot be empty.", file=sys.stderr)
        sys.exit(1)

    asyncio.run(main(args.query, args.mode, args.top_k))
