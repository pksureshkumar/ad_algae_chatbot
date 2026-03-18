"""
chat.py — Interactive chatbot for the AD/Algae research knowledge base.

Usage:
    python chat.py

In-session commands:
    :mode hybrid|local|global   Switch retrieval mode
    :topk <n>                   Change number of chunks retrieved
    quit / exit                 Exit
"""

import asyncio
import logging
import textwrap
from collections import defaultdict

from lightrag import QueryParam
from raganything import RAGAnything

from config import RAG_CONFIG, RAG_STORAGE_DIR, DEFAULT_TOP_K, DEFAULT_SEARCH_MODE
from models import llm_model_func, embedding_func, vision_model_func

EXCERPT_LEN = 300  # characters shown per chunk excerpt


def check_index():
    """Warn the user if the knowledge base has not been built yet."""
    if not RAG_STORAGE_DIR.exists() or not any(RAG_STORAGE_DIR.iterdir()):
        print(
            "\n[WARNING] rag_storage/ is empty or missing.\n"
            "Run `python ingest.py` first to build the knowledge base.\n"
        )


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


async def main():
    # Suppress noisy library logs during interactive use.
    logging.basicConfig(level=logging.WARNING)

    check_index()

    print("\n" + "=" * 62)
    print("  Anaerobic Digestion & Algae Research Chatbot")
    print("  Knowledge base: ~141 peer-reviewed papers")
    print("  Powered by RAG-Anything + GPT-4o")
    print("=" * 62)
    print("Type your question and press Enter.")
    print("Commands: :mode hybrid|local|global  |  :topk <n>  |  quit\n")

    rag = RAGAnything(
        config=RAG_CONFIG,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        vision_model_func=vision_model_func,
    )

    search_mode = DEFAULT_SEARCH_MODE
    top_k = DEFAULT_TOP_K

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # In-session command: change search mode
        if user_input.startswith(":mode "):
            mode = user_input.split(None, 1)[1].strip()
            if mode in ("hybrid", "local", "global"):
                search_mode = mode
                print(f"[Search mode → {search_mode}]")
            else:
                print("[Invalid mode. Choose: hybrid, local, global]")
            continue

        # In-session command: change top-k
        if user_input.startswith(":topk "):
            try:
                top_k = int(user_input.split(None, 1)[1].strip())
                print(f"[Top-k → {top_k}]")
            except ValueError:
                print("[Invalid value. Usage: :topk 15]")
            continue

        print("\nAssistant: ", end="", flush=True)
        try:
            param = QueryParam(mode=search_mode, top_k=top_k)
            answer, sources = await asyncio.gather(
                rag.aquery(query=user_input, top_k=top_k, search_mode=search_mode),
                rag.lightrag.aquery_data(user_input, param=param),
            )
            print(answer)
            print(format_sources(sources))
        except Exception as e:
            print(f"[Error: {e}]")

        print()

    await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
