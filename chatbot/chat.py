"""
chat.py — Interactive chatbot for the AD/Algae research knowledge base.

Usage:
    python chat.py

In-session commands:
    :mode hybrid|local|global   Switch retrieval mode
    :topk <n>                   Change number of chunks retrieved
    quit / exit                 Exit
"""

import _bootstrap  # noqa: F401  puts the repository root and core/ on sys.path
import fast_storage  # must be first — patches NanoVectorDB before LightRAG loads  # noqa: F401

import asyncio
import logging
import textwrap

from lightrag import QueryParam
from raganything import RAGAnything

from citations import build_citation_context, extract_unique_filenames
from config import (RAG_CONFIG, RAG_STORAGE_DIR, DEFAULT_TOP_K, DEFAULT_SEARCH_MODE,
                    LIGHTRAG_KWARGS)
from models import llm_model_func, embedding_func, vision_model_func

EXCERPT_LEN = 300


def check_index():
    if not RAG_STORAGE_DIR.exists() or not any(RAG_STORAGE_DIR.iterdir()):
        print(
            "\n[WARNING] rag_storage/ is empty or missing.\n"
            "Run `python ingest.py` first to build the knowledge base.\n"
        )


def format_references(refs: list[dict], chunks: list[dict]) -> str:
    if not refs:
        return ""

    # Build excerpt map
    excerpt_map: dict[str, list[str]] = {r["filename"]: [] for r in refs}
    for chunk in chunks:
        raw = chunk.get("file_path") or ""
        fname = raw.replace("\\", "/").split("/")[-1]
        if fname in excerpt_map:
            content = chunk.get("content", "").strip()
            excerpt = " ".join(content.split())[:EXCERPT_LEN]
            if len(content) > EXCERPT_LEN:
                excerpt += "..."
            excerpt_map[fname].append(excerpt)

    lines = ["\nReferences:"]
    for r in refs:
        # Citation line
        if r["doi"]:
            cite = f"  [{r['ref_num']}] {r['title']} ({r['year']})\n        https://doi.org/{r['doi']}"
        else:
            parts = [r["title"]]
            if r["authors"]:
                parts.append(r["authors"])
            if r["year"]:
                parts.append(r["year"])
            cite = f"  [{r['ref_num']}] " + " — ".join(parts)
        if r["authors"] and r["doi"]:
            cite += f"\n        {r['authors']}"
        lines.append(cite)

        # Excerpts
        for content in excerpt_map.get(r["filename"], []):
            for line in textwrap.wrap(content, width=76, initial_indent="        ", subsequent_indent="        "):
                lines.append(line)

    return "\n".join(lines)


async def main():
    logging.basicConfig(level=logging.WARNING)
    check_index()

    print("\n" + "=" * 62)
    print("  Anaerobic Digestion & Algae Research Chatbot")
    print("  Knowledge base: 274 peer-reviewed papers")
    print("  Powered by RAG-Anything + qwen3:14b + qwen2.5vl:32b")
    print("=" * 62)
    print("Type your question and press Enter.")
    print("Commands: :mode hybrid|local|global  |  :topk <n>  |  quit\n")

    rag = RAGAnything(
        config=RAG_CONFIG,
        lightrag_kwargs=LIGHTRAG_KWARGS,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        vision_model_func=vision_model_func,
    )
    print("Loading knowledge base into memory (this may take a few minutes)...")
    await rag._ensure_lightrag_initialized()
    print("✓ Ready — knowledge base loaded.\n")

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

        if user_input.startswith(":mode "):
            mode = user_input.split(None, 1)[1].strip()
            if mode in ("hybrid", "local", "global"):
                search_mode = mode
                print(f"[Search mode → {search_mode}]")
            else:
                print("[Invalid mode. Choose: hybrid, local, global]")
            continue

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
            sources_data = await rag.lightrag.aquery_data(user_input, param=param)

            chunks = (sources_data or {}).get("data", {}).get("chunks", [])
            filenames = extract_unique_filenames(chunks)
            instruction, refs = build_citation_context(filenames)

            augmented = f"{user_input}\n\n{instruction}" if instruction else user_input
            answer = await rag.aquery(query=augmented, mode=search_mode, vlm_enhanced=False)

            print(answer)
            print(format_references(refs, chunks))
        except Exception as e:
            print(f"[Error: {e}]")

        print()

    await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
