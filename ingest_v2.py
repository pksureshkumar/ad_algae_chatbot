"""
ingest_v2.py — Ingest all PDFs using local Ollama models into rag_storage_v2/.
Can run simultaneously alongside ingest.py (Azure) since they use separate directories.

Usage:
    python ingest_v2.py
    python ingest_v2.py --test     # first 2 PDFs only
    python ingest_v2.py --reset    # clear progress and start over
"""

import asyncio
import json
import logging
import argparse
from pathlib import Path

from raganything import RAGAnything

from config_v2 import RAG_CONFIG_V2, PAPERS_DIR, RAG_STORAGE_DIR_V2
from models_v2 import llm_model_func_v2, embedding_func_v2, vision_model_func_v2

PROGRESS_FILE = RAG_STORAGE_DIR_V2 / "ingested_files.json"


def setup_logging():
    RAG_STORAGE_DIR_V2.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(RAG_STORAGE_DIR_V2 / "ingest_v2.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def load_progress() -> set:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return set(json.load(f))
    return set()


def save_progress(ingested: set):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(sorted(ingested), f, indent=2)


async def main(test: bool = False, reset: bool = False):
    setup_logging()
    logger = logging.getLogger(__name__)

    RAG_STORAGE_DIR_V2.mkdir(parents=True, exist_ok=True)

    if reset:
        PROGRESS_FILE.unlink(missing_ok=True)
        logger.info("Progress file cleared — starting from scratch.")

    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))
    if not pdfs:
        logger.error(f"No PDFs found in {PAPERS_DIR}. Aborting.")
        return

    if test:
        pdfs = pdfs[:2]
        logger.info(f"Test mode: limiting to first {len(pdfs)} PDFs.")

    logger.info(f"Total PDFs found: {len(pdfs)}")

    ingested = load_progress()
    remaining = [p for p in pdfs if str(p) not in ingested]
    logger.info(f"Already ingested: {len(ingested)} | Remaining: {len(remaining)}")

    if not remaining:
        logger.info("Nothing to do — all PDFs are already in the knowledge base.")
        return

    rag = RAGAnything(
        config=RAG_CONFIG_V2,
        llm_model_func=llm_model_func_v2,
        embedding_func=embedding_func_v2,
        vision_model_func=vision_model_func_v2,
    )

    failed = []
    for i, pdf_path in enumerate(remaining, 1):
        logger.info(f"[{i}/{len(remaining)}] {pdf_path.name}")
        try:
            await rag.process_document_complete(file_path=str(pdf_path))
            ingested.add(str(pdf_path))
            save_progress(ingested)
            logger.info(f"  Done: {pdf_path.name}")
        except Exception as e:
            logger.error(f"  Failed: {pdf_path.name} -- {e}")
            failed.append(pdf_path.name)

    await rag.finalize_storages()

    logger.info("=" * 60)
    logger.info(f"Ingestion complete: {len(ingested)}/{len(pdfs)} files indexed.")
    if failed:
        logger.warning(f"Failed ({len(failed)}): {', '.join(failed)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs using local Ollama models.")
    parser.add_argument("--test", action="store_true",
                        help="Process only the first 2 PDFs (for testing).")
    parser.add_argument("--reset", action="store_true",
                        help="Reset ingestion progress and start over.")
    args = parser.parse_args()
    asyncio.run(main(test=args.test, reset=args.reset))
