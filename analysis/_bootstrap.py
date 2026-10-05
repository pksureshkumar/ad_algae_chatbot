"""Make the project root and core/ importable.

The five modules in core/ import each other by bare name (`import config`,
`from models import ...`). Rather than rewrite every one of those imports, this
puts both the repository root and core/ on sys.path, so those statements keep
working from any folder.

Import it first, before any project import:

    import _bootstrap  # noqa: F401

An intentional near-duplicate of this file sits in each top-level folder that
holds runnable scripts. It has to, because it is what makes imports work at all -
it cannot itself be imported from a shared location.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

for _p in (ROOT, ROOT / "core"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

RAG_STORAGE = ROOT / "rag_storage"
PAPERS_METADATA = ROOT / "papers_metadata.json"
PAPERS_DIR = ROOT / "papers"
OUTPUT_DIR = ROOT / "output"
STATIC_DIR = ROOT / "chatbot" / "static"
