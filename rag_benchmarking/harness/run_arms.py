"""Execute the local retrieval arms over the ground-truth dataset.

Writes one JSONL per arm to runs/, carrying the full retrieved context so that
score.py can check substantiation without re-querying anything. Results are
therefore reproducible from the run files alone.

The GPT arms are not run here: they need an API key and a pinned model snapshot,
and they are produced separately into runs/gpt_*.jsonl in the same shape.

Usage:
    python run_arms.py                        # both local arms, indexed questions
    python run_arms.py --arm multimodal
    python run_arms.py --top-k 15 --limit 3
    python run_arms.py --include-unindexed    # will score 0; for debugging only
"""
import argparse
import asyncio
import json
import time

from paths import DATASET, RUNS, add_root_to_path

add_root_to_path()
import _env  # noqa: F401  UTF-8 stdout, tiktoken cache, rerank default
from models import embedding_func, llm_model_func  # noqa: E402

from retrieval import (ARMS, Index, ANSWER_PROMPT, NO_CONTEXT_PROMPT,  # noqa: E402
                       NEUTRAL_SYSTEM, build_context)


async def answer_without_domain_prompt(question, model=None):
    """Query the LLM bypassing models.llm_model_func.

    llm_model_func unconditionally prepends config.DOMAIN_SYSTEM_PROMPT, which
    claims access to a corpus and asks for citations. That is correct for the
    retrieval arms and wrong for a context-free baseline, so this calls the same
    model and the same endpoint directly with a neutral system prompt. Model,
    temperature and endpoint are taken from config so the only difference
    between arms remains the presence of retrieved evidence.
    """
    import re
    import config
    from models import get_ollama_client

    client = get_ollama_client()
    resp = await client.chat.completions.create(
        model=model or config.OLLAMA_LLM_MODEL,
        messages=[{"role": "system", "content": NEUTRAL_SYSTEM},
                  {"role": "user", "content": NO_CONTEXT_PROMPT.format(question=question)}],
        temperature=0.1,
    )
    text = resp.choices[0].message.content or ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def load_dataset(include_unindexed):
    rows = [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not include_unindexed:
        rows = [r for r in rows if r.get("indexed")]
    return rows


async def run_arm(index, rows, arm, top_k):
    opts = dict(ARMS[arm])
    context_free = opts.pop("context_free", False)
    model_override = opts.pop("model", None)
    out_path = RUNS / f"{arm}.jsonl"
    results = []
    for i, row in enumerate(rows, 1):
        started = time.time()
        if context_free:
            hits = []
            answer = await answer_without_domain_prompt(row["user_input"], model_override)
        else:
            vector = (await embedding_func([row["user_input"]]))[0]
            hits = index.search(vector, top_k=top_k, **opts)
            answer = await llm_model_func(
                ANSWER_PROMPT.format(context=build_context(hits),
                                     question=row["user_input"])
            )
        rec = {
            **row,
            "arm": arm,
            "top_k": None if context_free else top_k,
            "context_free": context_free,
            "model": model_override or "qwen3:14b",
            "answer": answer,
            "retrieved": hits,
            "seconds": round(time.time() - started, 1),
        }
        results.append(rec)
        rank = next((n for n, h in enumerate(hits, 1)
                     if h["file"] == row.get("source_filename")), None)
        print(f"  [{i}/{len(rows)}] {row['id']} {arm:11} "
              f"paper_rank={rank} ({rec['seconds']}s)", flush=True)

    with out_path.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> {out_path}")
    return out_path


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS), help="default: both")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--include-unindexed", action="store_true")
    args = ap.parse_args()

    rows = load_dataset(args.include_unindexed)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("no runnable questions — has ingestion finished?")

    print(f"{len(rows)} questions, top_k={args.top_k}")
    index = Index()
    index.assert_retrievable(r.get("source_filename") for r in rows)
    for arm in ([args.arm] if args.arm else sorted(ARMS)):
        print(f"\n== {arm} ==")
        await run_arm(index, rows, arm, args.top_k)


if __name__ == "__main__":
    asyncio.run(main())
