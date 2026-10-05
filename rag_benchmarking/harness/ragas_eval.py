"""RAGAS metrics over the run files. RUNS IN ITS OWN VIRTUALENV.

    rag_benchmarking/.venv-ragas/Scripts/python.exe harness/ragas_eval.py --run runs/multimodal.jsonl

Do not install ragas into the `ad_algae` environment: that env holds a working
raganything + lightrag + MinerU stack with its own pins, and ragas pulls
langchain, instructor and pydantic. The two stages are decoupled by JSONL on disk
so they never share an interpreter.

Environment notes, both found the hard way:

* ragas 0.4.3 declares `langchain-community` with no upper bound, but its
  `llms/base.py` imports `langchain_community.chat_models.vertexai`, which was
  removed in 0.4.x. Installing plain `ragas==0.4.3` therefore fails at import.
  Pin `langchain-community<0.4` (resolved: 0.3.31).
* There is no "ollama" provider. Ollama serves an OpenAI-compatible API, so pass
  an OpenAI client pointed at it with provider="openai".

Judge model is deliberately NOT qwen3:14b. The system under test generates with
qwen3:14b, so using it as judge invites a self-preference objection that costs
nothing to avoid.

Metrics split by what they need:
    NonLLMContextPrecisionWithReference / NonLLMContextRecall
        deterministic, string-similarity based, need reference_contexts
    IDBasedContextPrecision / IDBasedContextRecall
        deterministic, compare retrieved vs reference paper ids; immune to the
        arms chunking differently
    Faithfulness / ResponseRelevancy
        LLM-judged, reference-free, run on every question
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
RUNS = BENCH / "runs"
RESULTS = BENCH / "results"

OLLAMA = "http://localhost:11434/v1"


def build(judge, embed_model, base_url):
    """Judge LLM plus a LEGACY-interface embeddings object.

    embedding_factory() returns the modern BaseRagasEmbedding, which exposes
    embed_text() but not embed_query(). ResponseRelevancy is a legacy metric and
    calls embed_query(), so it raises AttributeError on every sample -- producing
    no answer-relevancy column at all, silently. The collections.AnswerRelevancy
    class does take the modern interface, but evaluate() rejects it ("All metrics
    must be initialised metric objects"), because collections metrics derive from
    a different base class and cannot be mixed with legacy ones in one call.

    Wrapping a langchain OpenAIEmbeddings in LangchainEmbeddingsWrapper gives the
    legacy interface and keeps every metric on one code path.
    check_embedding_ctx_length=False is required: that check assumes OpenAI
    tokenisation and breaks against Ollama.
    """
    from openai import OpenAI
    from langchain_openai import OpenAIEmbeddings
    from ragas.llms import llm_factory
    from ragas.embeddings import LangchainEmbeddingsWrapper

    client = OpenAI(base_url=base_url, api_key="ollama")
    llm = llm_factory(judge, provider="openai", client=client)
    emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
        model=embed_model, base_url=base_url, api_key="ollama",
        check_embedding_ctx_length=False))
    return llm, emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--judge", default="gemma3:27b")
    ap.add_argument("--embed", default="bge-m3")
    ap.add_argument("--base-url", default=OLLAMA)
    ap.add_argument("--deterministic-only", action="store_true",
                    help="skip the LLM-judged metrics (no generation needed)")
    args = ap.parse_args()

    path = Path(args.run)
    if not path.is_absolute():
        path = BENCH / args.run if (BENCH / args.run).exists() else RUNS / path.name
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    from ragas import evaluate, EvaluationDataset
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import (
        Faithfulness,
        NonLLMContextPrecisionWithReference, NonLLMContextRecall,
        IDBasedContextPrecision, IDBasedContextRecall,
    )
    from ragas.metrics import ResponseRelevancy

    context_free = all(r.get("context_free") for r in rows)
    samples = [
        SingleTurnSample(
            user_input=r["user_input"],
            response=r.get("answer") or "",
            retrieved_contexts=[h["content"] for h in r.get("retrieved", [])] or [""],
            retrieved_context_ids=[h["file"] for h in r.get("retrieved", [])],
            reference=r.get("reference") or "",
            reference_contexts=r.get("reference_contexts") or [],
            reference_context_ids=r.get("reference_context_ids") or [],
        )
        for r in rows
    ]

    metrics = []
    if not context_free:
        # Retrieval metrics are undefined without retrieved context.
        metrics += [NonLLMContextPrecisionWithReference(), NonLLMContextRecall(),
                    IDBasedContextPrecision(), IDBasedContextRecall()]
    if not args.deterministic_only:
        llm, emb = build(args.judge, args.embed, args.base_url)
        # Faithfulness scores whether the response is entailed by the RETRIEVED
        # context. A context-free arm has none, so the metric is undefined -- it
        # would score ~0 for every answer regardless of quality, which is a
        # statement about the experimental design, not about the model. Response
        # relevancy only needs the question and the answer, so it applies to all
        # arms and is the only answer-quality metric comparable across them.
        if not context_free:
            metrics.append(Faithfulness(llm=llm))
        metrics.append(ResponseRelevancy(llm=llm, embeddings=emb))

    if not metrics:
        raise SystemExit("nothing to compute for this arm with these options")

    print(f"{path.name}: {len(samples)} samples, "
          f"{'context-free' if context_free else 'retrieval'} arm")
    print(f"  metrics: {', '.join(type(m).__name__ for m in metrics)}")
    if not args.deterministic_only:
        print(f"  judge: {args.judge}   embeddings: {args.embed}")

    result = evaluate(dataset=EvaluationDataset(samples=samples), metrics=metrics)
    df = result.to_pandas()

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / (path.stem + ".ragas.csv")
    df.to_csv(out, index=False)

    numeric = df.select_dtypes("number")
    summary = {c: round(float(numeric[c].mean()), 4) for c in numeric.columns}
    (RESULTS / (path.stem + ".ragas.json")).write_text(
        json.dumps({"arm": path.stem, "judge": args.judge, "n": len(df),
                    "means": summary}, indent=2), encoding="utf-8")

    print(f"\nwrote {out}")
    for k, v in summary.items():
        print(f"  {k:42} {v:.4f}")


if __name__ == "__main__":
    main()
