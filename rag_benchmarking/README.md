# RAG benchmarking

Quantitative validation of the AD/algae chatbot, built to answer Reviewer 2's
second comment: *"clearly report the ground truth, evaluation criteria,
test-question selection, scoring procedure, and appropriate retrieval/answer-quality
metrics."*

## Layout

```
rag_benchmarking/
├── ground_truth_final.xlsx          reviewed question set (the source of truth)
├── ground_truth_template_updated.xlsx   co-authors' original submission, untouched
├── doi_map.json                     PDF filename -> DOI for the added papers
├── additional_ingestation_papers/   PDFs named by DOI, ingested separately
├── harness/                         the code
├── data/                            generated dataset (ground_truth.jsonl)
├── runs/                            one JSONL per arm, with full retrieved context
└── results/                         scored output, summaries, screening report
```

## The four arms

| arm | what it isolates |
|---|---|
| `multimodal` | the full index |
| `text_only` | identical, but `table` and `image` chunks are ineligible |
| `gpt_strict` | general LLM, no retrieval, strict prompt |
| `gpt_open` | general LLM, no retrieval, unconstrained |

`multimodal` vs `text_only` is a single-variable ablation: same index, same
embeddings, same LLM, same prompt, same corpus. The only difference is whether
table- and figure-derived chunks can be retrieved.

The GPT arms need an API key and a **pinned model snapshot**, and are produced
outside this harness into `runs/gpt_*.jsonl` in the same record shape. Do not run
them through a subscription-backed agent CLI: those carry their own system prompts
and tools, may perform live web retrieval, and cannot pin a model version — you
would be benchmarking an agent, not a language model.

## Pipeline

```bash
cd rag_benchmarking/harness

python build_dataset.py        # xlsx -> data/ground_truth.jsonl  (RAGAS field names)
python screen.py --write       # assign strata; requires the papers to be indexed
python run_arms.py             # -> runs/multimodal.jsonl, runs/text_only.jsonl
python score.py --run ../runs/multimodal.jsonl
```

RAGAS scoring runs in a **separate environment** — see the header of
`ragas_eval.py`. Never install ragas into `ad_algae`.

## Two things that are easy to get wrong

**Scoring on "is the gold value in the answer" is unsafe.** In the pilot, the
text-only arm reported `Kkp = 35.38` and `n = 0.19` with a citation. Both values
were correct, and both existed in exactly one chunk of that paper — the table it
was blocked from retrieving. They were not in its context. Naive string matching
scored that fabrication 2/3. `score.py` therefore classifies every gold value as
*substantiated* (in the answer **and** the retrieved context), *fabricated* (in the
answer only), or *missed*, and tracks fabrication as its own rate.

**The ablation only works at chunk level.** Entities were extracted from all
chunks including table and image ones, so their descriptions already carry
multimodal content. Masking chunks in a graph query mode would not remove it and
the comparison would be confounded. `retrieval.py` bypasses LightRAG's graph modes
for this reason.

## Strata

Results are reported split, not pooled:

- **table-exclusive** — gold values appear only in table/figure chunks. This is
  where multimodal retrieval should win, and it is the primary claim.
- **text-available** — gold values also appear in body text. The control group,
  where both arms should perform alike. Reporting this is what shows the test was
  not rigged.

Roughly 25% of table-derived values also appear in the prose, so `screen.py`
reassigns strata mechanically rather than trusting the `evidence_type` label.

## Scope of the retrieval claim

Pinning one passage per question measures **known-item retrieval**: we know the
answer is in chunk C, did the system find chunk C? That is a lower bound on
retrieval failure, not exhaustive recall — measuring true recall would require
annotating every relevant passage across all indexed papers. State it as
gold-passage retrieval success at *k*, not as "recall".
