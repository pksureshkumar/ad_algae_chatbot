# RAG benchmarking

Quantitative validation of the AD/algae chatbot, built to answer a reviewer's request to
*"clearly report the ground truth, evaluation criteria, test-question selection, scoring
procedure, and appropriate retrieval/answer-quality metrics."*

**Start with [`results/benchmark_results.xlsx`](results/benchmark_results.xlsx)** — every metric,
every arm, all questions and answers, with the caveats written into the sheet. Text version:
[`results/REPORT.md`](results/REPORT.md).

## Results

16 questions, each pinned to a specific numeric value in a specific paper. 53 gold values total.

| arm | gold chunk retrieved | values recovered | mean accuracy | faithfulness | asserted numbers with no gold value |
|---|---|---|---|---|---|
| **multimodal** | **14/16** | **41/53** | **0.762** | **0.940** | 0 |
| `text_only` | 6/16 | 9/53 | 0.229 | 0.765 | 2 |
| `no_retrieval` (qwen3:14b) | n/a | 1/53 | 0.013 | n/a | 10 |
| `gemma3_no_retrieval` | n/a | 2/53 | 0.023 | n/a | 7 |
| `gpt_strict` (Astra, no retrieval) | n/a | 0/53 | 0.000 | n/a | 0 |
| `gpt_websearch` (Astra + web) | n/a | 37/53 | 0.675 | n/a | 0 |

Paired tests against `multimodal`, McNemar exact:

| versus | unit | split | p |
|---|---|---|---|
| `text_only` | gold chunk | 8–0 | **0.0078** |
| `text_only` | gold values | 33–1 | **<0.0001** |
| `no_retrieval` | gold values | 41–1 | **<0.0001** |
| `gemma3_no_retrieval` | gold values | 40–1 | **<0.0001** |
| `gpt_strict` | gold values | 41–0 | **<0.0001** |
| `gpt_websearch` | gold values | 14–10 | 0.54 — no detectable difference |

Three readings worth stating plainly. Withholding table and figure chunks from an otherwise
identical system costs it more than half its retrieval and three quarters of its answers.
Removing retrieval entirely collapses performance regardless of model scale — 14B, 27B and a
frontier model all land at 1, 2 and 0 of 53, and **0 of 25 on the table-exclusive stratum**. And a
14B model running locally over a curated corpus is indistinguishable from GPT-6 Astra with live
web access, while costing nothing per query and citing a fixed, auditable source set.

## The six arms

| arm | what it isolates |
|---|---|
| `multimodal` | the full index |
| `text_only` | identical, but `table`/`image`/`chart` chunks are ineligible |
| `no_retrieval` | retrieval, with the model held fixed (same qwen3:14b) |
| `gemma3_no_retrieval` | whether model scale substitutes for retrieval |
| `gpt_strict` | a frontier model answering from parametric knowledge only |
| `gpt_websearch` | a frontier model with live web retrieval |

`multimodal` vs `text_only` is a single-variable ablation: same index, embeddings, LLM, prompt and
corpus. `no_retrieval` exists because comparing only against another vendor's model would confound
retrieval with model identity.

The two GPT arms were collected by hand through the web interface (no API budget was available),
so they carry no pinned model snapshot — `gpt_arm/prompts.md` records the exact prompts and
`answers.md` the replies. Browsing could not be reliably disabled from the account settings, so
suppression was done in the prompt; `parse_answers.py` fails loudly if any answer contains a URL.
`gpt_websearch` is labelled as web-augmented because the first collection pass ran with search on
and 15 of 16 answers came back carrying live URLs.

## Pipeline

```bash
cd harness
python build_dataset.py      # spreadsheet + pilot set -> data/ground_truth.jsonl
python screen.py --write     # text-leakage and uniqueness screening; assigns strata
python run_arms.py           # the local arms -> runs/*.jsonl
python score.py --run ../runs/multimodal.jsonl
python report.py             # results/REPORT.md
python export_xlsx.py        # results/benchmark_results.xlsx
python make_public.py        # runs_public/ with copied paper text removed
```

RAGAS runs in a **separate virtualenv** — see the header of `harness/ragas_eval.py`, which
documents two install traps in ragas 0.4.3. Never install ragas alongside the chatbot's
dependencies.

```bash
../.venv-ragas/Scripts/python.exe harness/ragas_eval.py --run runs/multimodal.jsonl --judge phi4:14b
bash run_all_ragas.sh        # all six arms, sequentially
```

## Layout

```
├── results/            scores, RAGAS output, REPORT.md, the workbook
├── data/               the question set (public variant, copied text removed)
├── runs_public/        every arm's retrieval and answers, chunk bodies removed
├── harness/            the code
├── pilot/              five machine-proposed questions and the pilot run
├── design/             earlier 90-question design and the May 2026 corpus audit
├── gpt_strict_arm/     prompts and collected answers, browsing suppressed
├── gpt_websearch_arm/  prompts and collected answers, browsing on
└── corpus_paper_list.csv   the 274 indexed papers with DOIs
```

## Things that will bite you

**Scoring on "is the gold value in the answer" is unsafe.** In the pilot, the text-only arm
reported `Kkp = 35.38` and `n = 0.19` with a citation. Both values were correct, and both existed
in exactly one chunk of that paper — the table it was blocked from retrieving. They were not in its
context. Naive string matching scored that fabrication 2/3. `score.py` therefore classifies every
gold value as *substantiated* (in the answer **and** the retrieved context), *fabricated* (answer
only), or *missed*, and tracks fabrication separately.

**The ablation only works at chunk level.** Entities were extracted from all chunks including
table and image ones, so their descriptions already carry multimodal content. Masking chunks in a
graph query mode would not remove it. `retrieval.py` bypasses LightRAG's graph modes for this
reason.

**Masking by chunk *type* is not enough.** Some chunks typed `None` carry parsed table HTML or a
vision block verbatim — 1.2% of otherwise unmasked chunks in the original corpus, 3.6% in the
papers added in Sep 2026. Both arms also mask by content. MinerU 3.4.4 additionally emits a
`chart` type that the earlier ingest never produced.

**Verify the index before trusting a run.** An ingestion reported success for ten files while
leaving 1,329 chunks across five papers unvectorised, so those papers were in the knowledge base
and unreachable. Questions on them scored as retrieval failures for reasons unrelated to the
system. `Index` now refuses to run when a question's source paper has no vectors; repair with
`../repair_missing_vectors.py`.

**Two RAGAS metrics must not be read at face value.** `answer_relevancy` measures whether an
answer addresses the question, not whether it is correct, and scores abstention 0 — it ranks
`gemma3_no_retrieval` highest (0.861) on 2 of 53 values correct. The `non_llm_context_*` metrics
compare a short pinned passage against full retrieved chunks by string similarity at a 0.5
threshold, so they sit near zero mechanically. Use the `id_based_*` metrics for retrieval;
`id_based_context_recall` reproduces `score.py`'s paper-level figure exactly, which is a useful
independent check.

**LLM-judged metrics are not deterministic.** Faithfulness for `multimodal` came out 0.930, 0.979 and 0.940
across runs on near-identical data with the same judge. Lead with the deterministic metrics.

## Strata

Results are reported split, not pooled, and `screen.py` assigns strata mechanically rather than
trusting the `evidence_type` label — roughly 25% of table-derived values also appear in the prose.

| stratum | n | meaning |
|---|---|---|
| table-exclusive | 9 | gold values appear only in table/figure chunks — the primary claim |
| mixed | 4 | some values reachable without tables |
| text-available | 3 | all values also in body text — the control group |

On the control stratum the two retrieval arms perform **identically**. The advantage appears only
where values live in tables, which is what shows it comes from multimodal parsing rather than
general retrieval superiority.

## Scope of the retrieval claim

Pinning one passage per question measures **known-item retrieval**: we know the answer is in chunk
C, did the system find chunk C? That is a lower bound on retrieval failure, not exhaustive recall —
measuring true recall would require annotating every relevant passage across all 274 papers.
Report it as gold-passage retrieval success at *k*, not as "recall".

Three questions (Q01, Q05, Q07) do not identify which of 274 similar papers they refer to; GPT-6
Astra said so explicitly for Q01. They are retained and reported rather than dropped.
