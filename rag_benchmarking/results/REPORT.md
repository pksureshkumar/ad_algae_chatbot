# Benchmark results

16 questions, each pinned to a specific value in a specific paper.
All local arms use qwen3:14b at temperature 0.1 unless noted.

## Overall

| arm | gold chunk retrieved | values correct | mean accuracy | fabrications |
|---|---|---|---|---|
| `multimodal` | 14/16 | 41/53 | 0.762 | 0 |
| `text_only` | 6/16 | 9/53 | 0.229 | 0 |
| `no_retrieval` | n/a | 1/53 | 0.013 | n/a |
| `gemma3_no_retrieval` | n/a | 2/53 | 0.023 | n/a |
| `gpt_strict` | n/a | 0/53 | 0.000 | n/a |
| `gpt_websearch` | n/a | 37/53 | 0.675 | n/a |

## By stratum

The text-available stratum is the control group: values there also appear in
body text, so a text-only baseline can reach them. Equivalent performance in
that stratum, with divergence elsewhere, is what shows the advantage is
specific to table-derived content rather than general retrieval superiority.

| stratum | n | arm | gold chunk | values correct |
|---|---|---|---|---|
| table-exclusive | 9 | `multimodal` | 7/9 | 17/25 |
| table-exclusive | 9 | `text_only` | 1/9 | 3/25 |
| table-exclusive | 9 | `no_retrieval` | n/a | 0/25 |
| table-exclusive | 9 | `gemma3_no_retrieval` | n/a | 0/25 |
| table-exclusive | 9 | `gpt_strict` | n/a | 0/25 |
| table-exclusive | 9 | `gpt_websearch` | n/a | 23/25 |
| mixed | 4 | `multimodal` | 4/4 | 17/17 |
| mixed | 4 | `text_only` | 2/4 | 0/17 |
| mixed | 4 | `no_retrieval` | n/a | 0/17 |
| mixed | 4 | `gemma3_no_retrieval` | n/a | 1/17 |
| mixed | 4 | `gpt_strict` | n/a | 0/17 |
| mixed | 4 | `gpt_websearch` | n/a | 8/17 |
| text-available | 3 | `multimodal` | 3/3 | 7/11 |
| text-available | 3 | `text_only` | 3/3 | 6/11 |
| text-available | 3 | `no_retrieval` | n/a | 1/11 |
| text-available | 3 | `gemma3_no_retrieval` | n/a | 1/11 |
| text-available | 3 | `gpt_strict` | n/a | 0/11 |
| text-available | 3 | `gpt_websearch` | n/a | 6/11 |

## Paired tests

**multimodal vs text_only**, gold-chunk retrieval: 8 questions favour
multimodal, 0 favour text-only, 8 discordant pairs, McNemar exact two-sided **p = 0.00781**.

## Caveats

- Known-item retrieval, not exhaustive recall: each question pins one passage
  we know contains the answer. This bounds retrieval failure from below; it does
  not measure everything the system missed across the corpus.
- Context-free arms answer from parametric knowledge, so retrieval metrics and
  substantiation are undefined for them, not zero.
- Questions are deliberately specific and mostly table-derived. Ungrounded
  accuracy would be higher on general questions; this is not a general
  capability assessment.
- Three questions (Q01, Q05, Q07) miss the source paper in both retrieval arms.
  They do not identify which of 274 similar papers they refer to.
