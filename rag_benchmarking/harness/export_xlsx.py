"""Export the whole benchmark to one workbook.

Sheets:
    Summary          every metric, arms as columns, with caveats inline
    Paired tests     McNemar exact, each baseline vs multimodal
    Questions        the ground-truth set
    Answers          every arm's full answer, side by side per question
    Per-question     per-arm scores for each question
    Retrieved        what the retrieval arms actually retrieved, ranked

Usage:  python export_xlsx.py
"""
import csv
import json
from collections import defaultdict
from math import comb

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from paths import DATASET, RESULTS, RUNS, BENCH

NAVY, GREY, AMBER, RED = "1A365D", "F5F6F8", "FFF4E5", "FDE8E8"
ARMS = ["multimodal", "text_only", "no_retrieval", "gemma3_no_retrieval",
        "gpt_strict", "gpt_websearch"]
LABEL = {"multimodal": "multimodal RAG", "text_only": "text-only RAG",
         "no_retrieval": "qwen3:14b no-retrieval",
         "gemma3_no_retrieval": "gemma3:27b no-retrieval",
         "gpt_strict": "GPT-6 Astra no-retrieval",
         "gpt_websearch": "GPT-6 Astra + web search"}


def head(ws, values, fill=NAVY, color="FFFFFF"):
    for j, v in enumerate(values, 1):
        c = ws.cell(1, j, v)
        c.font = Font(bold=True, color=color, size=10)
        c.fill = PatternFill("solid", fgColor=fill)
        c.alignment = Alignment(vertical="center", wrap_text=True)


def widths(ws, ws_widths):
    for j, w in enumerate(ws_widths, 1):
        ws.column_dimensions[get_column_letter(j)].width = w


def load():
    rows = [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()]
    scored, summary, ragas, runs, ragas_rows = {}, {}, {}, {}, {}
    for a in ARMS:
        p = RESULTS / f"{a}.scored.jsonl"
        if p.exists():
            scored[a] = {s["id"]: s for s in map(json.loads, p.read_text(encoding="utf-8").splitlines())}
        q = RESULTS / f"{a}.summary.json"
        if q.exists():
            summary[a] = json.loads(q.read_text(encoding="utf-8"))
        r = RESULTS / f"{a}.ragas.json"
        if r.exists():
            ragas[a] = json.loads(r.read_text(encoding="utf-8")).get("means", {})
        u = RUNS / f"{a}.jsonl"
        if u.exists():
            runs[a] = {x["id"]: x for x in map(json.loads, u.read_text(encoding="utf-8").splitlines())}
        c = RESULTS / f"{a}.ragas.csv"
        if c.exists():
            with c.open(encoding="utf-8", newline="") as fh:
                ragas_rows[a] = list(csv.DictReader(fh))
    return rows, scored, summary, ragas, runs, ragas_rows


def mcnemar(b, c):
    n = b + c
    if not n:
        return 1.0
    return min(sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n * 2, 1.0)


def main():
    rows, scored, summary, ragas, runs, ragas_rows = load()
    wb = Workbook()

    # ── Summary ───────────────────────────────────────────────────────────────
    ws = wb.active
    ws.title = "Summary"
    head(ws, ["metric"] + [LABEL[a] for a in ARMS] + ["note"])

    def g(d, a, k):
        v = d.get(a, {}).get(k)
        return v if not isinstance(v, str) else None

    SPEC = [
        ("questions", lambda a: summary.get(a, {}).get("questions"), ""),
        ("RETRIEVAL — deterministic", None, ""),
        ("gold paper retrieved (of 16)", lambda a: g(summary, a, "gold_paper_retrieved"), ""),
        ("gold chunk retrieved (of 16)", lambda a: g(summary, a, "gold_chunk_retrieved"),
         "known-item retrieval: a lower bound on retrieval failure, not exhaustive recall"),
        ("RETRIEVAL — RAGAS 0.4.3", None, ""),
        ("id_based_context_recall", lambda a: ragas.get(a, {}).get("id_based_context_recall"),
         "matches 'gold paper retrieved' exactly — independent cross-validation"),
        ("id_based_context_precision", lambda a: ragas.get(a, {}).get("id_based_context_precision"), ""),
        ("non_llm_context_recall", lambda a: ragas.get(a, {}).get("non_llm_context_recall"),
         "NEAR-FLOOR FOR A MECHANICAL REASON: string similarity between a short pinned "
         "passage and a full retrieved chunk, thresholded at 0.5. Do not read as recall."),
        ("non_llm_context_precision", lambda a: ragas.get(a, {}).get("non_llm_context_precision_with_reference"),
         "same caveat as above"),
        ("ANSWER CONTENT — deterministic", None, ""),
        ("gold values correct (of 53)",
         lambda a: g(summary, a, "values_substantiated") or g(summary, a, "values_correct"),
         "for retrieval arms this is 'substantiated': in the answer AND in the retrieved context"),
        ("gold values missed", lambda a: g(summary, a, "values_missed"), ""),
        ("mean accuracy", lambda a: g(summary, a, "mean_accuracy"),
         "mean over questions of correct/total gold values; multi-value questions dominate"),
        ("ANSWER QUALITY — RAGAS", None, ""),
        ("faithfulness", lambda a: ragas.get(a, {}).get("faithfulness"),
         "undefined for context-free arms. Judge phi4:14b. ~5 points of run-to-run variance "
         "observed at n=16 (0.930 then 0.979 for multimodal)"),
        ("answer_relevancy", lambda a: ragas.get(a, {}).get("answer_relevancy"),
         "DO NOT READ AS QUALITY. Measures whether an answer addresses the question, not "
         "whether it is correct, and scores abstention 0. Ranks gemma3:27b no-retrieval "
         "highest (0.861) on 2/53 values correct."),
        ("BEHAVIOUR — deterministic", None, ""),
        ("values fabricated", lambda a: g(summary, a, "values_fabricated"),
         "in the answer but absent from retrieved context; undefined without context"),
        ("questions abstained", lambda a: g(summary, a, "questions_abstained"), ""),
        ("questions asserting numbers with no gold value",
         lambda a: g(summary, a, "questions_asserted_without_gold"),
         "the confident-wrong-answer rate"),
    ]
    r = 2
    for name, fn, note in SPEC:
        if fn is None:
            c = ws.cell(r, 1, name)
            c.font = Font(bold=True, color=NAVY, size=10)
            for j in range(1, len(ARMS) + 3):
                ws.cell(r, j).fill = PatternFill("solid", fgColor=GREY)
            r += 1
            continue
        ws.cell(r, 1, name).font = Font(size=10)
        for j, a in enumerate(ARMS, 2):
            v = fn(a)
            cell = ws.cell(r, j, "n/a" if v is None else v)
            cell.alignment = Alignment(horizontal="center")
            if isinstance(v, float):
                cell.number_format = "0.000"
        nc = ws.cell(r, len(ARMS) + 2, note)
        nc.font = Font(size=8, italic=True, color="506470")
        nc.alignment = Alignment(wrap_text=True, vertical="top")
        if note.startswith(("DO NOT", "NEAR-FLOOR")):
            for j in range(1, len(ARMS) + 3):
                ws.cell(r, j).fill = PatternFill("solid", fgColor=AMBER)
        r += 1
    widths(ws, [44] + [19] * len(ARMS) + [62])
    ws.freeze_panes = "B2"

    # ── Paired tests ──────────────────────────────────────────────────────────
    ws = wb.create_sheet("Paired tests")
    head(ws, ["comparison", "unit", "favours multimodal", "favours baseline",
              "discordant", "McNemar exact p", "reading"])
    mm = scored.get("multimodal", {})
    r = 2
    if mm and "text_only" in scored:
        to = scored["text_only"]
        ids = sorted(set(mm) & set(to))
        b = sum(1 for i in ids if mm[i]["gold_chunk_retrieved"] and not to[i]["gold_chunk_retrieved"])
        c = sum(1 for i in ids if not mm[i]["gold_chunk_retrieved"] and to[i]["gold_chunk_retrieved"])
        ws.append(["multimodal vs text-only", "gold chunk retrieved", b, c, b + c,
                   round(mcnemar(b, c), 5), "significant"])
        r += 1
    for a in ARMS[1:]:
        if a not in scored:
            continue
        o = scored[a]
        b = c = 0
        for i in sorted(set(mm) & set(o)):
            for v in mm[i]["substantiated"] + mm[i]["missed"] + mm[i]["fabricated"]:
                in_mm = v in mm[i]["substantiated"]
                in_o = v in o[i]["substantiated"]
                if in_mm and not in_o:
                    b += 1
                elif in_o and not in_mm:
                    c += 1
        p = mcnemar(b, c)
        ws.append([f"multimodal vs {LABEL[a]}", "gold value", b, c, b + c, round(p, 5),
                   "significant" if p < 0.05 else "no detectable difference"])
    widths(ws, [38, 22, 18, 18, 12, 17, 26])
    ws.freeze_panes = "A2"

    # ── Questions ─────────────────────────────────────────────────────────────
    ws = wb.create_sheet("Questions")
    head(ws, ["id", "question", "ideal answer (ground truth)", "gold values", "stratum",
              "evidence type", "source paper", "source DOI", "annotator", "access", "year"])
    for q in rows:
        ws.append([q["id"], q["user_input"], q["reference"], ", ".join(q["gold_values"]),
                   q.get("stratum"), q.get("evidence_type"), q.get("source_filename"),
                   q.get("source_doi"), q.get("annotator"), q.get("access"), q.get("year")])
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
            c.font = Font(size=9)
    widths(ws, [7, 56, 52, 24, 17, 13, 34, 28, 11, 13, 7])
    ws.freeze_panes = "B2"

    # ── Answers ───────────────────────────────────────────────────────────────
    ws = wb.create_sheet("Answers")
    head(ws, ["id", "question", "ground truth"] + [LABEL[a] for a in ARMS])
    for q in rows:
        ws.append([q["id"], q["user_input"], q["reference"]] +
                  [(runs.get(a, {}).get(q["id"], {}) or {}).get("answer", "") for a in ARMS])
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
            c.font = Font(size=9)
    widths(ws, [7, 44, 40] + [60] * len(ARMS))
    ws.freeze_panes = "D2"

    # ── Per-question scores ───────────────────────────────────────────────────
    ws = wb.create_sheet("Per-question")
    head(ws, ["id", "stratum", "n gold values", "arm", "gold paper rank",
              "gold chunk retrieved", "values correct", "values missed", "accuracy",
              "fabricated", "abstained", "asserted w/o gold", "faithfulness",
              "answer_relevancy"])
    rag_by = {}
    for a, rr in ragas_rows.items():
        for i, rec in enumerate(rr):
            if i < len(rows):
                rag_by[(a, rows[i]["id"])] = rec
    for q in rows:
        for a in ARMS:
            s = scored.get(a, {}).get(q["id"])
            if not s:
                continue
            rr = rag_by.get((a, q["id"]), {})
            def num(k):
                try: return round(float(rr.get(k, "")), 3)
                except Exception: return None
            ws.append([q["id"], q.get("stratum"), s["n_gold_values"], LABEL[a],
                       s.get("gold_paper_rank"), s.get("gold_chunk_retrieved"),
                       len(s["substantiated"]), len(s["missed"]), s.get("accuracy"),
                       len(s["fabricated"]), s.get("abstained"),
                       s.get("asserted_without_gold"), num("faithfulness"),
                       num("answer_relevancy")])
    widths(ws, [7, 17, 13, 24, 15, 20, 14, 14, 10, 11, 11, 18, 13, 15])
    ws.freeze_panes = "A2"

    # ── Retrieved chunks ──────────────────────────────────────────────────────
    ws = wb.create_sheet("Retrieved")
    # No chunk excerpts: the repository is public and the sources are mostly
    # paywalled. Everything needed to verify a retrieval score is still here.
    head(ws, ["id", "arm", "rank", "similarity", "chunk type", "source file",
              "is gold paper", "chunk id"])
    for q in rows:
        for a in ("multimodal", "text_only"):
            rec = runs.get(a, {}).get(q["id"])
            if not rec:
                continue
            for n, h in enumerate(rec.get("retrieved", []), 1):
                ws.append([q["id"], LABEL[a], n, round(h["similarity"], 4), h["type"],
                           h["file"], h["file"] == q.get("source_filename"),
                           h.get("chunk_id", "")])
    for row in ws.iter_rows(min_row=2):
        row[-1].alignment = Alignment(wrap_text=True, vertical="top")
        for c in row:
            c.font = Font(size=9)
    widths(ws, [7, 24, 7, 11, 13, 34, 13, 42])
    ws.freeze_panes = "A2"

    out = BENCH / "results" / "benchmark_results.xlsx"
    wb.save(out)
    print(f"wrote {out}")
    for s in wb.sheetnames:
        print(f"  {s:16} {wb[s].max_row - 1} data rows")


if __name__ == "__main__":
    main()
