"""
generate_eval_report.py — Produces evaluation_assessment.pdf in the project root.
"""

import _bootstrap  # noqa: F401  puts the project root on sys.path
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from pathlib import Path

from _bootstrap import ROOT
OUT = ROOT / "rag_benchmarking" / "design" / "evaluation_assessment.pdf"

# ── Colour palette ────────────────────────────────────────────────────────────
NAVY   = (26,  54,  93)
WHITE  = (255, 255, 255)
LGRAY  = (245, 246, 248)
MGRAY  = (200, 205, 212)
DGRAY  = (80,  90, 100)
GREEN  = (39, 174,  96)
RED    = (192,  57,  43)
AMBER  = (211, 136,  36)
BLUE   = (41, 128, 185)


FONTS = "C:/Windows/Fonts/"


class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font("Arial", style="",   fname=FONTS + "arial.ttf")
        self.add_font("Arial", style="B",  fname=FONTS + "arialbd.ttf")
        self.add_font("Arial", style="I",  fname=FONTS + "ariali.ttf")
        self.add_font("Arial", style="BI", fname=FONTS + "arialbi.ttf")

    def header(self):
        self.set_fill_color(*NAVY)
        self.rect(0, 0, 210, 18, "F")
        self.set_y(4)
        self.set_font("Arial", "B", 11)
        self.set_text_color(*WHITE)
        self.cell(0, 10, "AD & Algae Research Chatbot — Evaluation Question Assessment",
                  align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)
        self.ln(4)

    def footer(self):
        self.set_y(-13)
        self.set_font("Arial", "I", 8)
        self.set_text_color(*DGRAY)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    # ── helpers ───────────────────────────────────────────────────────────────

    def section_title(self, text):
        self.set_font("Arial", "B", 11)
        self.set_text_color(*NAVY)
        self.ln(3)
        self.cell(0, 7, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*NAVY)
        self.set_line_width(0.4)
        self.line(self.l_margin, self.get_y(), 210 - self.r_margin, self.get_y())
        self.ln(2)
        self.set_text_color(0, 0, 0)

    def body(self, text, indent=0):
        self.set_font("Arial", "", 9.5)
        self.set_text_color(*DGRAY)
        self.set_x(self.l_margin + indent)
        self.multi_cell(0, 5.5, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)

    def badge(self, symbol, label, colour):
        self.set_font("Arial", "B", 9)
        self.set_fill_color(*colour)
        self.set_text_color(*WHITE)
        self.set_x(self.l_margin)
        self.cell(28, 6, f"  {symbol}  {label}", fill=True,
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def q_header(self, q_num, title):
        self.set_fill_color(*LGRAY)
        self.set_draw_color(*MGRAY)
        self.set_line_width(0.3)
        self.set_font("Arial", "B", 10.5)
        self.set_text_color(*NAVY)
        self.ln(3)
        self.cell(0, 8, f"  {q_num}   {title}", fill=True, border=1,
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def bullet(self, text, indent=5):
        self.set_font("Arial", "", 9.5)
        self.set_text_color(*DGRAY)
        self.set_x(self.l_margin + indent)
        self.cell(5, 5.5, "•")
        self.set_x(self.l_margin + indent + 5)
        self.multi_cell(0, 5.5, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)

    def kv(self, key, val):
        self.set_font("Arial", "B", 9.5)
        self.set_text_color(*NAVY)
        self.set_x(self.l_margin + 5)
        self.cell(38, 5.5, key + ":")
        self.set_font("Arial", "", 9.5)
        self.set_text_color(*DGRAY)
        self.multi_cell(0, 5.5, val, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)

    def summary_table(self, rows):
        col_w = [12, 65, 22, 76]
        headers = ["Q", "Status", "Verdict", "Recommended Action"]
        self.set_font("Arial", "B", 9)
        self.set_fill_color(*NAVY)
        self.set_text_color(*WHITE)
        for i, (h, w) in enumerate(zip(headers, col_w)):
            self.cell(w, 7, f" {h}", fill=True, border=1)
        self.ln()
        self.set_text_color(0, 0, 0)

        fill_colours = [WHITE, LGRAY]
        for ri, (q, status, verdict, action) in enumerate(rows):
            self.set_fill_color(*fill_colours[ri % 2])
            h = 6
            self.set_font("Arial", "B", 9)
            self.cell(col_w[0], h, f" {q}", fill=True, border=1)
            self.set_font("Arial", "", 9)
            self.cell(col_w[1], h, f" {status}", fill=True, border=1)
            # verdict badge colour
            if verdict == "Keep":
                self.set_text_color(*GREEN)
            elif verdict == "Replace":
                self.set_text_color(*RED)
            elif verdict == "Partial":
                self.set_text_color(*AMBER)
            else:
                self.set_text_color(*BLUE)
            self.set_font("Arial", "B", 9)
            self.cell(col_w[2], h, f" {verdict}", fill=True, border=1)
            self.set_text_color(0, 0, 0)
            self.set_font("Arial", "", 9)
            self.cell(col_w[3], h, f" {action}", fill=True, border=1)
            self.ln()


# ── Build document ─────────────────────────────────────────────────────────────

pdf = PDF()
pdf.set_margins(18, 22, 18)
pdf.set_auto_page_break(auto=True, margin=18)
pdf.add_page()

# ── Intro ──────────────────────────────────────────────────────────────────────
pdf.set_font("Arial", "", 9.5)
pdf.set_text_color(*DGRAY)
pdf.multi_cell(0, 5.5,
    "This document reports the results of a corpus audit (v2, updated 2026-05-08) conducted "
    "against the indexed knowledge base as it stood in May 2026 (272 peer-reviewed papers; the "
    "corpus is now 274 unique papers) to assess whether each "
    "proposed evaluation question can be fairly answered by the RAG chatbot. Q1 and Q3 have "
    "been revised since the first audit; a source DOI was also supplied for Q4. For each "
    "question we checked whether the expected answer values appear verbatim in the parsed "
    "text or table chunks, and identified any structural reasons the chatbot may succeed or fail.",
    new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(2)

# ── Q1 ────────────────────────────────────────────────────────────────────────
pdf.q_header("Q1", "Highest methane yield from anaerobic co-digestion with algal biomass (corpus-wide synthesis)")
pdf.badge("OK", "GOOD — KEEP", GREEN)
pdf.kv("Question type", "Corpus-wide synthesis — find the maximum reported value across the corpus")
pdf.kv("Source paper", "Not pinned to a single DOI; answer depends on what the corpus contains")
pdf.kv("RAG suitability", "Appropriate — hybrid mode retrieves top-k chunks on methane yield; LLM synthesises")
pdf.body(
    "This is a well-designed synthesis question. The corpus contained 272 papers on algae-AD "
    "integration, many reporting methane yields for co-digestion experiments. The RAG chatbot "
    "will retrieve the most semantically relevant chunks and attempt to identify the highest "
    "reported value along with the associated co-substrate, species, and operating conditions."
)
pdf.body("\nCaveats:", indent=0)
pdf.bullet(
    "RAG retrieves by semantic similarity, not by magnitude. If the paper with the absolute "
    "highest yield is under-represented in the top-k retrieved chunks, the chatbot may report "
    "a high but not the highest value. This is an inherent ceiling for any top-k RAG system."
)
pdf.bullet(
    "GPT without retrieval would likely give a plausible but generic or stale answer "
    "(training data cut-off, no source citation). The RAG chatbot should outperform GPT here "
    "on citation accuracy and specificity — making this an excellent discriminating question."
)
pdf.bullet(
    "To make scoring unambiguous, identify in advance (from literature review) the paper and "
    "value you consider the correct answer, so you can check whether the chatbot's citation "
    "matches the ground-truth source."
)

# ── Q2 ────────────────────────────────────────────────────────────────────────
pdf.q_header("Q2", "Cultivation mode for C. reinhardtii in Table 2 — DOI: 10.1016/j.biotechadv.2025.108581")
pdf.badge("OK", "GOOD — KEEP", GREEN)
pdf.kv("Paper in corpus", "Yes — indexed as selecting_optimal_algal_strains.pdf")
pdf.kv("Expected answer", "Batch culture")
pdf.kv("Table 2 parsed", "Yes — full HTML table structure captured including 'Cultivation mode' column")
pdf.body(
    "Table 2 was successfully parsed by MinerU into a structured HTML table with columns for "
    "Algal species, Stress, Cultivation mode, Evolutionary duration, Change in phenotype, and "
    "Reference. The Chlamydomonas reinhardtii row should be present; the chatbot needs to "
    "locate the correct row and return 'Batch culture'. This is an appropriate test of "
    "table-specific retrieval and a question GPT is likely to hallucinate or answer generically."
)
pdf.body("\nCaveat:", indent=0)
pdf.bullet(
    "Verify that the C. reinhardtii row appears in the portion of the table that was "
    "captured in the chunk (large tables are sometimes split). A quick manual check "
    "of the paper is recommended before finalising this question."
)

# ── Q3 ────────────────────────────────────────────────────────────────────────
pdf.q_header("Q3", "Operational stage effects on CO2 removal in HRAP system — DOI: 10.1016/j.biortech.2023.129955")
pdf.badge("++", "STRONG — KEEP", GREEN)
pdf.kv("Paper in corpus", "Yes — indexed as 1-s2.0-S0960852423013834-main.pdf")
pdf.kv("Expected: 91.1% CO2-RE", "FOUND verbatim — stage VI text: 'CO2-RE = 91.1%' with 3.4% residual CO2")
pdf.kv("Expected: 94.2% CO2-RE", "FOUND verbatim — stage VII text: 'CO2-RE = 94.2%' with 2.2% residual CO2")
pdf.kv("ammonium stripping / pH 10", "FOUND — mentioned in body text describing stage transitions")
pdf.kv("LNP (nanoparticle) addition", "FOUND — referenced in stage VII context")
pdf.body(
    "All key expected values (91.1%, 94.2%, 3.4% CO2) appear verbatim in text chunks of this "
    "paper, not in figures. The operational stage descriptions (pH adjustment, ammonium stripping, "
    "biogas recirculation, liquid nanoparticle addition) are all present in the indexed text. "
    "This is now one of the two strongest questions in the set. It requires the chatbot to "
    "synthesise a multi-stage narrative from a single paper — something GPT is likely to "
    "either mis-state or report without a specific citation."
)
pdf.body("\nNote on scoring:", indent=0)
pdf.bullet(
    "The expected answer covers multiple stages. Accept the response if it correctly identifies "
    "the 91.1% and 94.2% CO2 removal efficiency values, attributes them to pH adjustment and "
    "LNP addition respectively, and notes the biomass-shading decline at the end of stage VII. "
    "Citation to DOI 10.1016/j.biortech.2023.129955 is required for full credit."
)

# ── Q4 ────────────────────────────────────────────────────────────────────────
pdf.q_header("Q4", "Biomethane purity & HRAP–AD vs PSA economic trade-off — DOI: 10.1016/j.cej.2022.138323")
pdf.badge("~", "PARTIAL", AMBER)
pdf.kv("98.9% CH4 purity", "FOUND — verbatim in insights_microalgaebased_technologies.pdf, parsed table")
pdf.kv("Source paper for PSA figures", "DOI 10.1016/j.cej.2022.138323 — NOT indexed as a standalone file")
pdf.kv("83.3% OpEx reduction vs PSA", "NOT FOUND — source paper absent from corpus (only in ref lists)")
pdf.kv("123.5% CapEx increase vs PSA", "NOT FOUND — same reason")
pdf.body(
    "The 98.9% CH4 purity figure is confirmed in the indexed corpus and will be retrievable. "
    "However, the DOI provided for the PSA comparison (10.1016/j.cej.2022.138323) corresponds "
    "to a paper that is cited by other indexed papers but is not itself in the knowledge base. "
    "The 83.3% and 123.5% figures do not appear anywhere in the index as audited. The chatbot "
    "may give a reasonable qualitative answer about HRAP vs PSA trade-offs from related papers, "
    "but cannot retrieve these specific quantitative values."
)
pdf.body("\nRequired action to promote to 'Keep':", indent=0)
pdf.bullet(
    "Add DOI 10.1016/j.cej.2022.138323 (Chemical Engineering Journal, 2022) to the papers "
    "folder and re-run ingest.py for that file only. Once indexed, Q4 becomes an excellent "
    "two-part test: single-paper lookup (98.9% CH4 purity) plus cross-paper comparison "
    "(HRAP vs PSA costs from the CEJ paper)."
)

# ── Q5 ────────────────────────────────────────────────────────────────────────
pdf.q_header("Q5", "Best methane improvement from algae + wheat straw co-digestion (synthesis question)")
pdf.badge("++", "STRONG — KEEP", GREEN)
pdf.kv("Expected answer", "77% higher methane production vs microalgae mono-digestion")
pdf.kv("Found in chunks", "Yes — verbatim in body text of 1-s2.0-S0960852417304339-main.pdf")
pdf.body(
    "The exact finding — 'The methane yield increased by 77% with the co-digestion as compared "
    "to microalgae mono-digestion' — appears verbatim in a parsed list-content chunk of an "
    "indexed paper. This is the strongest question in the set. It tests whether the chatbot can "
    "retrieve a specific quantitative result that GPT would likely approximate, misattribute, or "
    "state with lower confidence. Recommend keeping this question unchanged."
)

# ── Summary table ─────────────────────────────────────────────────────────────
pdf.add_page()
pdf.section_title("Summary")

pdf.summary_table([
    ("Q1", "Corpus-wide synthesis; methane yield data present", "Keep",
     "Pre-register expected answer + source DOI for scoring"),
    ("Q2", "Table 2 parsed; 'Cultivation mode' column present", "Keep",
     "Verify C. reinhardtii row is in captured chunk"),
    ("Q3", "91.1%, 94.2%, 3.4% CO2 found verbatim in text", "Keep",
     "Strong multi-stage question; all values confirmed"),
    ("Q4", "98.9% CH4 confirmed; PSA source paper not indexed", "Partial",
     "Index DOI 10.1016/j.cej.2022.138323 to complete Q4"),
    ("Q5", "77% value found verbatim in indexed body text", "Keep",
     "No changes needed — strongest single-lookup question"),
])

pdf.ln(6)
pdf.section_title("General Recommendations")
pdf.bullet(
    "The set is now substantially stronger after the Q1 and Q3 revisions. Q1 (corpus-wide "
    "synthesis), Q3 (multi-stage HRAP text), and Q5 (verbatim co-digestion yield) are all "
    "ready to use. Q2 needs a quick manual check; Q4 needs one additional paper indexed."
)
pdf.bullet(
    "Index DOI 10.1016/j.cej.2022.138323 to unlock Q4. This requires downloading the PDF, "
    "placing it in the papers/ folder, and running: python ingest.py (it will skip already-"
    "indexed files and process only the new one)."
)
pdf.bullet(
    "For each question, record the source paper DOI alongside the expected answer before "
    "running the evaluation. This makes scoring unambiguous and lets you verify whether the "
    "chatbot's in-text citation matches the ground-truth source."
)
pdf.bullet(
    "Run a parallel baseline: ask GPT-4o (no retrieval) the same five questions. Q2 tests "
    "table lookup (GPT likely gives a generic cultivation mode), Q3 tests multi-stage "
    "quantitative recall (GPT unlikely to cite 91.1% or 94.2% correctly), and Q5 tests "
    "specific yield data. These are the three clearest cases where RAG provides a verifiable "
    "factual advantage."
)

pdf.output(str(OUT))
print(f"Saved: {OUT}")
