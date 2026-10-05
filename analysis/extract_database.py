"""
extract_database.py — Extract structured numerical data from rag_storage into Excel.

Reads text chunks (especially parsed table HTML) from kv_store_text_chunks.json,
runs GPT-4o to extract rows matching the 65-column database schema, and saves to
output/database_extraction.xlsx.

Checkpoint system: per-paper results are saved immediately to
output/extraction_checkpoint.json so the run can be resumed after interruption.

Usage:
    python extract_database.py                  # run on 15 curated papers
    python extract_database.py --all            # run on all ingested papers
    python extract_database.py --paper some_file.pdf  # single paper
    python extract_database.py --reset          # clear checkpoint and re-run
"""

import _bootstrap  # noqa: F401  puts the project root on sys.path
import asyncio
import json
import argparse
import logging
from pathlib import Path

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

from models import llm_model_func
from config import RAG_STORAGE_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from _bootstrap import OUTPUT_DIR
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_XLSX = OUTPUT_DIR / "database_extraction.xlsx"
CHECKPOINT_FILE = OUTPUT_DIR / "extraction_checkpoint.json"

# ── 15 curated papers ─────────────────────────────────────────────────────────
# 10 validation papers (data-rich) + 5 additional high-value AD/algae papers
CURATED_PAPERS = [
    # Original 10 validation papers
    "comprehensive_review_biomethane_production.pdf",   # 13 tables, 165 numerical — broad AD data
    "biohythane_production_anaerobic_digestion.pdf",    # 20 tables — H2 + CH4 from AD
    "lifecycle_assessment_powertomethane.pdf",          # 28 tables — LCA / TEA
    "prefeasibility_analysis_different_ad.pdf",         # 11 tables — AD feasibility / TEA
    "technoeconomic_feasibility_macroalgae_ad.pdf",     # 9 tables  — macroalgae AD TEA
    "integrating_ad_effluent_microalgal.pdf",           # 8 tables  — AD effluent + microalgae
    "comparative_technoeconomic_lifecycle.pdf",         # 8 tables  — TEA + LCA comparison
    "kinetic_modeling_optimization_biogas_production.pdf", # 7 tables — kinetics + yield data
    "biogas_production_optimization_anaerobic.pdf",     # 6 tables  — AD optimization, yield data
    "insights_microalgaebased_technologies.pdf",        # 5 tables  — HRAP + microalgae tech
    # 5 additional high-value papers
    "digestate_dilution_shapes_carb.pdf",               # digestate dilution for algae cultivation
    "application_fenps_enhancing_methane.pdf",          # FeNPs enhancing methane — experimental data
    "enhancing_biogas_production_ultrasound.pdf",       # ultrasound pretreatment yields
    "enhancing_vfa_seaweed.pdf",                        # VFA from seaweed AD
    "harnessing_green_tide_ulva.pdf",                   # Ulva macroalgae AD integration
]

# ── 36-column schema ─────────────────────────────────────────────────────────
# Dropped from original 65:
#   journal              — not in papers_metadata.json
#   inoculum_source      — 1.7% fill, rarely reported
#   vs_ts_ratio          — 0.7% fill
#   reactor_volume_L     — 4.9% fill, not critical
#   pretreatment_conditions — redundant with pretreatment_method + notes
#   srt_value/unit       — 0.2% fill
#   digestate_dilution_factor — 0.2% fill
#   inoculum_to_substrate_ratio — 0.7% fill
#   bmp_protocol         — 0.1% fill
#   propionate/butyrate_fraction_percent — 0.1-0.3% fill
#   vfa_improvement_percent — 0.2% fill
#   tea_system_boundary/scale — 2-6% fill, captured in notes
#   lca_gwp_value/unit   — 1.9% fill
#   net_energy_ratio     — 0.4% fill
#   pretreatment_energy_cost_reported — 6.1% fill, binary flag rarely useful
#   free_ammonia/total_ammonium/inhibition/ic50 — <1.5% fill
#   control_condition    — captured in notes
COLUMNS = [
    # Paper metadata (3)
    "paper_id", "doi", "article_title", "publication_year",
    # Biology / feedstock (4)
    "algae_species", "algae_role", "primary_substrate", "co_substrate",
    # Process / reactor (5)
    "process_category", "reactor_type", "feed_mode", "scale", "pretreatment_method",
    # Operating conditions (9)
    "temperature_value", "temperature_unit", "temperature_regime",
    "ph_value",
    "hrt_value", "hrt_unit",
    "olr_value", "olr_unit",
    "duration_value", "duration_unit",
    # Methane / biogas (7)
    "methane_yield_value", "methane_yield_unit",
    "methane_yield_normalized_mL_per_gVS",
    "methane_content_percent", "methane_improvement_percent",
    "biogas_yield_value", "biogas_yield_unit",
    # VFA (3)
    "total_vfa_value", "total_vfa_unit", "acetate_fraction_percent",
    # Other bioproducts (3)
    "bioproduct_name", "bioproduct_yield_value", "bioproduct_yield_unit",
    # TEA / LCA (3)
    "tea_metric", "tea_value", "tea_unit",
    # Data quality (3)
    "data_source", "extraction_confidence", "notes",
]

EXTRACTION_PROMPT = """\
Extract structured experimental data from the scientific paper chunk below for a \
research database on anaerobic digestion (AD) and algal bioprocesses.

Rules:
- Return a JSON array; each element = one distinct experimental condition / data point.
- One row per condition. If results are reported at 3 HRTs → 3 rows.
- If methane yield AND VFA come from the same condition → one row.
- Only populate a field if the value is explicitly stated. Use null otherwise.
- For ranges (e.g. "200–350 mL/g VS"), record the midpoint or the reported value as-is in notes.
- For methane_yield_normalized_mL_per_gVS: 1 L/kg VS = 1 mL/g VS = 1 NmL/g VS = 1 m3/t VS.
- If the chunk is a literature review table aggregating data from other papers, still extract each row.
- If no quantitative data exists, return [].

Fields (use exact key names, null if absent):
  algae_species           - scientific name, or null
  algae_role              - feedstock | cultivation_medium | biogas_upgrading | photosynthetic_biocathode | co_culture | none
  primary_substrate       - main AD feedstock (e.g. microalgae biomass, sewage sludge, food waste)
  co_substrate            - secondary substrate in co-digestion, or null
  process_category        - anaerobic_digestion | algae_cultivation | algae_AD_integrated | bioelectrochemical | hydrothermal | fermentation | other
  reactor_type            - CSTR | batch | HRAP | PBR | MEC | BES | UASB | AnMBR | tubular_PBR | raceway | or as written
  feed_mode               - batch | semi_continuous | continuous | fed_batch | not reported
  scale                   - lab | pilot | demonstration | not reported
  pretreatment_method     - thermal | ultrasound | microwave | enzymatic | chemical | mechanical | combination | none | not reported
  temperature_value       - numeric
  temperature_unit        - C or K
  temperature_regime      - mesophilic | thermophilic | psychrophilic | not reported
  ph_value                - numeric, or null
  hrt_value               - numeric, or null
  hrt_unit                - days or hours
  olr_value               - numeric, or null
  olr_unit                - g_VS/L/d | g_COD/L/d | kg_VS/m3/d | as written
  duration_value          - numeric experiment duration, or null
  duration_unit           - days or hours
  methane_yield_value     - numeric only (not a range)
  methane_yield_unit      - exact unit as written (mL CH4/g VS, L/kg VS, NmL/g VS, mL/g TS, etc.)
  methane_yield_normalized_mL_per_gVS - converted value in mL CH4/g VS, or null if unit unclear
  methane_content_percent - % CH4 in biogas, numeric only
  methane_improvement_percent - % improvement over control if stated, numeric only
  biogas_yield_value      - total biogas yield numeric (only if methane yield not given separately)
  biogas_yield_unit       - mL/g_VS | L/kg_VS | mL/g_TS | as written
  total_vfa_value         - numeric total VFA concentration
  total_vfa_unit          - g/L | mg/L | g_COD/L
  acetate_fraction_percent - acetate as % of total VFA, numeric only
  bioproduct_name         - e.g. astaxanthin | PHB | lipids | hydrogen | ethanol | null
  bioproduct_yield_value  - numeric
  bioproduct_yield_unit   - as written
  tea_metric              - LCOE | production_cost | NPV | IRR | payback_period | MFSP | as written
  tea_value               - numeric
  tea_unit                - $/kg | €/kg | $/GJ | $/MWh | M$ | as written
  data_source             - abstract | table | figure | text | supplementary
  extraction_confidence   - high | medium | low
  notes                   - ranges, unit conversions, control description, caveats, or null

Return ONLY valid JSON (array of objects). If no quantitative data exists, return [].

Chunk:
{chunk}
"""


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    """Load existing checkpoint {paper_name: [rows...]}."""
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text())
    return {}


def save_checkpoint(checkpoint: dict):
    """Persist checkpoint atomically."""
    CHECKPOINT_FILE.write_text(json.dumps(checkpoint, indent=2, default=str))


# ── Extraction helpers ────────────────────────────────────────────────────────

async def extract_from_chunk(chunk_content: str, paper_meta: dict) -> list[dict]:
    """Run LLM extraction on one chunk. Returns list of row dicts."""
    prompt = EXTRACTION_PROMPT.format(chunk=chunk_content[:4000])
    try:
        raw = await llm_model_func(
            prompt,
            system_prompt="You are a precise scientific data extraction assistant. Return only valid JSON.",
        )
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0]
        rows = json.loads(raw)
        if not isinstance(rows, list):
            return []
        for row in rows:
            row["doi"] = paper_meta.get("doi")
            row["article_title"] = paper_meta.get("title")
            row["publication_year"] = paper_meta.get("year")
            # journal not in papers_metadata.json — omitted
        return rows
    except Exception as e:
        logger.warning(f"Extraction failed for chunk: {e}")
        return []


async def process_paper(
    paper_name: str, paper_idx: int, all_chunks: dict, meta_lookup: dict
) -> list[dict]:
    """Extract all data rows from one paper."""
    paper_meta = meta_lookup.get(paper_name, {})

    paper_chunks = [
        v for v in all_chunks.values()
        if Path(v.get("file_path", "")).name == paper_name
    ]
    table_chunks = [c for c in paper_chunks if "Table Analysis:" in c.get("content", "")]
    other_chunks = [
        c for c in paper_chunks
        if "Table Analysis:" not in c.get("content", "")
        and any(kw in c.get("content", "") for kw in [
            "mL CH", "g VS", "methane", "VFA", "acetate", "yield",
            "HRT", "OLR", "$/", "€/", "cost", "efficiency", "removal",
            "ammonia", "inhibition", "LCA", "GWP", "LCOE", "NPV",
        ])
    ]

    logger.info(
        f"  {paper_name}: {len(table_chunks)} table chunks, "
        f"{len(other_chunks)} numerical text chunks"
    )

    all_rows = []

    for chunk in table_chunks:
        rows = await extract_from_chunk(chunk["content"], paper_meta)
        for row in rows:
            row.setdefault("data_source", "table")
            row.setdefault("extraction_confidence", "high")
        all_rows.extend(rows)

    # Cap non-table chunks at 20 per paper to control API cost
    for chunk in other_chunks[:20]:
        rows = await extract_from_chunk(chunk["content"], paper_meta)
        for row in rows:
            row.setdefault("data_source", "text")
            row.setdefault("extraction_confidence", "medium")
        all_rows.extend(rows)

    # Deduplicate on key metric combination
    seen = set()
    deduped = []
    for row in all_rows:
        key = (
            row.get("methane_yield_value"), row.get("methane_yield_unit"),
            row.get("total_vfa_value"), row.get("tea_value"),
            row.get("reactor_type"), row.get("primary_substrate"),
            row.get("temperature_value"), row.get("hrt_value"),
        )
        if key not in seen:
            seen.add(key)
            row["paper_id"] = f"p{paper_idx}"
            deduped.append(row)

    logger.info(f"  → {len(deduped)} rows extracted (after dedup)")
    return deduped


# ── Excel output ──────────────────────────────────────────────────────────────

def write_excel(all_rows: list[dict]):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Extraction"

    # Section colour bands (column index ranges, 1-based, inclusive)
    # 1-4:   paper_id, doi, article_title, publication_year
    # 5-8:   algae_species, algae_role, primary_substrate, co_substrate
    # 9-13:  process_category, reactor_type, feed_mode, scale, pretreatment_method
    # 14-23: temperature_value…duration_unit (10 condition cols)
    # 24-30: methane_yield_value…biogas_yield_unit (7 methane/biogas cols)
    # 31-33: total_vfa_value, total_vfa_unit, acetate_fraction_percent
    # 34-36: bioproduct_name, bioproduct_yield_value, bioproduct_yield_unit
    # 37-39: tea_metric, tea_value, tea_unit
    # 40-42: data_source, extraction_confidence, notes
    SECTION_COLOURS = {
        "metadata":    ("A1F7FF", range(1,  5)),
        "biology":     ("D4EDDA", range(5,  9)),
        "process":     ("FFF3CD", range(9,  14)),
        "conditions":  ("FCE4EC", range(14, 24)),
        "methane":     ("E8EAF6", range(24, 31)),
        "vfa":         ("F3E5F5", range(31, 34)),
        "bioproducts": ("FFF9C4", range(34, 37)),
        "tea":         ("E0F2F1", range(37, 40)),
        "quality":     ("ECEFF1", range(40, 43)),
    }

    header_font = Font(bold=True, size=10)

    # Write header row
    for col_idx, col_name in enumerate(COLUMNS, start=1):
        cell = ws.cell(row=1, column=col_idx, value=col_name)
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
        # Apply section fill
        for colour, col_range in SECTION_COLOURS.values():
            if col_idx in col_range:
                cell.fill = PatternFill("solid", fgColor=colour)
                break

    # Write data rows
    for row_idx, row_data in enumerate(all_rows, start=2):
        for col_idx, col_name in enumerate(COLUMNS, start=1):
            val = row_data.get(col_name)
            ws.cell(row=row_idx, column=col_idx, value=val)

    # Freeze header row
    ws.freeze_panes = "A2"

    # Auto-fit column widths (heuristic)
    for col_idx, col_name in enumerate(COLUMNS, start=1):
        col_letter = get_column_letter(col_idx)
        max_len = max(
            len(col_name),
            max(
                (len(str(row.get(col_name, "") or "")) for row in all_rows),
                default=0,
            ),
        )
        ws.column_dimensions[col_letter].width = min(max_len + 2, 40)

    wb.save(OUTPUT_XLSX)
    logger.info(f"Saved → {OUTPUT_XLSX}")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(papers: list[str], reset: bool = False):
    if reset and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        logger.info("Checkpoint cleared.")

    checkpoint = load_checkpoint()
    already_done = set(checkpoint.keys())

    papers_to_run = [p for p in papers if p not in already_done]
    skipped = [p for p in papers if p in already_done]

    if skipped:
        logger.info(f"Resuming — skipping {len(skipped)} already-extracted paper(s): {skipped}")

    logger.info("Loading rag_storage chunks...")
    all_chunks = json.loads((RAG_STORAGE_DIR / "kv_store_text_chunks.json").read_text())

    from _bootstrap import PAPERS_METADATA as meta_path
    meta_list = json.loads(meta_path.read_text())
    meta_lookup = {r["filename"]: r for r in meta_list}

    logger.info(f"Processing {len(papers_to_run)} paper(s) (of {len(papers)} total requested)...")

    # paper_idx continues from however many are already checkpointed
    start_idx = len(already_done) + 1

    for i, paper in enumerate(papers_to_run, start=start_idx):
        logger.info(f"[{i}/{len(papers)}] {paper}")
        rows = await process_paper(paper, i, all_chunks, meta_lookup)
        checkpoint[paper] = rows
        save_checkpoint(checkpoint)  # ← checkpoint saved after each paper
        logger.info(f"  Checkpoint saved ({len(checkpoint)}/{len(papers)} papers done)")

    # Compile all rows in original paper order
    all_rows = []
    for paper in papers:
        all_rows.extend(checkpoint.get(paper, []))

    write_excel(all_rows)
    logger.info(f"\nDone. {len(all_rows)} total rows across {len(papers)} papers → {OUTPUT_XLSX}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract structured data from rag_storage.")
    parser.add_argument("--all", action="store_true", help="Run on all ingested papers.")
    parser.add_argument("--paper", type=str, help="Run on a single paper filename.")
    parser.add_argument("--reset", action="store_true", help="Clear checkpoint and re-run from scratch.")
    args = parser.parse_args()

    if args.paper:
        papers = [args.paper]
    elif args.all:
        all_chunks = json.loads((RAG_STORAGE_DIR / "kv_store_text_chunks.json").read_text())
        papers = sorted({
            Path(v.get("file_path", "")).name
            for v in all_chunks.values()
            if v.get("file_path")
        })
        logger.info(f"Running on all {len(papers)} ingested papers.")
    else:
        papers = CURATED_PAPERS
        logger.info(f"Running on {len(papers)} curated papers.")

    asyncio.run(main(papers, reset=args.reset))
