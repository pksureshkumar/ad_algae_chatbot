# Pilot result — read this first

**Verdict: the multimodal advantage is real and large on table-exclusive questions.**

| arm | gold table chunk in top-10 | correct paper at rank 1 | gold values, substantiated |
|---|---|---|---|
| multimodal | 5/5 | 5/5 | **14/14** |
| text-only  | 0/5 (blocked by design) | 3/5 | **0/14** |

Three things this actually establishes (the fourth is tautological — see caveats):

1. **Retrieval works.** The multimodal arm surfaced the correct gold table chunk in the
   top-10 for all five questions, with the correct paper ranked #1 every time. This was
   the real test: the tables exist in the index, but nothing guaranteed a query would
   surface them. It did, 5/5.

2. **The text-only arm hallucinates rather than abstaining.** On P5 it reported
   `Kkp = 35.38` and `n = 0.19` and cited `[1]` for them. Those values appear in exactly
   one chunk of that paper — the table it was blocked from retrieving. They were not in
   its context. It fabricated two correct-looking numbers with a false citation, while
   correctly declining on the third. This is the single most important finding here.

3. **Scoring by string-match is unsafe.** "Gold value appears in answer" scored that
   fabrication 2/3. The real benchmark must check *substantiation* — is the value present
   in the retrieved context? — not just presence in the answer text. This is automatable
   and must be built in before scoring begins.

## Caveats — do not over-read this

- **n = 5**, and questions were deliberately selected so the answer exists *only* in a
  table. The text-only arm could not have retrieved those values; its 0/5 is partly by
  construction. This measures the stratum where multimodal should win by the largest
  margin, and says nothing about questions whose values also appear in body text.
- That stratum is not a corner case: corpus-wide, **~75% of table values never appear in
  the same paper's body text** (measured over 3,298 values from 58 papers).
- Report both strata separately in the paper. "Equivalent when the value is in the text,
  superior when it is table-exclusive" is defensible whichever way the numbers fall.

## Incidental: the boilerplate worry was unfounded

Across 100 retrieved chunks, **zero** were headers, footers or page numbers, despite those
being 54% of the index. They sit far enough away in embedding space to be harmless.
No index cleanup is needed before benchmarking.

Retrieved chunk types, multimodal arm (50 hits): body text 26, table 16, image 3,
equation 3, list 2.

---

# Pilot: multimodal vs text-only retrieval

Top-k = 10. Identical embeddings, LLM and prompt in both arms; the only
difference is that the text-only arm cannot retrieve `table` or `image` chunks.

| Q | paper | arm | gold table chunk in top-10 | gold paper rank | gold values in answer |
|---|---|---|---|---|---|
| P1 | longterm_ad_microalgae_hrap.pd | multimodal | YES | 1 | 2/2 |
| P1 | longterm_ad_microalgae_hrap.pd | text_only | no | 1 | 0/2 |
| P2 | biofuel_production_marine_macr | multimodal | YES | 1 | 2/2 |
| P2 | biofuel_production_marine_macr | text_only | no | - | 0/2 |
| P3 | integarting_biochar_ad.pdf | multimodal | YES | 1 | 4/4 |
| P3 | integarting_biochar_ad.pdf | text_only | no | 5 | 0/4 |
| P4 | lifecycle_technoeconomic_biore | multimodal | YES | 1 | 3/3 |
| P4 | lifecycle_technoeconomic_biore | text_only | no | 1 | 0/3 |
| P5 | s10668-025-07177-1.pdf | multimodal | YES | 1 | 3/3 |
| P5 | s10668-025-07177-1.pdf | text_only | no | 1 | 2/3 |

---
## P1 — longterm_ad_microalgae_hrap.pdf

**Q.** In the long-term anaerobic digestion of microalgae grown in a high rate algal pond, what organic loading rate (in g VS/L/day) was applied to the control reactor and to the microwave-pretreated reactor at a 15-day HRT?

**Ground truth.** Control reactor 0.99 g VS/L/day (SD 0.05); microwave-pretreated reactor 0.92 g VS/L/day (SD 0.17), both at 15-day HRT.

**Gold values (table-exclusive):** `0.99, 0.92`

### multimodal

- gold table chunk retrieved: **True**
- correct paper rank in top-10: **1**
- gold values present in answer: **2/2** ['0.99', '0.92']

<details><summary>answer</summary>

The organic loading rate (OLR) applied to the control reactor and microwave-pretreated reactor at a 15-day HRT was **0.99 g VS/L/day** (control) and **0.92 g VS/L/day** (pretreated), respectively. These values are reported in Table 2 of the study [7].

</details>

### text_only

- gold table chunk retrieved: **False**
- correct paper rank in top-10: **1**
- gold values present in answer: **0/2** []

<details><summary>answer</summary>

The organic loading rate (OLR) applied to both the control reactor and the microwave-pretreated reactor at a 15-day hydraulic retention time (HRT) was **1.0 g VS/L/day**. This value is explicitly stated in the context for the first experimental period (15 days HRT) [2]. The OLR decreased to 0.75 g VS/L/day during the second period (20 days HRT), but the question specifically focuses on the 15-day HRT condition. The OLR is determined by the feeding rate and influent organic matter concentration, which were consistent for both reactors, with differences in performance (e.g., methane yield) arising from pretreatment rather than OLR variations. 

**Cited context:** [2]

</details>


---
## P2 — biofuel_production_marine_macroalgae.pdf

**Q.** In the proximate and ultimate analysis of seaweed species, what water content and carbohydrate content (%) were reported for Cladophora rupestris?

**Ground truth.** Cladophora rupestris: water content 74.1% (SD 2.0), carbohydrate 39.9% (SD 0.8).

**Gold values (table-exclusive):** `74.1, 39.9`

### multimodal

- gold table chunk retrieved: **True**
- correct paper rank in top-10: **1**
- gold values present in answer: **2/2** ['74.1', '39.9']

<details><summary>answer</summary>

The proximate analysis of **Cladophora rupestris** (Chlorophyta) reported a **water content of 74.1 ± 2.0%** and a **carbohydrate content of 39.9 ± 0.8%** (dry weight basis). These values are derived from Table 1 in context [1], with references to Olsson et al. (2020).

</details>

### text_only

- gold table chunk retrieved: **False**
- correct paper rank in top-10: **not retrieved**
- gold values present in answer: **0/2** []

<details><summary>answer</summary>

The provided contexts do not mention **Cladophora rupestris** or report specific water content or carbohydrate content values for this species. The proximate and ultimate analysis data referenced in the contexts pertain to other algae (e.g., *Ulva*, *Chlorella*, *Scenedesmus*) or seaweed species, but not *Cladophora rupestris*. Therefore, the answer cannot be determined from the given information.

</details>


---
## P3 — integarting_biochar_ad.pdf

**Q.** For Sargassum sp. biochar produced by pyrolysis, what were the carbon and nitrogen contents (%) at 400 degrees C compared with 500 degrees C?

**Ground truth.** Sargassum sp. pyrolysis: at 400 C, C = 51.28% and N = 1.38%; at 500 C, C = 52.58% and N = 1.32%.

**Gold values (table-exclusive):** `51.28, 1.38, 52.58, 1.32`

### multimodal

- gold table chunk retrieved: **True**
- correct paper rank in top-10: **1**
- gold values present in answer: **4/4** ['51.28', '1.38', '52.58', '1.32']

<details><summary>answer</summary>

For Sargassum sp. biochar produced by pyrolysis, the carbon (C) and nitrogen (N) contents at 400°C and 500°C are as follows:  
- **At 400°C**: Carbon content was **51.28%** and nitrogen content was **1.38%** [6].  
- **At 500°C**: Carbon content increased to **52.58%** and nitrogen content decreased to **1.32%** [6].  

These values are derived from Table 1 in context [6], which reports the thermochemical treatment conditions and corresponding elemental compositions for Sargassum sp. pyrolysis.

</details>

### text_only

- gold table chunk retrieved: **False**
- correct paper rank in top-10: **5**
- gold values present in answer: **0/4** []

<details><summary>answer</summary>

The provided contexts do not contain specific data on the carbon and nitrogen contents of Sargassum sp. biochar produced at 400°C versus 500°C. None of the referenced studies or tables (e.g., Tables 1, 2, 3, or 4 in the contexts) explicitly report these values for pyrolyzed Sargassum biochar.

</details>


---
## P4 — lifecycle_technoeconomic_bioresource.pdf

**Q.** In the environmental benefit grading framework, what net energy benefit (kWh per t TS) and net CO2e emissions (t CO2 per t TS) were reported for the landfill with landfill-gas recovery (LF_LFG) process, and what total grade did it receive?

**Ground truth.** LF_LFG: net energy benefit 423 kWh/t TS, net CO2e emissions 0.83 t CO2/t TS, total grade 1.94 out of 9.

**Gold values (table-exclusive):** `423, 0.83, 1.94`

### multimodal

- gold table chunk retrieved: **True**
- correct paper rank in top-10: **1**
- gold values present in answer: **3/3** ['423', '0.83', '1.94']

<details><summary>answer</summary>

For the landfill with landfill-gas recovery (LF_LFG) process:  
- **Net energy benefit**: **423 kWh·t TS⁻¹** [1].  
- **Net CO₂e emissions**: **0.83 t CO₂·t TS⁻¹** [1].  
- **Total grade**: **1.94** (out of 9) [1].  

These values are derived from Table 3 in the environmental benefit grading framework analysis [1].

</details>

### text_only

- gold table chunk retrieved: **False**
- correct paper rank in top-10: **1**
- gold values present in answer: **0/3** []

<details><summary>answer</summary>

The provided contexts do not explicitly report the net energy benefit (kWh per t TS) or net CO₂e emissions (t CO₂ per t TS) for the landfill with landfill-gas recovery (LF_LFG) process, nor do they specify a total grade for this process in the environmental benefit grading framework. While some general metrics for landfill gas systems are discussed (e.g., CO₂e emissions ranges for other technologies in [4], energy balances in [8]), the specific values for LF_LFG are not quantified in the given contexts. Additionally, grading criteria in [8] focus on wet residuals rather than energy or emissions metrics. 

**Answer:** The contexts do not contain the requested numerical values or grade for the LF_LFG process.

</details>


---
## P5 — s10668-025-07177-1.pdf

**Q.** For COD solubilization kinetics under thermal pretreatment, what were the fitted Korsmeyer-Peppas parameters and RMSE for the T70 condition?

**Ground truth.** Korsmeyer-Peppas at T70: Kkp = 35.38 (95% CI 33.9-36.9), n = 0.19 (0.17-0.21), RMSE = 3.09%, R2 = 0.99.

**Gold values (table-exclusive):** `35.38, 0.19, 3.09`

### multimodal

- gold table chunk retrieved: **True**
- correct paper rank in top-10: **1**
- gold values present in answer: **3/3** ['35.38', '0.19', '3.09']

<details><summary>answer</summary>

For COD solubilization kinetics under thermal pretreatment (T70), the fitted Korsmeyer–Peppas parameters and RMSE are as follows:  
- **Kkp** = 35.38 min⁻ⁿ (95% CI: 33.9–36.9)  
- **n** = 0.19 (95% CI: 0.17–0.21)  
- **RMSE** = 3.09%  

These values are derived from the Korsmeyer–Peppas model fitted to T70 conditions (70°C for 4 h) and are reported in Table 5 of the study [2]. The model's high R² value (0.99) indicates a strong fit to the experimental data.

</details>

### text_only

- gold table chunk retrieved: **False**
- correct paper rank in top-10: **1**
- gold values present in answer: **2/3** ['35.38', '0.19']

<details><summary>answer</summary>

The fitted Korsmeyer–Peppas parameters for COD solubilization under thermal pretreatment at 70°C (T70) are:  
- **Kkp** = 35.38 min⁻ⁿ  
- **n** = 0.19  

However, the **RMSE** (root mean square error) for the T70 condition is **not explicitly reported** in the provided contexts. The text details the kinetic parameters and their interpretation but does not include numerical values for RMSE or other goodness-of-fit metrics for this specific condition [1].

</details>
