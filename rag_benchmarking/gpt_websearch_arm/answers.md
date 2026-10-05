# GPT arm — collected answers

Model shown in interface: __________    Date collected: __________
Reasoning setting: __________
Search off: [ ]    Memory off / temporary chat: [ ]

Paste each full reply inside the fences under its question.
The question is repeated here so you can confirm you are pasting the
reply to the right one — mis-mapping is the main risk when running
several chats at once.

---

## Q01

> For an anaerobic digestion (AD) + CHP system, what were the total CAPEX and OPEX before and after adding a microalgal cultivation system?

*stratum: text-available · source: 10.1016_j.cej.2022.138323.pdf*

```
**I do not know a verified, paper-specific set of total CAPEX and OPEX values for this before-and-after comparison.** I also cannot reliably identify the intended paper from the question alone, so I should not supply numerical values or attach an uncertain citation.

To report the comparison accurately, the study would need to provide:

| Cost metric | AD + CHP alone | AD + CHP with microalgal cultivation |
|---|---|---|
| Total capital expenditure (CAPEX; currency and cost year) | Value not known to me | Value not known to me |
| Total annual operating expenditure (OPEX; currency/year) | Value not known to me | Value not known to me |

The associated conditions matter: plant capacity and feedstock throughput, cultivation technology and scale, inclusion of algal harvesting and downstream processing, and whether OPEX represents **gross operating costs or net costs after revenue or avoided-cost credits**.

Please provide the paper title, DOI, or relevant excerpt to identify the specific comparison.
```

---

## Q02

> In the summary table of photosynthetic biogas upgrading reactor configurations, what is the highest CH4 concentration (% v/v) reported, and what were the corresponding CO2 and N2 concentrations?

*stratum: table-exclusive · source: insights_microalgaebased_technologies.pdf*

```
**I cannot confirm the highest CH₄ concentration in the specific summary table because its source has not been identified.** However, a directly verified result from **Franco-Morgado et al. (2018)** reports this matched gas composition:

| Component | Reported concentration (% v/v) |
|---|---:|
| **CH₄** | **99.1 ± 1** |
| **CO₂** | **0.5 ± 0.2** |
| **N₂** | **0.6 ± 0.5** |

These are the concentrations reported together for the upgraded biogas, not values combined from separate experimental conditions. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1385894718314931?utm_source=chatgpt.com)

**Operating conditions:** A pilot-scale **high-rate algal pond connected to a biogas absorption column**, operated with **biogas supplied during the night** and a **liquid-recirculation-to-biogas flow-rate ratio (L/G) of 0.5**. [HERO](https://hero.epa.gov/reference/7744736/?utm_source=chatgpt.com)

**Paper:** Franco-Morgado, M., Toledo-Cervantes, A., González-Sánchez, A., Lebrero, R., and Muñoz, R. (2018). “Integral (VOCs, CO₂, mercaptans and H₂S) photosynthetic biogas upgrading using innovative biogas and digestate supply strategies.” *Chemical Engineering Journal*, **354**, 363–369. DOI: **10.1016/j.cej.2018.08.026**. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1385894718314931?utm_source=chatgpt.com)

This verifies the **99.1% CH₄ / 0.5% CO₂ / 0.6% N₂** result, but the table itself is needed to establish whether it is the highest entry.
```

---

## Q03

> How does the economic performance of HRAP-AC photosynthetic biogas upgrading compare with pressure swing adsorption (PSA) in terms of CAPEX, OPEX, amortization period, and land requirements?

*stratum: text-available · source: insights_microalgaebased_technologies.pdf*

```
**HRAP–AC photosynthetic upgrading has higher reported CAPEX but lower OPEX than PSA, with land availability being a major constraint.** However, the published cost benchmarks should not be combined with payback and footprint figures from a different comparator and presented as a single, like-for-like HRAP–AC/PSA assessment. [Incover](https://incover-project.eu/sites/default/files/delivrables/bioresource_technology_technology_validation.pdf)

Here, **HRAP–AC** denotes a **high-rate algal pond coupled to an absorption column**.

| Metric | HRAP–AC photosynthetic upgrading | Pressure swing adsorption (PSA) | Source and conditions |
|---|---|---|---|
| **Specific CAPEX** | **€6,000 per (Nm³ h⁻¹)** of biogas-treatment capacity | **€2,700 per (Nm³ h⁻¹)** | Reported in **Rodero et al. (2019), Table 3**, as literature-based technology benchmarks—not costs measured in a matched operating trial. [Incover](https://incover-project.eu/sites/default/files/delivrables/bioresource_technology_technology_validation.pdf) |
| **Specific OPEX** | **€0.03 Nm⁻³** of biogas treated | **€0.18 Nm⁻³** | **Rodero et al. (2019), Table 3**. These are operating-cost benchmarks, not total levelized costs including capital recovery. [Incover](https://incover-project.eu/sites/default/files/delivrables/bioresource_technology_technology_validation.pdf) |
| **Amortization / capital recovery** | **Returns from year 5** were reported by the economic model, mainly because of **algal biofertilizer sales**, even without biomethane tax incentives. | **I cannot substantiate an equivalent PSA-specific amortization period from the primary studies cited here.** | **Toledo-Cervantes et al. (2017)** modeled upgrading **300 Nm³ h⁻¹** of biogas from anaerobic digestion of wastewater-treatment sludge. Its conventional comparator was **activated-carbon filtration plus water scrubbing**, not PSA. [ResearchGate](https://www.researchgate.net/publication/317252031_A_comparative_analysis_of_biogas_upgrading_technologies_Photosynthetic_vs_physicalchemical_processes) |
| **Land requirement** | Approximately **13.4 ha** of HRAP surface to treat **300 Nm³ h⁻¹**, assuming **0.2 m pond depth**. | **I do not know a verified, directly comparable PSA footprint from these papers.** | This is a **design-scale estimate**, reported in **Rodero et al. (2019), Section 3.5**, citing Toledo-Cervantes et al. (2017). [Incover](https://incover-project.eu/sites/default/files/delivrables/bioresource_technology_technology_validation.pdf) |

### Important qualification

The often-cited **1,860-fold greater land requirement** for photosynthetic upgrading comes from **Toledo-Cervantes et al. (2017)** and is relative to **activated-carbon filtration plus water scrubbing—not PSA**. Likewise, its five-year economic result depends on the modeled biofertilizer revenue; it is not a demonstrated, technology-wide repayment period. Consequently, I would **not report “five years for HRAP–AC versus nine years for PSA” as a verified primary-study comparison** without a PSA-specific source establishing the latter value and its assumptions. [ResearchGate](https://www.researchgate.net/publication/317252031_A_comparative_analysis_of_biogas_upgrading_technologies_Photosynthetic_vs_physicalchemical_processes)

The defensible conclusion is therefore **higher initial investment and substantial pond-area requirements, offset by lower reported operating costs and potential biomass revenue**. The evidence above supports the CAPEX/OPEX comparison, but does not establish an equivalent PSA amortization period or land requirement. [Incover](https://incover-project.eu/sites/default/files/delivrables/bioresource_technology_technology_validation.pdf)

**Papers:** Rodero, M. del R., et al. (2019). “Technology validation of photosynthetic biogas upgrading in a semi-industrial scale algal-bacterial photobioreactor.” *Bioresource Technology*, **279**, 43–49. DOI: **10.1016/j.biortech.2019.01.110**. [Incover](https://incover-project.eu/sites/default/files/delivrables/bioresource_technology_technology_validation.pdf)

Toledo-Cervantes, A., Estrada, J. M., Lebrero, R., and Muñoz, R. (2017). “A comparative analysis of biogas upgrading technologies: Photosynthetic vs physical/chemical processes.” *Algal Research*, **25**, 237–243. DOI: **10.1016/j.algal.2017.05.006**. [ResearchGate](https://www.researchgate.net/publication/317252031_A_comparative_analysis_of_biogas_upgrading_technologies_Photosynthetic_vs_physicalchemical_processes)
```

---

## Q04

> How does the energy demand of photosynthetic biogas upgrading compare with membrane-based upgrading?

*stratum: text-available · source: 10.3390_pr12122794.pdf*

```
**Photosynthetic biogas upgrading can have a lower electrical demand than membrane upgrading, but the advantage is not necessarily large.** In a pre-commercial demonstration, its reported consumption was close to the membrane benchmark used by the authors—not an order of magnitude lower. [ResearchGate](https://www.researchgate.net/publication/386583907_Pre-Commercial_Demonstration_of_a_Photosynthetic_Upgrading_Plant_Investment_and_Operating_Cost_Analysis)

### Reported energy demands

| Technology and source | Specific electrical consumption | Conditions and interpretation |
|---|---:|---|
| **Photosynthetic upgrading — Ruiz Palomar et al. (2024)** | **0.191 kWh m⁻³ of biogas treated** | A pre-commercial installation at a pig farm in Soria, Spain, using a raceway microalgal culture coupled to an absorption column. The energy calculation used a biogas throughput of **5.00 m³ h⁻¹**; the absorption column operated at a volumetric liquid-to-gas ratio of **1**. [ResearchGate](https://www.researchgate.net/publication/386583907_Pre-Commercial_Demonstration_of_a_Photosynthetic_Upgrading_Plant_Investment_and_Operating_Cost_Analysis) |
| **Membrane upgrading — benchmark in Ruiz Palomar et al. (2024)** | **0.200 kWh m⁻³ of biogas** | A literature-based comparator, **not a membrane installation tested alongside the photosynthetic system**. I do not know the specific feed composition, pressure, temperature, or throughput underlying this benchmark. [ResearchGate](https://www.researchgate.net/publication/386583907_Pre-Commercial_Demonstration_of_a_Photosynthetic_Upgrading_Plant_Investment_and_Operating_Cost_Analysis) |
| **Multistage membrane upgrading — Lems, Langerak and Dirkse, DMT technical paper** | **0.19–0.23 kWh m⁻³ of biogas** | Manufacturer-reported performance for the Carborex® MS system at a design methane slip of **0.3–0.7%** and a recycle flow of approximately **38–42%**. This is an industrial technical report rather than an independent, peer-reviewed comparison. [BiogasWorld](https://biogasworld.com/wp-content/uploads/2017/08/EAW-Next-generation-biogas-upgrading_20170322.pdf) |

### Why the comparison needs qualification

The photosynthetic system still consumed electricity for liquid circulation, biogas feeding, paddlewheel mixing, digestate and settler pumping, and controls. Its reported energy demand therefore was not zero despite using photosynthesis. In the membrane system, feed compression creates the separation driving force, while recycling gas to improve methane recovery adds recompression demand. [ResearchGate](https://www.researchgate.net/publication/386583907_Pre-Commercial_Demonstration_of_a_Photosynthetic_Upgrading_Plant_Investment_and_Operating_Cost_Analysis)

**The defensible conclusion is that this demonstrated photosynthetic installation had an electrical demand comparable to efficient membrane upgrading, with a slightly lower value than the selected membrane benchmark.** A general percentage saving cannot be established from these figures alone: an equivalent comparison would require matching gas-volume reference conditions, feed composition, methane purity and recovery, delivery pressure, and the equipment included in the energy accounting.

**Sources:** Ruiz Palomar, C., et al. (2024), “Pre-Commercial Demonstration of a Photosynthetic Upgrading Plant: Investment and Operating Cost Analysis,” *Processes*, **12**, 2794, DOI: **10.3390/pr12122794**; and Lems, R., Langerak, J., and Dirkse, E.H.M., “Next generation biogas upgrading using highly selective gas separation membranes: Showcasing the Poundbury project,” DMT technical paper. [UPM Open Access](https://oa.upm.es/89014/?utm_source=chatgpt.com)
```

---

## Q05

> For AD + CHP systems, what NPV, discounted payback time, IRR and profitability index were reported for the base case, for the microalgae biofertilizer scenarios, and for microalgae biostimulants?

*stratum: mixed · source: 10.1016_j.cej.2022.138323.pdf*

```
The study matching these scenarios is **Barbera et al. (2022)**, which evaluated a small-scale anaerobic digestion–combined heat and power (AD + CHP) plant integrated with microalgae cultivation. **I do not know the exact scenario-specific financial values well enough to report a complete numerical comparison reliably.** The accessible abstract establishes only the following results. [research.unipd.it](https://research.unipd.it/handle/11577/3454205)

| Financial indicator | What I can substantiate | What remains unverified |
|---|---|---|
| **Net present value (NPV)** | No exact value verified. | NPV for the base case, each biofertilizer scenario, and the biostimulant scenario. |
| **Discounted payback time** | **3.3–7.1 years** across the integrated-process scenarios. [research.unipd.it](https://research.unipd.it/handle/11577/3454205) | The base-case value and the exact assignment of values to individual product scenarios. |
| **Internal rate of return (IRR)** | No exact percentage verified. | IRR for each scenario. |
| **Profitability index (PI)** | **Greater than 1**, dimensionless, for the integrated process. [research.unipd.it](https://research.unipd.it/handle/11577/3454205) | The base-case PI and the exact PI for each microalgal product scenario. |

**Study conditions:** These were **techno-economic modeling results, not measured investment returns**, using a reference plant in **Northern Italy** treating the **organic fraction of municipal solid waste**. Microalgae cultivation utilized liquid digestate and CHP off-gases. [research.unipd.it](https://research.unipd.it/handle/11577/3454205)

The authors reported that selling microalgal biomass as **biofertilizer was less profitable than the AD process alone**, whereas selling it for **biostimulant applications could substantially increase profitability**. This qualitative conclusion does not establish the missing numerical indicators. [research.unipd.it](https://research.unipd.it/handle/11577/3454205)

**Reference:** Barbera, E., Bertucco, A., Nigam, K. D. P., and Kumar, S. (2022). “Techno-economic analysis of a micro-scale biogas plant integrated with microalgae cultivation for the treatment of organic municipal waste.” *Chemical Engineering Journal*, **450**, 138323. DOI: **10.1016/j.cej.2022.138323**. [ResearchGate](https://www.researchgate.net/publication/362369884_Techno-economic_analysis_of_a_micro-scale_biogas_plant_integrated_with_microalgae_cultivation_for_the_treatment_of_organic_municipal_waste)

I located the matching study, but not an accessible scenario-by-scenario financial results table; assigning exact values beyond those above would require guessing.
```

---

## Q06

> In the algae-biogas-energy system cost analysis, what were the total plant cost and total operational cost across the three scenarios, and which single item was the largest contributor to operating costs?

*stratum: table-exclusive · source: technoeconomic_potential_renewable.pdf*

```
In **Zamalloa et al. (2011)**, the three scenarios differed in assumed microalgal biomass productivity. **Table 3 reports costs per hectare**, whereas **Table 4 reports totals for the complete 400-ha algae cultivation–anaerobic digestion–combined heat and power (CHP) system**. These are modelled economic estimates, not experimentally measured costs. [Academia](https://www.academia.edu/6130948/The_techno_economic_potential_of_renewable_energy_through_the_anaerobic_digestion_of_microalgae)

| Parameter | Scenario 1 | Scenario 2 | Scenario 3 |
|---|---:|---:|---:|
| Assumed biomass productivity (t dry matter ha⁻¹ yr⁻¹) | 70 | 90 | 110 |
| **Total plant cost (€ ha⁻¹)** | **48,157** | **51,406** | **54,655** |
| **Total operational cost (€ ha⁻¹ yr⁻¹)** | **7,560** | **8,064** | **8,568** |
| Total plant cost, complete 400-ha system (€) | 19,262,775 | 20,562,322 | 21,861,974 |
| Total operating cost, complete 400-ha system (€ yr⁻¹) | 3,023,830 | 3,225,467 | 3,427,105 |

All cost figures above are reported directly in Tables 3–4. [Academia](https://www.academia.edu/6130948/The_techno_economic_potential_of_renewable_energy_through_the_anaerobic_digestion_of_microalgae)

**The largest single operating-cost item was “labor and overheads”: €2,967 ha⁻¹ yr⁻¹ in every scenario**, representing **39.2%, 36.8%, and 34.6%** of operating expenditure, respectively. The base model assumed **330 operating days per year**; land and pond levee/geotextile costs were excluded, and energy consumption was treated as parasitic demand rather than an operating-cost charge. [Academia](https://www.academia.edu/6130948/The_techno_economic_potential_of_renewable_energy_through_the_anaerobic_digestion_of_microalgae)

**Reference:** Zamalloa, C., Vulsteke, E., Albrecht, J., & Verstraete, W. (2011). *The techno-economic potential of renewable energy through the anaerobic digestion of microalgae.* **Bioresource Technology, 102**, 1149–1158. DOI: **10.1016/j.biortech.2010.09.017**. [Academia](https://www.academia.edu/6130948/The_techno_economic_potential_of_renewable_energy_through_the_anaerobic_digestion_of_microalgae)
```

---

## Q07

> What is the total installed cost of the PBR support structure for microalgae cultivation in 2024 dollars?

*stratum: table-exclusive · source: 10.1016_j.compchemeng.2025.109409.pdf*

```
**I cannot verify a reported cost in 2024 U.S. dollars.** The matching primary-source estimate is **US$67.7 million in 2016 dollars** for the installed PBR support structure, reported in Table 3.1 of **Zhu, Y., Jones, S. B., and Anderson, D. B. (2018), *Algae Farm Cost Model: Considerations for Photobioreactors*, PNNL-28201**. It should not be cited as a 2024-dollar figure. [PNNL](https://www.pnnl.gov/main/publications/external/technical_reports/PNNL-28201.pdf)

This is a **modeled capital-cost estimate, not an experimentally measured cost**, for a vertical hanging flat-bag photobioreactor farm with **1,000 acres of cultivation area**, an assumed annual-average productivity of **25 g ash-free dry biomass m⁻² d⁻¹**, **50-L working volume per bag**, and a **3-day hydraulic retention time**. [PNNL](https://www.pnnl.gov/main/publications/external/technical_reports/PNNL-28201.pdf)

The source establishes the original cost and design basis, but not the requested 2024-dollar amount. I do not know a verified 2024-dollar value and would not substitute an undocumented inflation adjustment.
```

---

## Q08

> What was the net electricity balance for horizontal tubular reactors, external loop reactors, and raceway ponds after after accounting for electricity generated from methane?

*stratum: table-exclusive · source: tea_integrated_microalgae_photobio.pdf*

```
**Harun et al. (2011, Table 4)** reported the following balances after crediting **70.7 GWh yr⁻¹ of electricity generated from methane**. Here, net balance is generation minus cultivation electricity consumption: negative values indicate deficits; positive values indicate surpluses. [Academia](https://www.academia.edu/10490801/Technoeconomic_analysis_of_an_integrated_microalgae_photobioreactor_biodiesel_and_biogas_production_facility)

| Cultivation system | Electricity consumption (GWh yr⁻¹) | Net electricity balance (GWh yr⁻¹) |
|---|---:|---:|
| Horizontal tubular reactors | 153.0 | **−82.3 — deficit**. [Academia](https://www.academia.edu/10490801/Technoeconomic_analysis_of_an_integrated_microalgae_photobioreactor_biodiesel_and_biogas_production_facility) |
| External loop reactors | 136.8 | **−66.1 — deficit**. [Academia](https://www.academia.edu/10490801/Technoeconomic_analysis_of_an_integrated_microalgae_photobioreactor_biodiesel_and_biogas_production_facility) |
| Raceway ponds | 8.7 | **+62.0 — surplus**. [Academia](https://www.academia.edu/10490801/Technoeconomic_analysis_of_an_integrated_microalgae_photobioreactor_biodiesel_and_biogas_production_facility) |

**Basis and boundary:** These were **theoretical estimates, not experimentally measured balances**, for production of **50 kt yr⁻¹ of dry *Tetraselmis suecica* biomass**. The methane-derived electricity credit used the study’s **whole-biomass digestion scenario**, rather than digestion of only lipid-extracted residues. These balances are after cultivation but **before downstream dewatering, lipid extraction, and transesterification**. [Academia](https://www.academia.edu/10490801/Technoeconomic_analysis_of_an_integrated_microalgae_photobioreactor_biodiesel_and_biogas_production_facility)

**Paper:** Harun, R., et al. (2011). “Technoeconomic analysis of an integrated microalgae photobioreactor, biodiesel and biogas production facility.” *Biomass and Bioenergy*, **35**(1), 741–747. DOI: **10.1016/j.biombioe.2010.10.007**. [Academia](https://www.academia.edu/10490801/Technoeconomic_analysis_of_an_integrated_microalgae_photobioreactor_biodiesel_and_biogas_production_facility)
```

---

## Q09

> What total CO2 equivalent reduction and annual carbon credit were estimated for the wastewater-based algal biofuel production system?

*stratum: mixed · source: 10.1016_j.biortech.2016.03.102.pdf*

```
For the wastewater-based algal biofuel system evaluated by **Xin et al. (2016)**, Table 4 reports an estimated **total CO₂-equivalent reduction of 4,688,293 kg CO₂-eq yr⁻¹** and corresponding **annual carbon-credit revenue of US$46,883 yr⁻¹**. The calculation assumed **320 operating days per year** and a carbon-credit price of **US$10 per tonne of CO₂**. [ScienceDirect](https://www.sciencedirect.com/science/article/am/pii/S0960852416304059?utm_source=chatgpt.com)

The system used **municipal wastewater centrate for algal cultivation**. These figures are modeled estimates of annual emissions reduction and potential revenue—not measured abatement or evidence of actual carbon-credit sales. The study’s accounting combined two carbon-removal contributions and subtracted emissions associated with algal biofuel production. [ScienceDirect](https://www.sciencedirect.com/science/article/am/pii/S0960852416304059?utm_source=chatgpt.com)

**Reference:** Xin, C., Addy, M. M., Zhao, J., et al. (2016). *Comprehensive techno-economic analysis of wastewater-based algal biofuel production: A case study.* **Bioresource Technology, 211**, 584–593. DOI: **10.1016/j.biortech.2016.03.102**. [PubMed](https://pubmed.ncbi.nlm.nih.gov/27039331/?utm_source=chatgpt.com)
```

---

## Q10

> What were the main annual cost savings and revenue sources estimated for the wastewater-based algal biofuel production system?

*stratum: mixed · source: 10.1016_j.biortech.2016.03.102.pdf*

```
For the **Saint Paul, Minnesota case study by Xin et al. (2016)**, the estimates assumed **5,420 tonnes of centrate wastewater/day**, **320 operating days/year**, and **10.298 tonnes of dry algal biomass/day**. These were **modeled economic benefits**, not measured commercial revenues. [Academia](https://www.academia.edu/56101723/Comprehensive_techno_economic_analysis_of_wastewater_based_algal_biofuel_production_A_case_study)

| Cost saving or revenue source | Estimated annual value | Basis |
|---|---:|---|
| **Wastewater-treatment savings** | **US$564,768/year** | US$164,768 in operating savings plus US$400,000 in annualized H₂S-control cover-cost savings. [Academia](https://www.academia.edu/56101723/Comprehensive_techno_economic_analysis_of_wastewater_based_algal_biofuel_production_A_case_study) |
| **Avoided cultivation-nutrient purchases** | **Approximately US$550,000/year** | Wastewater supplies nutrients otherwise requiring purchase; reported separately in the discussion. [Academia](https://www.academia.edu/56101723/Comprehensive_techno_economic_analysis_of_wastewater_based_algal_biofuel_production_A_case_study) |
| **Bio-oil sales** | **US$6,908 × X/year** | *X* is the bio-oil selling price in US$/barrel; Table 5 does not specify a fixed price. [Academia](https://www.academia.edu/56101723/Comprehensive_techno_economic_analysis_of_wastewater_based_algal_biofuel_production_A_case_study) |
| **Phosphorus-fertilizer sales** | **US$232,455/year** | Sale of phosphorus-rich ash remaining after biochar combustion. [Academia](https://www.academia.edu/56101723/Comprehensive_techno_economic_analysis_of_wastewater_based_algal_biofuel_production_A_case_study) |
| **Syngas sales** | **US$47,607/year** | Sale of pyrolysis syngas as heating fuel. [Academia](https://www.academia.edu/56101723/Comprehensive_techno_economic_analysis_of_wastewater_based_algal_biofuel_production_A_case_study) |
| **Carbon credits** | **US$46,883/year** | Assumed carbon value of US$10/tonne CO₂-equivalent. [Academia](https://www.academia.edu/56101723/Comprehensive_techno_economic_analysis_of_wastewater_based_algal_biofuel_production_A_case_study) |

The cover-cost saving assumed a **20% reduction in a US$40 million investment, annualized over 20 years**. The nutrient saving is **not an additional revenue line in Table 5** and should not be double-counted where purchased nutrients are already excluded from operating costs. [Academia](https://www.academia.edu/56101723/Comprehensive_techno_economic_analysis_of_wastewater_based_algal_biofuel_production_A_case_study)

**Reference:** Xin, C., et al. (2016). “Comprehensive techno-economic analysis of wastewater-based algal biofuel production: A case study.” *Bioresource Technology*, **211**, 584–593. DOI: **10.1016/j.biortech.2016.03.102**. [experts.umn.edu](https://experts.umn.edu/en/publications/comprehensive-techno-economic-analysis-of-wastewater-based-algal-/)
```

---

## Q11

> For dairy manure management scenarios, what total annual energy input (GJ/yr) was reported for anaerobic digestion alone, for AD integrated with an open pond system (OPS), and for AD integrated with an algal turf scrubber (ATS)?

*stratum: mixed · source: 10.1016_j.biortech.2012.11.123.pdf*

```
**I do not know the exact total annual energy-input values for AD alone, AD + OPS, or AD + ATS with sufficient confidence to report them, and I could not verify them from the accessible source material.** I therefore will not estimate or reconstruct those values.

The study matching these scenarios is **Zhang, Y., White, M. A., and Colosi, L. M. (2013). “Environmental and economic assessment of integrated systems for dairy manure treatment coupled with algae bioenergy production.” *Bioresource Technology*, 130, 486–494. DOI: 10.1016/j.biortech.2012.11.123.** [pubmed.ncbi.nlm.nih.gov](https://pubmed.ncbi.nlm.nih.gov/23313697/)

It compares anaerobic digestion with land application of liquid digestate against AD with liquid-digestate recycling to either an **open-pond algae cultivation system (OPS)** or an **algal turf scrubber (ATS)**. These are **life-cycle assessment estimates**, rather than directly measured annual energy consumption. The accessible abstract confirms the configurations but does not provide their individual total annual energy inputs in **GJ yr⁻¹**; its net-energy-output results are not substitutes for the requested input totals. [pubmed.ncbi.nlm.nih.gov](https://pubmed.ncbi.nlm.nih.gov/23313697/)
```

---

## Q12

> In the long-term anaerobic digestion of microalgae grown in a high rate algal pond, what organic loading rate (in g VS/L/day) was applied to the control reactor and to the microwave-pretreated reactor at a 15-day HRT?

*stratum: table-exclusive · source: longterm_ad_microalgae_hrap.pdf*

```
At a **15-day hydraulic retention time (HRT)**, Passos et al. (2014) reported the following average organic loading rates in **Table 2**:

| Reactor | Organic loading rate |
|---|---:|
| Control, fed untreated microalgal biomass | **0.99 g VS/L/day** |
| Fed microwave-pretreated microalgal biomass | **0.92 g VS/L/day** |

Here, VS denotes volatile solids. These are the reported experimental averages, not values calculated from the HRT. [Academia](https://www.academia.edu/29777537/Long_term_anaerobic_digestion_of_microalgae_grown_in_HRAP_for_wastewater_treatment_Effect_of_microwave_pretreatment)

Both digesters operated at **35 °C**, with a **1.5 L working volume**, using microalgal biomass grown in a wastewater-treatment high-rate algal pond. The pretreatment consisted of microwave irradiation at **900 W for 3 minutes** before feeding the biomass to the digester. [Academia](https://www.academia.edu/29777537/Long_term_anaerobic_digestion_of_microalgae_grown_in_HRAP_for_wastewater_treatment_Effect_of_microwave_pretreatment)

**Reference:** Passos, F., Hernández-Mariné, M., García, J., & Ferrer, I. (2014). “Long-term anaerobic digestion of microalgae grown in HRAP for wastewater treatment. Effect of microwave pretreatment.” *Water Research*, **49**, 351–359. DOI: **10.1016/j.watres.2013.10.013**. [pubmed.ncbi.nlm.nih.gov](https://pubmed.ncbi.nlm.nih.gov/24183557/)
```

---

## Q13

> In the proximate and ultimate analysis of seaweed species, what water content and carbohydrate content (%) were reported for Cladophora rupestris?

*stratum: table-exclusive · source: biofuel_production_marine_macroalgae.pdf*

```
For *Cladophora rupestris*, the reported values correspond to **74.1% water content** and **39.9% carbohydrate content**, expressed on different mass bases. Olsson, Toth, and Albers (2020), Table 1, reported the following measurements: [D-NB](https://d-nb.info/1214349137/34)

| Component | Content expressed as a percentage | Measurement basis | Original reported value |
|---|---:|---|---:|
| Water | **74.1 ± 2.0%** | Wet weight | 741 ± 20 g kg⁻¹ wet biomass |
| Carbohydrates | **39.9 ± 0.8%** | Dry weight | 399 ± 8 g kg⁻¹ dry biomass |

The percentages above are direct unit conversions of the published measurements. The ± values represent standard deviations of **technical replicates**, not biological variability: three replicates for water and two for carbohydrates. [D-NB](https://d-nb.info/1214349137/34)

**Sampling and analytical conditions:** The seaweed was collected on **1 July 2014**, near Tjärnö Marine Laboratory on Sweden’s west coast. Water content was determined by weighing samples before and after freeze-drying; carbohydrates were quantified using the MBTH colorimetric assay following two-step sulfuric-acid hydrolysis. [D-NB](https://d-nb.info/1214349137/34)

**Reference:** Olsson, J., Toth, G. B., & Albers, E. (2020). “Biochemical composition of red, green and brown seaweeds on the Swedish west coast.” *Journal of Applied Phycology*, **32**, 3305–3317. DOI: **10.1007/s10811-020-02145-w**. [link.springer.com](https://link.springer.com/article/10.1007/s10811-020-02145-w)
```

---

## Q14

> For Sargassum sp. biochar produced by pyrolysis, what were the carbon and nitrogen contents (%) at 400 degrees C compared with 500 degrees C?

*stratum: table-exclusive · source: integarting_biochar_ad.pdf*

```
Farobie et al. (2022) reported the following carbon and nitrogen contents for **biochar produced by slow pyrolysis of *Sargassum* sp.**, expressed as **weight percentages on a dry, ash-free basis** (Table 1). [Repo Dosen](https://repo-dosen.ulm.ac.id/bitstream/handle/123456789/24267/Apip%20Bid%20B%20A-15.pdf?isAllowed=y\&sequence=1)

| Pyrolysis temperature | Carbon content (wt%, dry ash-free) | Nitrogen content (wt%, dry ash-free) |
|---|---:|---:|
| **400 °C** | **51.28 ± 0.06%** | **1.38 ± 0.02%** |
| **500 °C** | **52.58 ± 0.06%** | **1.32 ± 0.01%** |

The experiments used dried, ground biomass in a stainless-steel batch reactor, purged with nitrogen to establish an oxygen-free atmosphere, and heated at **30 °C/min**. Carbon and nitrogen were measured using a **LECO CHN628 elemental analyzer**. The study tested holding times of 10, 30, and 50 minutes, but Table 1 does not specify which holding time corresponds to these elemental-composition values. The ± values above are reproduced as reported. [Repo Dosen](https://repo-dosen.ulm.ac.id/bitstream/handle/123456789/24267/Apip%20Bid%20B%20A-15.pdf?isAllowed=y\&sequence=1)

**Reference:** Farobie, O., Amrullah, A., Bayu, A., Syaftika, N., Anis, L. A., and Hartulistiyoso, E. (2022). “In-depth study of bio-oil and biochar production from macroalgae *Sargassum* sp. via slow pyrolysis.” *RSC Advances*, **12**, 9567–9578. DOI: **10.1039/D2RA00702A**. [Repo Dosen](https://repo-dosen.ulm.ac.id/bitstream/handle/123456789/24267/Apip%20Bid%20B%20A-15.pdf?isAllowed=y\&sequence=1)
```

---

## Q15

> In the environmental benefit grading framework, what net energy benefit (kWh per t TS) and net CO2e emissions (t CO2 per t TS) were reported for the landfill with landfill-gas recovery (LF_LFG) process, and what total grade did it receive?

*stratum: table-exclusive · source: lifecycle_technoeconomic_bioresource.pdf*

```
For **landfill with landfill-gas recovery (LF_LFG)**, **Table 3 of Clack, Rajagopal, and Hoek (2024)** reports the following values in the environmental benefit grading framework. [Nature](https://www.nature.com/articles/s41545-024-00314-9/tables/3?error=cookies_not_supported)

| Metric | Reported value |
|---|---:|
| Net energy benefit | **+423 kWh per t TS** |
| Net CO₂-equivalent emissions | **+0.83 t CO₂e per t TS** |
| Total environmental benefit grade | **1.94 out of 9** |

The total grade comprises **0.00** for residual disposal, **1.29** for net energy benefit, and **0.65** for net CO₂e emissions. Thus, the process provides a positive net energy benefit but remains a net greenhouse-gas emitter. [Nature](https://www.nature.com/articles/s41545-024-00314-9/tables/3?error=cookies_not_supported)

**Basis and conditions:** These are **modeled life-cycle results, not direct experimental measurements**, normalized per metric tonne of initial dry total solids (TS) managed. The assessment assumed **80% landfill-gas recovery**, **55% electricity-generation conversion efficiency** for recovered landfill gas, and **160 km transportation to final disposal**. [Nature](https://www.nature.com/articles/s41545-024-00314-9)

**Reference:** Clack, K., Rajagopal, D., & Hoek, E. M. V. (2024). *Life cycle and techno-economic assessment of bioresource production from wastewater*. **npj Clean Water, 7**, Article 22. DOI: **10.1038/s41545-024-00314-9**. [Nature](https://www.nature.com/articles/s41545-024-00314-9)
```

---

## Q16

> For COD solubilization kinetics under thermal pretreatment, what were the fitted Korsmeyer-Peppas parameters and RMSE for the T70 condition?

*stratum: table-exclusive · source: s10668-025-07177-1.pdf*

```
For **T70—thermal pretreatment at 70 °C for 4 h**, using wastewater-derived microalgal biomass at **3.5% total solids**—Lamberti et al. (2026) fitted the Korsmeyer–Peppas model \(Q_t=K_{kp}t^n\), with time \(t\) in minutes. [Springer](https://link.springer.com/article/10.1007/s10668-025-07177-1)

The **Table 5** results were: [Springer](https://link.springer.com/content/pdf/10.1007/s10668-025-07177-1.pdf)

| Parameter or fit metric | Reported value | 95% confidence interval |
|---|---:|---:|
| Kinetic constant, \(K_{kp}\) | **35.38 %·min\(^{-n}\)**¹ | 33.9–36.9 |
| Diffusion exponent, \(n\) | **0.19**, dimensionless | 0.17–0.21 |
| Root mean square error, RMSE | **3.09%** | Not reported |

¹ **Unit caveat:** The Methods define \(K_{kp}\) in **%·min\(^{-n}\)**, whereas the Results text gives **min\(^{-n}\)**. The table above follows the Methods’ unit definition; the numerical value is unchanged. [Springer](https://link.springer.com/article/10.1007/s10668-025-07177-1)

**Source:** Lamberti, G. G., et al. (2026). “Comparative assessment of thermal and alkaline pretreatments for enhanced solubilization of wastewater-derived microalgal biomass.” *Environment, Development and Sustainability*. DOI: **10.1007/s10668-025-07177-1**. [Springer](https://link.springer.com/article/10.1007/s10668-025-07177-1)
```

---
