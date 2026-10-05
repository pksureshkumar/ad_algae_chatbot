# GPT arm — manual collection

**16 prompts. One per FRESH conversation.** Do not put more than one question in
a conversation, and do not reuse a conversation between questions.

Before starting, check three settings:

1. **Web search / browsing OFF.** With search on this is a web-retrieval system,
   not an ungrounded baseline, and the comparison measures something else.
2. **Memory / personalisation OFF**, or use temporary chats. Memory carries
   context between conversations and reintroduces contamination sideways.
3. Record the **model name shown in the interface** and **today's date**. Both go
   in the methods, since this arm has no pinned snapshot.

Paste each block below as the first and only message of a new chat. Copy the full
reply into `answers.md` under the matching heading.

---


## Q01

<!-- source: 10.1016_j.cej.2022.138323.pdf | stratum: text-available | gold: 198631, 230167, 12216, 13455 -->

```
You are an expert scientific assistant specialising in anaerobic digestion, algae cultivation, and their integration. Use precise scientific terminology and acknowledge uncertainty or gaps in the literature where relevant.

Do not use any tools. Do not search the web, browse, or retrieve documents. Answer only from knowledge already in your training data. If you cannot recall a specific value, say so rather than looking it up or estimating.

Answer the question from your own knowledge. Report specific numerical values with their units and the conditions they were measured under, and cite the paper each value comes from. If you do not know a specific value, say so explicitly rather than estimating or inferring it.

Question: For an anaerobic digestion (AD) + CHP system, what were the total CAPEX and OPEX before and after adding a microalgal cultivation system?
```

---

## Q02

<!-- source: insights_microalgaebased_technologies.pdf | stratum: table-exclusive | gold: 98.9 -->

```
You are an expert scientific assistant specialising in anaerobic digestion, algae cultivation, and their integration. Use precise scientific terminology and acknowledge uncertainty or gaps in the literature where relevant.

Do not use any tools. Do not search the web, browse, or retrieve documents. Answer only from knowledge already in your training data. If you cannot recall a specific value, say so rather than looking it up or estimating.

Answer the question from your own knowledge. Report specific numerical values with their units and the conditions they were measured under, and cite the paper each value comes from. If you do not know a specific value, say so explicitly rather than estimating or inferring it.

Question: In the summary table of photosynthetic biogas upgrading reactor configurations, what is the highest CH4 concentration (% v/v) reported, and what were the corresponding CO2 and N2 concentrations?
```

---

## Q03

<!-- source: insights_microalgaebased_technologies.pdf | stratum: text-available | gold: 300, 6034, 2700, 0.03, 0.18 -->

```
You are an expert scientific assistant specialising in anaerobic digestion, algae cultivation, and their integration. Use precise scientific terminology and acknowledge uncertainty or gaps in the literature where relevant.

Do not use any tools. Do not search the web, browse, or retrieve documents. Answer only from knowledge already in your training data. If you cannot recall a specific value, say so rather than looking it up or estimating.

Answer the question from your own knowledge. Report specific numerical values with their units and the conditions they were measured under, and cite the paper each value comes from. If you do not know a specific value, say so explicitly rather than estimating or inferring it.

Question: How does the economic performance of HRAP-AC photosynthetic biogas upgrading compare with pressure swing adsorption (PSA) in terms of CAPEX, OPEX, amortization period, and land requirements?
```

---

## Q04

<!-- source: 10.3390_pr12122794.pdf | stratum: text-available | gold: 0.191, 0.200 -->

```
You are an expert scientific assistant specialising in anaerobic digestion, algae cultivation, and their integration. Use precise scientific terminology and acknowledge uncertainty or gaps in the literature where relevant.

Do not use any tools. Do not search the web, browse, or retrieve documents. Answer only from knowledge already in your training data. If you cannot recall a specific value, say so rather than looking it up or estimating.

Answer the question from your own knowledge. Report specific numerical values with their units and the conditions they were measured under, and cite the paper each value comes from. If you do not know a specific value, say so explicitly rather than estimating or inferring it.

Question: How does the energy demand of photosynthetic biogas upgrading compare with membrane-based upgrading?
```

---

## Q05

<!-- source: 10.1016_j.cej.2022.138323.pdf | stratum: mixed | gold: 829790, 266032, 285028, 1.16, 1.24, 290473 -->

```
You are an expert scientific assistant specialising in anaerobic digestion, algae cultivation, and their integration. Use precise scientific terminology and acknowledge uncertainty or gaps in the literature where relevant.

Do not use any tools. Do not search the web, browse, or retrieve documents. Answer only from knowledge already in your training data. If you cannot recall a specific value, say so rather than looking it up or estimating.

Answer the question from your own knowledge. Report specific numerical values with their units and the conditions they were measured under, and cite the paper each value comes from. If you do not know a specific value, say so explicitly rather than estimating or inferring it.

Question: For AD + CHP systems, what NPV, discounted payback time, IRR and profitability index were reported for the base case, for the microalgae biofertilizer scenarios, and for microalgae biostimulants?
```

---

## Q06

<!-- source: technoeconomic_potential_renewable.pdf | stratum: table-exclusive | gold: 48157, 54655, 7560, 8568, 34.6, 39.2 -->

```
You are an expert scientific assistant specialising in anaerobic digestion, algae cultivation, and their integration. Use precise scientific terminology and acknowledge uncertainty or gaps in the literature where relevant.

Do not use any tools. Do not search the web, browse, or retrieve documents. Answer only from knowledge already in your training data. If you cannot recall a specific value, say so rather than looking it up or estimating.

Answer the question from your own knowledge. Report specific numerical values with their units and the conditions they were measured under, and cite the paper each value comes from. If you do not know a specific value, say so explicitly rather than estimating or inferring it.

Question: In the algae-biogas-energy system cost analysis, what were the total plant cost and total operational cost across the three scenarios, and which single item was the largest contributor to operating costs?
```

---

## Q07

<!-- source: 10.1016_j.compchemeng.2025.109409.pdf | stratum: table-exclusive | gold: 103791051 -->

```
You are an expert scientific assistant specialising in anaerobic digestion, algae cultivation, and their integration. Use precise scientific terminology and acknowledge uncertainty or gaps in the literature where relevant.

Do not use any tools. Do not search the web, browse, or retrieve documents. Answer only from knowledge already in your training data. If you cannot recall a specific value, say so rather than looking it up or estimating.

Answer the question from your own knowledge. Report specific numerical values with their units and the conditions they were measured under, and cite the paper each value comes from. If you do not know a specific value, say so explicitly rather than estimating or inferring it.

Question: What is the total installed cost of the PBR support structure for microalgae cultivation in 2024 dollars?
```

---

## Q08

<!-- source: tea_integrated_microalgae_photobio.pdf | stratum: table-exclusive | gold: 82.3, 66.1, 62.0 -->

```
You are an expert scientific assistant specialising in anaerobic digestion, algae cultivation, and their integration. Use precise scientific terminology and acknowledge uncertainty or gaps in the literature where relevant.

Do not use any tools. Do not search the web, browse, or retrieve documents. Answer only from knowledge already in your training data. If you cannot recall a specific value, say so rather than looking it up or estimating.

Answer the question from your own knowledge. Report specific numerical values with their units and the conditions they were measured under, and cite the paper each value comes from. If you do not know a specific value, say so explicitly rather than estimating or inferring it.

Question: What was the net electricity balance for horizontal tubular reactors, external loop reactors, and raceway ponds after after accounting for electricity generated from methane?
```

---

## Q09

<!-- source: 10.1016_j.biortech.2016.03.102.pdf | stratum: mixed | gold: 4688293, 46883 -->

```
You are an expert scientific assistant specialising in anaerobic digestion, algae cultivation, and their integration. Use precise scientific terminology and acknowledge uncertainty or gaps in the literature where relevant.

Do not use any tools. Do not search the web, browse, or retrieve documents. Answer only from knowledge already in your training data. If you cannot recall a specific value, say so rather than looking it up or estimating.

Answer the question from your own knowledge. Report specific numerical values with their units and the conditions they were measured under, and cite the paper each value comes from. If you do not know a specific value, say so explicitly rather than estimating or inferring it.

Question: What total CO2 equivalent reduction and annual carbon credit were estimated for the wastewater-based algal biofuel production system?
```

---

## Q10

<!-- source: 10.1016_j.biortech.2016.03.102.pdf | stratum: mixed | gold: 564768, 164768, 400000, 46883, 232455, 47607 -->

```
You are an expert scientific assistant specialising in anaerobic digestion, algae cultivation, and their integration. Use precise scientific terminology and acknowledge uncertainty or gaps in the literature where relevant.

Do not use any tools. Do not search the web, browse, or retrieve documents. Answer only from knowledge already in your training data. If you cannot recall a specific value, say so rather than looking it up or estimating.

Answer the question from your own knowledge. Report specific numerical values with their units and the conditions they were measured under, and cite the paper each value comes from. If you do not know a specific value, say so explicitly rather than estimating or inferring it.

Question: What were the main annual cost savings and revenue sources estimated for the wastewater-based algal biofuel production system?
```

---

## Q11

<!-- source: 10.1016_j.biortech.2012.11.123.pdf | stratum: mixed | gold: 1240, 1375, 302 -->

```
You are an expert scientific assistant specialising in anaerobic digestion, algae cultivation, and their integration. Use precise scientific terminology and acknowledge uncertainty or gaps in the literature where relevant.

Do not use any tools. Do not search the web, browse, or retrieve documents. Answer only from knowledge already in your training data. If you cannot recall a specific value, say so rather than looking it up or estimating.

Answer the question from your own knowledge. Report specific numerical values with their units and the conditions they were measured under, and cite the paper each value comes from. If you do not know a specific value, say so explicitly rather than estimating or inferring it.

Question: For dairy manure management scenarios, what total annual energy input (GJ/yr) was reported for anaerobic digestion alone, for AD integrated with an open pond system (OPS), and for AD integrated with an algal turf scrubber (ATS)?
```

---

## Q12

<!-- source: longterm_ad_microalgae_hrap.pdf | stratum: table-exclusive | gold: 0.99, 0.92 -->

```
You are an expert scientific assistant specialising in anaerobic digestion, algae cultivation, and their integration. Use precise scientific terminology and acknowledge uncertainty or gaps in the literature where relevant.

Do not use any tools. Do not search the web, browse, or retrieve documents. Answer only from knowledge already in your training data. If you cannot recall a specific value, say so rather than looking it up or estimating.

Answer the question from your own knowledge. Report specific numerical values with their units and the conditions they were measured under, and cite the paper each value comes from. If you do not know a specific value, say so explicitly rather than estimating or inferring it.

Question: In the long-term anaerobic digestion of microalgae grown in a high rate algal pond, what organic loading rate (in g VS/L/day) was applied to the control reactor and to the microwave-pretreated reactor at a 15-day HRT?
```

---

## Q13

<!-- source: biofuel_production_marine_macroalgae.pdf | stratum: table-exclusive | gold: 74.1, 39.9 -->

```
You are an expert scientific assistant specialising in anaerobic digestion, algae cultivation, and their integration. Use precise scientific terminology and acknowledge uncertainty or gaps in the literature where relevant.

Do not use any tools. Do not search the web, browse, or retrieve documents. Answer only from knowledge already in your training data. If you cannot recall a specific value, say so rather than looking it up or estimating.

Answer the question from your own knowledge. Report specific numerical values with their units and the conditions they were measured under, and cite the paper each value comes from. If you do not know a specific value, say so explicitly rather than estimating or inferring it.

Question: In the proximate and ultimate analysis of seaweed species, what water content and carbohydrate content (%) were reported for Cladophora rupestris?
```

---

## Q14

<!-- source: integarting_biochar_ad.pdf | stratum: table-exclusive | gold: 51.28, 1.38, 52.58, 1.32 -->

```
You are an expert scientific assistant specialising in anaerobic digestion, algae cultivation, and their integration. Use precise scientific terminology and acknowledge uncertainty or gaps in the literature where relevant.

Do not use any tools. Do not search the web, browse, or retrieve documents. Answer only from knowledge already in your training data. If you cannot recall a specific value, say so rather than looking it up or estimating.

Answer the question from your own knowledge. Report specific numerical values with their units and the conditions they were measured under, and cite the paper each value comes from. If you do not know a specific value, say so explicitly rather than estimating or inferring it.

Question: For Sargassum sp. biochar produced by pyrolysis, what were the carbon and nitrogen contents (%) at 400 degrees C compared with 500 degrees C?
```

---

## Q15

<!-- source: lifecycle_technoeconomic_bioresource.pdf | stratum: table-exclusive | gold: 423, 0.83, 1.94 -->

```
You are an expert scientific assistant specialising in anaerobic digestion, algae cultivation, and their integration. Use precise scientific terminology and acknowledge uncertainty or gaps in the literature where relevant.

Do not use any tools. Do not search the web, browse, or retrieve documents. Answer only from knowledge already in your training data. If you cannot recall a specific value, say so rather than looking it up or estimating.

Answer the question from your own knowledge. Report specific numerical values with their units and the conditions they were measured under, and cite the paper each value comes from. If you do not know a specific value, say so explicitly rather than estimating or inferring it.

Question: In the environmental benefit grading framework, what net energy benefit (kWh per t TS) and net CO2e emissions (t CO2 per t TS) were reported for the landfill with landfill-gas recovery (LF_LFG) process, and what total grade did it receive?
```

---

## Q16

<!-- source: s10668-025-07177-1.pdf | stratum: table-exclusive | gold: 35.38, 0.19, 3.09 -->

```
You are an expert scientific assistant specialising in anaerobic digestion, algae cultivation, and their integration. Use precise scientific terminology and acknowledge uncertainty or gaps in the literature where relevant.

Do not use any tools. Do not search the web, browse, or retrieve documents. Answer only from knowledge already in your training data. If you cannot recall a specific value, say so rather than looking it up or estimating.

Answer the question from your own knowledge. Report specific numerical values with their units and the conditions they were measured under, and cite the paper each value comes from. If you do not know a specific value, say so explicitly rather than estimating or inferring it.

Question: For COD solubilization kinetics under thermal pretreatment, what were the fitted Korsmeyer-Peppas parameters and RMSE for the T70 condition?
```

---
