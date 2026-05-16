# Deepfakes-and-Political-Propaganda-BSc-Thesis
### A Multimodal Study of Deepfake Detection on Romanian Social Media Using Consumer-Grade Multimodal LLMs

**BSc Project for a BSc in Data Science — IT University of Copenhagen — 2026**  
**Author:** Marin Giurgica

---

## Overview

This repository contains the code, prompts, results, and analysis for a BSc thesis investigating whether consumer-grade multimodal large language models (LLMs) can detect AI-generated and AI-manipulated political images on Romanian social media, and how prompt engineering, as well as the language of the prompt, affect their performance.

**36 experimental conditions:** 3 models × 6 prompting strategies × 2 languages (English / Romanian)

| | |
|---|---|
| **Models** | GPT-4o-mini, Gemini 2.5 Flash-Lite, Qwen 3.5 Vision |
| **Strategies** | S1 Simple · S2 Simple+Context · S3 Chain-of-Thought · S4 Persona · S5 CoT+Persona · S6 Few-shot |
| **Languages** | English prompts · Romanian prompts (post captions always in Romanian) |
| **Dataset** | 1,200 Romanian political Facebook images (600 fake / 600 real) |
| **Best result** | 87.83% Macro-F1 — Qwen 3.5 Vision · Romanian · S1 |

---

## Dataset

The image dataset cannot be released publicly due to Facebook's Terms of Service, GDPR right-to-erasure provisions, and copyright considerations. A full index of post URLs and ground truth labels is publicly available on Zenodo under CC BY 4.0:

**[https://zenodo.org/records/19686696](https://zenodo.org/records/19686696)**

---

## Repository Structure

```
├── Qwen/Gemini/GPT - output excels # The resulting excels for each strategy for each model
├── BSc Project arxiv Version       # The final report 
├── Metrics.xlsx                    # Aggregated metrics across all conditions
├── plots/                          # Various plots
├── Prompts_Final.md                # Final prompt text, readable format
├── Prompts_All_Versions.md         # Full prompt evolution history (v8–v14)
├── BatchAPI calls.py               # Build and submit batch jobs to model APIs
├── JsonL_to_Excel.py               # Convert batch output JSONL files to Excel
├── Clean_Outputs.py                # Parse, normalise, and flag UNPARSED rows
├── Thesis_Analysis.py              # Analysis and plotting scripts used in the thesis
├── Analysis.py                     # Exploratory analysis scripts
├── Public Dataset.xlsx             # Excel with URLs to Facebook images used in the experiment, to comply with GDPR, as well as author`s  annotation on Real/Fake
```

---

## Pipeline

This project, unfortunately, due to time limitations, requires multiple manual steps and has not been fully automated. However, the running logic of the attached scripts is:

1. **`BatchAPI calls 1-5.py`** — Encodes images, builds JSONL batch files for all 3 LLMs for strategies S1-S5.
2. **`BatchAPI S6.py`** — Same as above but for S6, few-shot prompting.
3. **`JsonL_to_Excel.py`** — Converts downloaded batch output JSONL files into structured Excel files.
4. **`Metrics.py`** — Reads all `*_clean.xlsx` files and computes per-condition metrics, resulting in `Metrics.xlsx` with pivot tables for each metric.
5.  **`Analysis.py`** — Builds various plots and analysis elements.
---

## Requirements

```bash
pip install pandas openpyxl scikit-learn requests matplotlib seaborn openpyxl wordcloud
```
---

## Prompting Strategies

| ID | Name | Description |
|---|---|---|
| S1 | Simple | Bare-minimum ask about image being real or AI-generated; no context or guidance |
| S2 | Simple + Context | Image + Romanian post caption injected |
| S3 | Chain-of-Thought | Step-by-step reasoning with explicit visual cue guidance |
| S4 | Persona | Model assigned forensics analyst role |
| S5 | CoT + Persona | S3 and S4 combined |
| S6 | Few-shot | 4 labelled examples (2 fake, 2 real) prepended to query |

Full prompt text, as well as other variations for all strategies can be found, in both languages, in:  'Prompts_Variations.pdf'.
---

## Key Results

| Model | Best Cohen`s Kappa | Condition |
|---|---|---|
| Qwen 3.5 Vision | **0.75** | Romanian · S1 - simple |
| Gemini 2.5 Flash-Lite | **0.7** | English · S6 - few shot |
| GPT-4o-mini | **0.62** | English · S6 - few-shot  |

**Cross-lingual finding:** Models have great troubles in correctly identifying AI-altered images. Few-shot prompting is the only prompting strategy that performs consistently across all 3 models. GPT, widely used in Romanian, achieves only 13% sensitivity in S1, and therefore should not be trusted.

**Novel finding:** AI image generators consistently fail to reproduce Romanian diacritics (ă, â, î, ș, ț) correctly — absent from English-language deepfake detection literature and directly usable as a detection heuristic.

---

## License

Code: [MIT License](LICENSE)  
Annotations (Zenodo): [CC BY 4.0](**https://creativecommons.org/licenses/by/4.0/**)  
