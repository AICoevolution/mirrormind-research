---
dataset_name: s64-validation-v4
pretty_name: "S64 Validation Results (v4) – Symbolic 64 Transformation Framework"
license: cc-by-4.0
language:
  - en
tags:
  - symbolic-ai
  - human-ai-interaction
  - transformation-detection
  - embeddings
  - evaluation
task_categories:
  - other
papers:
  - title: "S64: A Symbolic Framework for Human-AI Meaning Negotiation"
    url: https://www.aicoevolution.com/s64-paper
    doi: 10.5281/zenodo.17784637
repository: https://github.com/AICoevolution/mirrormind-research
---

## Dataset Overview

This folder contains a ready-to-use Hugging Face dataset card for the **S64 v4 validation bundle**.  
It is designed to accompany the paper *"S64: A Symbolic Framework for Human-AI Meaning Negotiation"* [[website](https://www.aicoevolution.com/s64-paper), [Zenodo](https://doi.org/10.5281/zenodo.17784637)] and the research data hosted in `mirrormind-research`.

The dataset includes:

- **Synthetic baselines (B1–B8)** with ground truth and detection outputs.
- **Naturalistic baselines (B9–B10)** with consensus-focused analysis.
- **Analysis scripts** for recomputing TUS, precision/recall/F1, consensus metrics, and regenerating all figures from the paper.

## How to Use on Hugging Face

1. Create a new **Dataset** on Hugging Face (e.g. `AICoevolution/s64-validation-v4`).
2. Upload:
   - The `v4/` directory from `mirrormind-research`.
   - The `analysis_output/run_XXX/` directory you want to publish (typically the paper run).
   - The `scripts/` directory (analysis + visualization tools).
   - The `s64-paper.pdf` file.
3. Set this file as `README.md` in the HF dataset (or copy its contents into the HF README editor).

Once published, others can:

- Reproduce all metrics reported in the paper.
- Run alternative analyses over the same baselines.
- Benchmark new models or embeddings by adding their own result JSON files following the same naming conventions.

## Citation

If you use this dataset or the accompanying tools, please cite:

- Zenodo record: `10.5281/zenodo.17784637`
- Once available, the arXiv version of *"S64: A Symbolic Framework for Human-AI Meaning Negotiation"*.


