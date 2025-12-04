---
dataset_name: s64-validation-v4
pretty_name: "S64 Validation Results (v4) – Symbolic 64 Transformation Framework"
license: cc-by-4.0
viewer: false
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

# S64 Validation Dataset (v4)

This dataset contains the full **S64 v4 validation bundle** used in the paper *"S64: A Symbolic Framework for Human-AI Meaning Negotiation"*.

- **Paper**: [aicoevolution.com/s64-paper](https://www.aicoevolution.com/s64-paper)
- **Zenodo (archival)**: [10.5281/zenodo.17784637](https://doi.org/10.5281/zenodo.17784637)
- **GitHub mirror**: [AICoevolution/mirrormind-research](https://github.com/AICoevolution/mirrormind-research)

## What's Inside

| Folder | Description |
|--------|-------------|
| `v4/` | All baseline data and detection results (JSON) |
| `v3/` | Legacy Channel C results (pre-domain-tags) |
| `analysis_output/` | Computed metrics and figures from the paper |
| `scripts/` | Python tools for analysis and visualization |
| `examples/` | Quickstart scripts to explore the dataset |
| `s64-paper.pdf` | The full paper (also available on website/Zenodo) |

---

## Quick Start

### Option 1: Run the example scripts

```bash
# Clone or download this dataset, then:
cd examples
python s64_quickstart.py
```

This will:
1. List all available baselines
2. Inspect a sample result file
3. (Optionally) run the full analysis pipeline

### Option 2: Run the full analysis

```bash
cd scripts
python analyze_results_v4.py   # Creates analysis_output/run_XXX/
python visualize_results_v4.py # Generates all figures
```

---

## Dataset Structure

### V4 Baselines

```
v4/
├── baselines/
│   ├── synthetic/
│   │   ├── B1_surface_deception/
│   │   │   ├── baseline.json          ← Ground truth spec
│   │   │   ├── results_e5/            ← E5 embedding results
│   │   │   │   └── B1_{model}_e5_{uuid}.json
│   │   │   ├── results_ada02/         ← Ada-002 embedding results
│   │   │   └── results_cohere/        ← Cohere embedding results
│   │   └── ... (B2-B8)
│   │
│   └── naturalistic/
│       ├── B9_self_discovery_jjjs/
│       └── B10_self_discovery_AI/
```

### V3 Legacy (Channel C Only)

```
v3/
├── Baseline 01 - Surface Deception/
│   └── No Domains Axes/
│       └── B1_{model}_E5_{uuid}.json
└── ... (Baseline 02 - 08)
```

### File Naming Convention

**Format:** `B{N}_{model}_{embedding}_{uuid}.json`

| Component | Values |
|-----------|--------|
| `B{N}` | Baseline ID (B1–B10) |
| `{model}` | `dee` (DeepSeek), `gem` (Gemini), `gpt` (GPT-5.1), `haiku`, `sonnet`, `opus` |
| `{embedding}` | `e5`, `ada02`, `cohere` |
| `{uuid}` | 8-character unique ID |

**Examples:**
- `B1_dee_e5_a1b2c3d4.json` – Baseline 1, DeepSeek, E5 embeddings
- `B6_sonnet_cohere_badc113a.json` – Baseline 6, Sonnet, Cohere embeddings

---

## Ground Truth Definitions

### Synthetic Baselines

| Baseline | Label | Ground Truth | Type |
|----------|-------|--------------|------|
| B1 | Surface Deception | None (empty) | Deception |
| B2 | Implicit Transformation | M10, M11, M55, M34 | Transformation |
| B3 | Rapid Oscillation | M9, M33, M34 | Transformation |
| B4 | Stuck States | None (T1-only) | Deception |
| B5 | Nested Complexity | M16, M27, M41, M59-62 | Transformation |
| B6 | Explicit Transformation | M12, M56 | Transformation |
| B7 | Failed Transformation | None (reversal) | Deception |
| B8 | False Completion | None (claims w/o evidence) | Deception |

### Naturalistic Baselines

| Baseline | Description | Analysis Method |
|----------|-------------|-----------------|
| B9 | JJJS Self-Discovery | Consensus across models/embeddings |
| B10 | AI Self-Discovery | Consensus across models/embeddings |

---

## Examples Folder

The `examples/` folder contains ready-to-run scripts:

### `s64_quickstart.py`

A comprehensive quickstart that:
- Lists all synthetic and naturalistic baselines
- Inspects a sample result file (shows LLM model, status, detected paths)
- Can run the full analysis pipeline (uncomment the last line)

```bash
python examples/s64_quickstart.py
```

### `minimal_baseline_inspect.py`

A minimal script to peek at one baseline and one result file:

```bash
python examples/minimal_baseline_inspect.py
```

---

## Analysis Scripts

### `analyze_results_v4.py`

Comprehensive analysis of all v4 results.

**Features:**
- Automatic detection of synthetic vs. naturalistic baselines
- Precision, Recall, F1, TUS calculation for synthetic baselines
- Consensus analysis for naturalistic baselines
- Cross-embedding agreement metrics

**Output:**
- Creates incremental `run_XXX` directories (001, 002, 003...)
- `ANALYSIS_SUMMARY.txt` – Human-readable comprehensive summary
- `synthetic_baselines_analysis.csv` – Detailed metrics for each run
- `model_summary_synthetic.csv` – Aggregated model performance
- `naturalistic_baselines_analysis.json` – Consensus analysis

### `visualize_results_v4.py`

Generates all figures used in the paper.

**Synthetic Baseline Visualizations:**
- `llm_tus.png` – LLM Performance TUS ranking (Channel C)
- `embedding_tus.png` – Embedding TUS for Channel A and A+
- `llm_f1_heatmap.png` – LLM F1 scores by baseline
- `llm_precision_recall.png` – LLM precision-recall scatter
- `embedding_precision_recall.png` – Embedding precision-recall

**Naturalistic Baseline Visualizations:**
- `{B}_consensus_distribution.png` – Consensus category distribution
- `{B}_high_consensus_paths.png` – Top agreed-upon paths
- `{B}_summary.png` – Overall analysis summary

---

## Metrics Explained

### Synthetic Baselines

| Metric | Description |
|--------|-------------|
| **Precision** | What % of detections were correct |
| **Recall** | What % of ground truth was detected |
| **F1 Score** | Harmonic mean of precision and recall |
| **TUS** | Transformation Understanding Score (0–100%) |

**TUS Calculation:**
- *Positive baselines (B2, B3, B5, B6):* TUS = F1 score
- *Deception baselines (B1, B4, B7, B8):* TUS = 100% for correct rejection, 0% for any false positives

### Naturalistic Baselines

| Consensus Level | Agreement |
|-----------------|-----------|
| High | 75%+ |
| Moderate | 50–74% |
| Low | 25–49% |
| Outliers | <25% |

---

## Detection Channels

| Channel | Description |
|---------|-------------|
| **A** | Embedding-based detection (full transcript) |
| **A+** | Embedding-based detection (assistant→user pairs) |
| **B/B+** | LLM validation of A/A+ (disabled by default) |
| **C** | Independent LLM reasoning and extraction |

### Embedding Thresholds

| Backend | T1 Threshold | T2 Threshold | Dimensions |
|---------|--------------|--------------|------------|
| E5-Large (fine-tuned) | > 0.72 | > 0.75 | 768 |
| Ada-002 (OpenAI) | > 0.72 | > 0.75 | 1536 |
| Cohere embed-v3.0 | > 0.50 | > 0.55 | 1024 |

---

## Model Codes

| Code | Full Name |
|------|-----------|
| `dee` | DeepSeek |
| `gem` | Gemini 2.5 Pro |
| `gpt` | ChatGPT 5.1 |
| `haiku` | Claude Haiku 4.5 |
| `sonnet` | Claude Sonnet 4.5 |
| `opus` | Claude Opus 4.1 |

---

## Interpreting Results

### High Performance
- F1 > 0.8: Excellent detection
- TUS > 70: Strong transformation understanding

### Moderate Performance
- F1 0.5–0.8: Decent detection
- TUS 50–70: Acceptable understanding

### Low Performance
- F1 < 0.5: Poor detection
- TUS < 50: Weak understanding

---

## Requirements

```bash
pip install pandas numpy matplotlib seaborn
```

---

## Citation

If you use this dataset, please cite:

**Zenodo:**
```
Jimenez Sanchez, J. J. (2025). S64: A Symbolic Framework for Human-AI Meaning Negotiation.
Zenodo. https://doi.org/10.5281/zenodo.17784637
```

**BibTeX:**
```bibtex
@misc{jimenez2025s64,
  author = {Jimenez Sanchez, Juan Jacobo},
  title = {S64: A Symbolic Framework for Human-AI Meaning Negotiation},
  year = {2025},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.17784637},
  url = {https://doi.org/10.5281/zenodo.17784637}
}
```

---

## Links

- **Paper (HTML & PDF)**: [aicoevolution.com/s64-paper](https://www.aicoevolution.com/s64-paper)
- **Zenodo (archival)**: [10.5281/zenodo.17784637](https://doi.org/10.5281/zenodo.17784637)
- **GitHub**: [AICoevolution/mirrormind-research](https://github.com/AICoevolution/mirrormind-research)
- **Author**: research@aicoevolution.com
