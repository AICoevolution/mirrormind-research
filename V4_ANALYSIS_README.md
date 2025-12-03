# S64 Validation Analysis - V4 Structure

## Overview

Comprehensive analysis and visualization tools for S64 validation results with support for:
- **Synthetic baselines (B1-B8)**: Ground truth comparison with precision/recall/F1/TUS metrics
- **Naturalistic baselines (B9+)**: Consensus analysis across models and embeddings

## V3 Structure (Legacy - Channel C Only)

The `v3/` directory contains **Channel C results only**, generated before the adoption of the domain-tags approach. These files represent independent LLM reasoning and extraction without domain axis annotations.

```
v3/
├── Baseline 01 - Surface Deception/
│   ├── No Domains Axes/
│   │   └── B1_{model}_E5_{uuid}.json
│   └── ...
├── Baseline 02 - Implicit Transformation/
│   ├── No Domains Axes/
│   │   └── ...
│   └── ...
└── ... (Baseline 03 - Baseline 08)
```

**Note:** Only the `No Domains Axes/` subfolders are included in this research bundle. These represent the raw Channel C outputs prior to the introduction of domain-tagged analysis in v4.

## V4 Structure

```
v4/
├── baselines/
│   ├── synthetic/
│   │   ├── B1_surface_deception/
│   │   │   ├── baseline.json
│   │   │   ├── results_e5/
│   │   │   │   └── B1_{model}_e5_{uuid}.json
│   │   │   ├── results_ada02/
│   │   │   │   └── B1_{model}_ada02_{uuid}.json
│   │   │   └── results_cohere/
│   │   │       └── B1_{model}_cohere_{uuid}.json
│   │   └── ... (B2-B8)
│   │
│   └── naturalistic/
│       ├── B9_self_discovery_jjjs/
│       │   ├── baseline.json
│       │   ├── results_e5/
│       │   ├── results_ada02/
│       │   └── results_cohere/
│       └── B10_self_discovery_AI/
│           └── ...
```

## File Naming Convention

**Format:** `B{N}_{model}_{embedding}_{uuid}.json`

**Components:**
- `B{N}`: Baseline ID (B1, B2, ..., B10)
- `{model}`: Model code (dee, gem, gpt, haiku, sonnet, opus)
- `{embedding}`: Embedding backend (e5, ada02, cohere)
- `{uuid}`: 8-character unique identifier

**Examples:**
- `B1_dee_e5_a1b2c3d4.json` - Baseline 1, DeepSeek, E5 embeddings
- `B5_gpt_ada02_8902e40b.json` - Baseline 5, GPT-5.1, Ada-002 embeddings
- `B9_sonnet_cohere_badc113a.json` - Baseline 9, Sonnet, Cohere embeddings

## Scripts

### 1. `analyze_results_v4.py`

**Purpose:** Comprehensive analysis of all v4 results

**Features:**
- Automatic detection of synthetic vs. naturalistic baselines
- Precision, Recall, F1, TUS calculation for synthetic baselines
- Consensus analysis for naturalistic baselines
- Cross-embedding agreement metrics
- Model and embedding backend comparisons

**Output:**
- Creates incremental `run_XXX` directories (001, 002, 003...)
- `ANALYSIS_SUMMARY.txt` - **Human-readable comprehensive summary**
- `synthetic_baselines_analysis.csv` - Detailed metrics for each run
- `model_summary_synthetic.csv` - Aggregated model performance
- `naturalistic_baselines_analysis.json` - Consensus analysis

**Usage:**
```bash
python analyze_results_v4.py
# Creates: analysis_output/run_001/
# Next run: analysis_output/run_002/
# etc.
```

**Metrics Explained:**

**Synthetic Baselines:**
- **Precision**: What % of detections were correct
- **Recall**: What % of ground truth was detected
- **F1 Score**: Harmonic mean of precision and recall
- **TUS (Transformation Understanding Score)**: Composite 0-100% metric measuring detection accuracy calibrated to baseline type:
  - *Positive baselines (B2, B3, B5, B6):* TUS = F1 score, rewarding balanced precision and recall
  - *Deception baselines (B1, B4, B7, B8):* TUS = 100% for correct rejection (zero paths detected), 0% for any false positives
  - *Aggregate:* TUS = (1/N) × Σ TUS_i × 100%, where N = number of baselines evaluated
  - Separate TUS scores are computed for Channel C (LLM reasoning) and Channels A/A+ (embedding detection)

**Naturalistic Baselines:**
- **High Consensus**: Paths detected by 75%+ methods
- **Moderate Consensus**: 50-74% agreement
- **Low Consensus**: 25-49% agreement
- **Outliers**: <25% agreement
- **Cross-Embedding Agreement**: Jaccard Index across embedding backends

### 2. `visualize_results_v4.py`

**Purpose:** Generate comprehensive visualizations

**Synthetic Baseline Visualizations:**
1. `llm_tus.png` - LLM Performance TUS ranking (Channel C)
2. `embedding_tus.png` - Embedding TUS for Channel A and A+ (side-by-side)
3. `llm_f1_heatmap.png` - LLM F1 scores by baseline (Model × Baseline heatmap)
4. `channel_synergy.png` - LLM-Embedding agreement (for reference)
5. `llm_precision_recall.png` - LLM precision-recall scatter (Channel C)
6. `embedding_precision_recall.png` - Embedding precision-recall (Channel A and A+ side-by-side)

**Naturalistic Baseline Visualizations (per baseline):**
1. `{B}_consensus_distribution.png` - Consensus category distribution
2. `{B}_high_consensus_paths.png` - Top agreed-upon paths
3. `{B}_summary.png` - Overall analysis summary

**Usage:**
```bash
# Visualize latest run (automatic)
python visualize_results_v4.py

# Visualize specific run
python visualize_results_v4.py 001
python visualize_results_v4.py run_005
```

**Note:** Visualizer automatically uses the latest analysis run if no run ID specified.

### 3. `add_embedding_to_filenames.py`

**Purpose:** Standardize v4 filenames to include embedding backend

**Usage:**
```bash
python add_embedding_to_filenames.py
```

**Before:** `B1_deepseek_a1b2c3d4.json`  
**After:** `B1_dee_e5_a1b2c3d4.json`

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

## Workflow

### Full Analysis Pipeline

```bash
# 1. Ensure v4 structure is correct
ls -R ../v4/baselines/

# 2. Run analysis (creates run_001, run_002, etc.)
python analyze_results_v4.py
# Output: "Run ID: 001"

# 3. Generate visualizations (uses latest run automatically)
python visualize_results_v4.py

# 4. Review outputs
ls analysis_output/run_001/

# 5. Run again with new data (creates run_002)
python analyze_results_v4.py
python visualize_results_v4.py

# 6. Visualize specific run
python visualize_results_v4.py 001
```

### Output Directory Structure

```
analysis_output/
├── run_001/
│   ├── ANALYSIS_SUMMARY.txt              ← **Human-readable summary**
│   ├── synthetic_baselines_analysis.csv
│   ├── model_summary_synthetic.csv
│   ├── naturalistic_baselines_analysis.json
│   ├── llm_tus.png
│   ├── embedding_tus.png
│   ├── llm_f1_heatmap.png
│   ├── channel_synergy.png
│   ├── llm_precision_recall.png
│   ├── embedding_precision_recall.png
│   ├── B9_consensus_distribution.png
│   ├── B9_high_consensus_paths.png
│   ├── B9_summary.png
│   └── ... (additional baseline visualizations)
│
├── run_002/
│   └── ... (same structure)
│
└── run_003/
    └── ... (same structure)
```

**Run ID Features:**
- Automatic incremental numbering (001, 002, 003...)
- No overwriting of previous analyses
- Easy comparison between runs
- Visualizer auto-detects latest run
- **ANALYSIS_SUMMARY.txt** - Quick overview without opening CSV/PNG files

## Interpreting Results

### For Synthetic Baselines

**High Performance:**
- F1 > 0.8: Excellent detection
- TUS > 70: Strong transformation understanding
- Channel Synergy > 0.6: Good embedding-LLM agreement

**Moderate Performance:**
- F1 0.5-0.8: Decent detection with room for improvement
- TUS 50-70: Acceptable understanding
- Channel Synergy 0.4-0.6: Some disagreement

**Low Performance:**
- F1 < 0.5: Poor detection
- TUS < 50: Weak understanding
- Channel Synergy < 0.4: Significant disagreement

### For Naturalistic Baselines

**Strong Signal:**
- High consensus paths with 90%+ agreement
- Cross-embedding agreement > 0.6
- Multiple models converging on same paths

**Moderate Signal:**
- Moderate consensus (50-75% agreement)
- Cross-embedding agreement 0.4-0.6
- Some model variation

**Weak/Noisy Signal:**
- Mostly outliers (<25% agreement)
- Cross-embedding agreement < 0.4
- High model disagreement

## Model Codes

| Code | Full Name |
|------|-----------|
| dee | DeepSeek |
| gem | Gemini 2.5 Pro |
| gpt | ChatGPT 5.1 |
| haiku | Claude Haiku 4.5 |
| sonnet | Claude Sonnet 4.5 |
| opus | Claude Opus 4.1 |

## Embedding Backends

| Code | Full Name | Dimensions |
|------|-----------|------------|
| e5 | E5-Large (Local fine-tuned) | 768 |
| ada02 | Ada-002 (OpenAI) | 1536 |
| cohere | Cohere embed-english-v3.0 | 1024 |

## Notes

### Channel Definitions

- **Channel A**: Embedding-based detection (full transcript)
- **Channel A+**: Embedding-based detection (assistant→user pairs)
- **Channel B**: LLM validation of A (disabled by default)
- **Channel B+**: LLM validation of A+ (disabled by default)
- **Channel C**: Independent LLM reasoning and extraction

### Thresholds

**Embedding Detection (A/A+):**
- E5/Ada-002: T1 > 0.72, T2 > 0.75
- Cohere: T1 > 0.50, T2 > 0.55 (softer semantic space)

**Consensus Levels:**
- High: 75%+ agreement
- Moderate: 50-74% agreement
- Low: 25-49% agreement
- Outliers: <25% agreement

## Troubleshooting

**Issue:** No results loaded  
**Solution:** Check v4 directory structure, ensure files follow naming convention

**Issue:** Missing baselines  
**Solution:** Verify baseline.json exists in each baseline folder

**Issue:** Visualization errors  
**Solution:** Ensure matplotlib/seaborn installed: `pip install matplotlib seaborn pandas numpy`

**Issue:** Encoding errors  
**Solution:** All scripts use UTF-8 encoding, Windows users should verify

## Future Enhancements

- [ ] Weighted scoring for secondary targets
- [ ] Temporal cascade architecture analysis
- [ ] T1-only vs T1+T2 detection comparison
- [ ] Statistical significance testing
- [ ] Interactive dashboards
- [ ] Automated report generation

## Contact

For questions or issues, refer to the main S64 validation documentation.

