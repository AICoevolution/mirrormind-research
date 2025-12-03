#!/usr/bin/env python3
"""
S64 Validation Results Analyzer - V4 Structure

Analyzes results from the v4 standardized structure:
- Synthetic baselines (B1-B8): Compare against ground truth targets
- Naturalistic baselines (B9+): Consensus analysis across models/embeddings

Supports multiple embedding backends (E5, Ada-002, Cohere) and multiple LLM models.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set

# ============================================================================
# GROUND TRUTH DEFINITIONS (Synthetic Baselines B1-B8)
# ============================================================================

GROUND_TRUTH = {
    'B1': [],  # Surface deception - NO real transformations
    'B2': ['M10', 'M11', 'M55', 'M34'],  # Implicit transformation (memory/mirror/presence cluster)
    'B3': ['M9', 'M33', 'M34'],  # Rapid oscillation (crisis → insight → embodiment)
    'B4': [],  # Stuck States - T1-only, NO completed paths
    'B5': ['M16', 'M27', 'M41', 'M59', 'M60', 'M61', 'M62'],  # Nested complexity
    'B6': ['M12', 'M56'],  # Explicit transformation (somatic mechanics)
    'B7': [],  # Failed transformation - NO completed transformations
    'B8': []   # False completion - NO real transformations
}

GROUND_TRUTH_SECONDARY = {
    'B2': ['M38', 'M54', 'M56'],
    'B3': ['M24', 'M55', 'M23'],
    'B4': ['M15', 'M22', 'M14'],  # T1-only partial activations
    'B6': ['M24', 'M55']
}

BASELINE_LABELS = {
    'B1': 'Surface Deception',
    'B2': 'Implicit Transformation',
    'B3': 'Rapid Oscillation',
    'B4': 'Stuck States',
    'B5': 'Nested Complexity',
    'B6': 'Explicit Transformation',
    'B7': 'Failed Transformation',
    'B8': 'False Completion',
}

# Model code to full name mapping
MODEL_NAMES = {
    'haiku': 'Claude Haiku 4.5',
    'hai': 'Claude Haiku 4.5',
    'sonnet': 'Claude Sonnet 4.5',
    'son': 'Claude Sonnet 4.5',
    'opus': 'Claude Opus 4.1',
    'opu': 'Claude Opus 4.1',
    'gemini': 'Gemini 3.0 Pro',
    'gem': 'Gemini 3.0 Pro',
    'gpt': 'ChatGPT 5.1',
    'gpt5': 'ChatGPT 5.1',
    'deepseek': 'DeepSeek',
    'dee': 'DeepSeek'
}

EMBEDDING_NAMES = {
    'e5': 'E5-Large (Local)',
    'ada02': 'Ada-002 (OpenAI)',
    'ada002': 'Ada-002 (OpenAI)',
    'cohere': 'Cohere v3.0'
}

# ============================================================================
# FILE LOADING AND PARSING
# ============================================================================

def parse_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Parse v4 filename format: B{N}_{model}_{embedding}_{uuid}.json
    
    Returns dict with: baseline, model, embedding, uuid
    """
    stem = Path(filename).stem
    parts = stem.split('_')
    
    if len(parts) >= 4:
        return {
            'baseline': parts[0],  # B1, B2, etc.
            'model': parts[1].lower(),  # dee, gem, etc.
            'embedding': parts[2].lower(),  # e5, ada02, cohere
            'uuid': parts[3]  # 8-char UUID
        }
    return None


def load_v4_results(v4_dir: Path) -> Dict[str, Dict]:
    """
    Load all result files from v4 structure
    
    Returns dict keyed by: "{baseline}_{model}_{embedding}_{uuid}"
    """
    results = {}
    
    # Find all JSON files in results_* directories
    for json_file in v4_dir.rglob('**/results_*/**/*.json'):
        if json_file.name == 'baseline.json':
            continue  # Skip baseline conversation files
        
        # Parse filename
        file_info = parse_filename(json_file.name)
        if not file_info:
            print(f"⚠️  Could not parse filename: {json_file.name}")
            continue
        
        # Load data
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Create unique key
            key = f"{file_info['baseline']}_{file_info['model']}_{file_info['embedding']}"
            results[key] = {
                'data': data,
                'file_info': file_info,
                'file_path': json_file
            }
        except Exception as e:
            print(f"❌ Error loading {json_file.name}: {e}")
    
    return results


def categorize_baselines(v4_dir: Path) -> Tuple[List[str], List[str]]:
    """
    Categorize baselines into synthetic and naturalistic
    
    Returns: (synthetic_list, naturalistic_list)
    """
    synthetic = []
    naturalistic = []
    
    baselines_dir = v4_dir / 'baselines'
    
    # Check synthetic
    synthetic_dir = baselines_dir / 'synthetic'
    if synthetic_dir.exists():
        for baseline_dir in synthetic_dir.iterdir():
            if baseline_dir.is_dir():
                # Extract baseline ID from dirname (e.g., "B1_surface_deception" -> "B1")
                baseline_id = baseline_dir.name.split('_')[0]
                if baseline_id.startswith('B'):
                    synthetic.append(baseline_id)
    
    # Check naturalistic
    naturalistic_dir = baselines_dir / 'naturalistic'
    if naturalistic_dir.exists():
        for baseline_dir in naturalistic_dir.iterdir():
            if baseline_dir.is_dir():
                baseline_id = baseline_dir.name.split('_')[0]
                if baseline_id.startswith('B'):
                    naturalistic.append(baseline_id)
    
    return sorted(synthetic), sorted(naturalistic)


# ============================================================================
# METRICS EXTRACTION
# ============================================================================

def extract_channel_c_paths(data: Dict) -> Set[int]:
    """Extract path numbers detected by Channel C"""
    paths = set()
    channel_c = data.get('channels', {}).get('C', {})
    for path in channel_c.get('paths', []):
        if 'path_number' in path:
            paths.add(path['path_number'])
    return paths


def extract_t1_only_paths(data: Dict, t1_threshold=0.72) -> Set[int]:
    """
    Extract paths where T1 is strong but T2 is weak (incomplete transformations)
    These indicate transformations that were initiated but not completed
    """
    t1_only_paths = set()
    
    # Channel A
    channel_a = data.get('channels', {}).get('A', {})
    for path in channel_a.get('paths', []):
        t1 = path.get('t1_confidence', 0)
        t2 = path.get('t2_confidence', 0)
        # T1 is strong but T2 is weak
        if t1 > t1_threshold and t2 < t1_threshold:
            t1_only_paths.add(path['path_number'])
    
    # Channel A+
    channel_aplus = data.get('channels', {}).get('A+', {})
    for path in channel_aplus.get('paths', []):
        t1 = path.get('t1_confidence', 0)
        t2 = path.get('t2_confidence', 0)
        if t1 > t1_threshold and t2 < t1_threshold:
            t1_only_paths.add(path['path_number'])
    
    return t1_only_paths


def get_embedding_thresholds(backend: str) -> Tuple[float, float]:
    """
    Backend-specific thresholds for embedding channels.
    
    - Ada-002 / E5:  T1 > 0.72, T2 > 0.75
    - Cohere:        T1 > 0.50, T2 > 0.55
    """
    backend = (backend or "").lower()
    if backend == "cohere":
        return 0.50, 0.55
    # Default for Ada-002 / E5 (and any others not yet tuned)
    return 0.72, 0.75


def extract_channel_a_paths(data: Dict) -> Set[int]:
    """
    Extract paths detected by Channel A (whole-conversation embedding).
    """
    backend = data.get('embedding_backend', '')
    t1_threshold, t2_threshold = get_embedding_thresholds(backend)
    
    paths = set()
    channel_a = data.get('channels', {}).get('A', {})
    for path in channel_a.get('paths', []):
        t1 = path.get('t1_confidence', 0)
        t2 = path.get('t2_confidence', 0)
        if t1 > t1_threshold and t2 > t2_threshold:
            paths.add(path['path_number'])
    return paths


def extract_channel_aplus_paths(data: Dict) -> Set[int]:
    """
    Extract paths detected by Channel A+ (pairwise user–assistant embedding).
    """
    backend = data.get('embedding_backend', '')
    t1_threshold, t2_threshold = get_embedding_thresholds(backend)
    
    paths = set()
    channel_aplus = data.get('channels', {}).get('A+', {})
    for path in channel_aplus.get('paths', []):
        t1 = path.get('t1_confidence', 0)
        t2 = path.get('t2_confidence', 0)
        if t1 > t1_threshold and t2 > t2_threshold:
            paths.add(path['path_number'])
    return paths


def extract_embedding_paths(data: Dict) -> Set[int]:
    """
    Backwards-compatible helper: union of Channel A and A+ paths.
    """
    return extract_channel_a_paths(data) | extract_channel_aplus_paths(data)


def calculate_precision_recall_f1(detected: Set[int], ground_truth: Set[int]) -> Dict[str, float]:
    """Calculate precision, recall, and F1 score"""
    if not ground_truth:
        # Special case: no ground truth (deception baselines)
        if not detected:
            # Correctly detected nothing
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
        else:
            # False positives
            return {'precision': 0.0, 'recall': 1.0, 'f1': 0.0}
    
    if not detected:
        # Detected nothing but there was ground truth
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    true_positives = len(detected & ground_truth)
    false_positives = len(detected - ground_truth)
    false_negatives = len(ground_truth - detected)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {'precision': precision, 'recall': recall, 'f1': f1}


def calculate_inference_depth(metrics: Dict) -> float:
    """
    Recreate the original "inference_depth_component" metric from the v3 analyzer.
    
    - B1: reward NOT detecting false positives (up to 20 points)
    - B2, B3, B5: proportional to Channel C F1 (0–20)
    - B4: proportional to precision (0–20)
    - B6–B8: currently contribute 0 (same as original script)
    """
    baseline = metrics.get('baseline')
    score = 0.0
    
    if baseline == 'B1':
        # Same heuristic as original: assume up to 5 "slots" for false positives
        detected = metrics.get('c_paths_detected', 0)
        ratio = min(detected / 5.0, 1.0)
        score = (1.0 - ratio) * 20.0
    elif baseline in ('B2', 'B3', 'B5'):
        score = metrics.get('c_f1', 0.0) * 20.0
    elif baseline == 'B4':
        score = metrics.get('c_precision', 0.0) * 20.0
    # B6, B7, B8 → 0.0 by design
    
    return round(score, 2)


def calculate_tus(f1: float, precision: float, recall: float, 
                   is_deception: bool = False, paths_detected: int = 0) -> float:
    """
    Transformation Understanding Score (TUS) – standardized metric.
    
    Works the same for any detection channel (LLM or Embedding).
    Max score: 70 points (detection + calibration + structural)
    
    Args:
        f1: F1 score for this channel
        precision: Precision for this channel
        recall: Recall for this channel
        is_deception: Whether this is a deception baseline (no ground truth)
        paths_detected: Number of paths detected (for structural bonus on deception)
    """
    # 1) Detection accuracy (40 points)
    detection_score = f1 * 40.0
    
    # 2) Confidence calibration (20 points) - penalize imbalance between precision/recall
    if precision + recall > 0:
        balance = 1.0 - abs(precision - recall) / (precision + recall)
        calibration_score = balance * 20.0
    else:
        calibration_score = 0.0
    
    # 3) Structural bonus (10 points)
    if is_deception:
        # Reward correct "detect nothing" behaviour
        structural_score = 10.0 if paths_detected == 0 else 0.0
    else:
        # Small bonus for very strong F1
        if f1 > 0.8:
            structural_score = 10.0
        elif f1 > 0.5:
            structural_score = 5.0
        else:
            structural_score = 0.0
    
    # Total TUS (max 70 points)
    tus = detection_score + calibration_score + structural_score
    return round(tus, 2)


# ============================================================================
# SYNTHETIC BASELINE ANALYSIS
# ============================================================================

def analyze_synthetic_baseline(baseline_id: str, results: Dict) -> pd.DataFrame:
    """
    Analyze a synthetic baseline against ground truth
    
    Returns DataFrame with metrics for each model/embedding combination
    """
    ground_truth = set([int(p[1:]) for p in GROUND_TRUTH.get(baseline_id, [])])
    is_deception = len(ground_truth) == 0
    
    rows = []
    
    # Filter results for this baseline
    baseline_results = {k: v for k, v in results.items() if k.startswith(f"{baseline_id}_")}
    
    for key, result_obj in baseline_results.items():
        data = result_obj['data']
        file_info = result_obj['file_info']
        
        # Extract detected paths
        channel_c_paths = extract_channel_c_paths(data)
        channel_a_paths = extract_channel_a_paths(data)
        channel_aplus_paths = extract_channel_aplus_paths(data)
        embedding_paths = channel_a_paths | channel_aplus_paths  # union
        
        # Store detected paths as lists for CSV
        c_paths_list = sorted(list(channel_c_paths))
        embed_a_paths_list = sorted(list(channel_a_paths))
        embed_aplus_paths_list = sorted(list(channel_aplus_paths))
        embed_paths_list = sorted(list(embedding_paths))
        
        # Calculate metrics for Channel C
        c_metrics = calculate_precision_recall_f1(channel_c_paths, ground_truth)
        
        # Calculate metrics for embeddings (separate channels + union)
        embed_a_metrics = calculate_precision_recall_f1(channel_a_paths, ground_truth)
        embed_aplus_metrics = calculate_precision_recall_f1(channel_aplus_paths, ground_truth)
        embed_metrics = calculate_precision_recall_f1(embedding_paths, ground_truth)
        
        # Channel synergy: Jaccard similarity between C and embeddings (union)
        if channel_c_paths or embedding_paths:
            intersection = len(channel_c_paths & embedding_paths)
            union = len(channel_c_paths | embedding_paths)
            channel_synergy = intersection / union if union > 0 else 0.0
        else:
            channel_synergy = 1.0 if is_deception else 0.0
        
        # Token usage (we only have Channel C in v4 runs, but keep shape compatible)
        # Build metrics dict
        metrics = {
            'baseline': baseline_id,
            'baseline_label': BASELINE_LABELS.get(baseline_id, baseline_id),
            'model': file_info['model'],
            'model_name': MODEL_NAMES.get(file_info['model'], file_info['model']),
            'embedding': file_info['embedding'],
            'embedding_name': EMBEDDING_NAMES.get(file_info['embedding'], file_info['embedding']),
            
            # Channel C metrics
            'c_paths_detected': len(channel_c_paths),
            'c_precision': c_metrics['precision'],
            'c_recall': c_metrics['recall'],
            'c_f1': c_metrics['f1'],
            
            # Embedding metrics (union of A and A+)
            'embed_paths_detected': len(embedding_paths),
            'embed_precision': embed_metrics['precision'],
            'embed_recall': embed_metrics['recall'],
            'embed_f1': embed_metrics['f1'],
            
            # Channel A (whole conversation) metrics
            'embed_a_paths_detected': len(channel_a_paths),
            'embed_a_precision': embed_a_metrics['precision'],
            'embed_a_recall': embed_a_metrics['recall'],
            'embed_a_f1': embed_a_metrics['f1'],
            
            # Channel A+ (turn pairs) metrics
            'embed_aplus_paths_detected': len(channel_aplus_paths),
            'embed_aplus_precision': embed_aplus_metrics['precision'],
            'embed_aplus_recall': embed_aplus_metrics['recall'],
            'embed_aplus_f1': embed_aplus_metrics['f1'],
            
            # Combined metrics
            'channel_synergy': channel_synergy,
            'is_deception_baseline': is_deception,
            
            # Detected paths (as lists for summary display)
            'c_paths_detected_list': c_paths_list,
            'embed_paths_detected_list': embed_paths_list,
            'embed_a_paths_detected_list': embed_a_paths_list,
            'embed_aplus_paths_detected_list': embed_aplus_paths_list,
            
            # Ground truth
            'ground_truth_count': len(ground_truth),
            'ground_truth_paths': sorted(list(ground_truth))
        }
        
        # Inference depth (original metric, per baseline)
        metrics['inference_depth_component'] = calculate_inference_depth(metrics)
        
        # Calculate TUS for LLM (Channel C) - standardized metric
        metrics['llm_tus'] = calculate_tus(
            f1=c_metrics['f1'],
            precision=c_metrics['precision'],
            recall=c_metrics['recall'],
            is_deception=is_deception,
            paths_detected=len(channel_c_paths)
        )
        
        # Calculate TUS for Embeddings, per channel (A and A+)
        metrics['embed_a_tus'] = calculate_tus(
            f1=embed_a_metrics['f1'],
            precision=embed_a_metrics['precision'],
            recall=embed_a_metrics['recall'],
            is_deception=is_deception,
            paths_detected=len(channel_a_paths)
        )
        metrics['embed_aplus_tus'] = calculate_tus(
            f1=embed_aplus_metrics['f1'],
            precision=embed_aplus_metrics['precision'],
            recall=embed_aplus_metrics['recall'],
            is_deception=is_deception,
            paths_detected=len(channel_aplus_paths)
        )
        
        # Legacy combined embedding TUS (max of A / A+) for backwards compatibility
        metrics['embed_tus'] = max(metrics['embed_a_tus'], metrics['embed_aplus_tus'])
        
        # Legacy 'tus' field for backwards compatibility (uses LLM TUS)
        metrics['tus'] = metrics['llm_tus']
        
        rows.append(metrics)
    
    return pd.DataFrame(rows)


# ============================================================================
# NATURALISTIC BASELINE ANALYSIS
# ============================================================================

def analyze_naturalistic_baseline(baseline_id: str, results: Dict) -> Dict:
    """
    Analyze a naturalistic baseline using consensus analysis
    
    Returns dict with consensus metrics and detected paths by each method
    """
    # Filter results for this baseline
    baseline_results = {k: v for k, v in results.items() if k.startswith(f"{baseline_id}_")}
    
    # Track detections by model, embedding, and method
    detections_by_model = defaultdict(lambda: {'c': set(), 'embed': set(), 't1_only': set()})
    detections_by_embedding = defaultdict(lambda: {'c': set(), 'embed': set(), 't1_only': set()})
    all_c_detections = []
    all_embed_detections = []
    all_t1_only_detections = []
    
    for key, result_obj in baseline_results.items():
        data = result_obj['data']
        file_info = result_obj['file_info']
        
        model = file_info['model']
        embedding = file_info['embedding']
        
        # Extract paths
        c_paths = extract_channel_c_paths(data)
        embed_paths = extract_embedding_paths(data)
        t1_only_paths = extract_t1_only_paths(data)
        
        # Track by model
        detections_by_model[model]['c'].update(c_paths)
        detections_by_model[model]['embed'].update(embed_paths)
        detections_by_model[model]['t1_only'].update(t1_only_paths)
        
        # Track by embedding
        detections_by_embedding[embedding]['c'].update(c_paths)
        detections_by_embedding[embedding]['embed'].update(embed_paths)
        detections_by_embedding[embedding]['t1_only'].update(t1_only_paths)
        
        # Track all
        all_c_detections.append(c_paths)
        all_embed_detections.append(embed_paths)
        all_t1_only_detections.append(t1_only_paths)
    
    # Calculate consensus
    def calculate_consensus(detection_sets: List[Set[int]]) -> Dict:
        """Calculate consensus categories for path detections"""
        if not detection_sets:
            return {
                'high': [],
                'moderate': [],
                'low': [],
                'outliers': []
            }
        
        path_counts = defaultdict(int)
        for paths in detection_sets:
            for path in paths:
                path_counts[path] += 1
        
        total_methods = len(detection_sets)
        consensus = {
            'high': [],      # Detected by 75%+ methods
            'moderate': [],  # Detected by 50-74% methods
            'low': [],       # Detected by 25-49% methods
            'outliers': []   # Detected by <25% methods
        }
        
        for path, count in path_counts.items():
            ratio = count / total_methods
            if ratio >= 0.75:
                consensus['high'].append((path, count, ratio))
            elif ratio >= 0.50:
                consensus['moderate'].append((path, count, ratio))
            elif ratio >= 0.25:
                consensus['low'].append((path, count, ratio))
            else:
                consensus['outliers'].append((path, count, ratio))
        
        # Sort each category by count (descending)
        for category in consensus:
            consensus[category].sort(key=lambda x: x[1], reverse=True)
        
        return consensus
    
    c_consensus = calculate_consensus(all_c_detections)
    embed_consensus = calculate_consensus(all_embed_detections)
    t1_only_consensus = calculate_consensus(all_t1_only_detections)
    
    # Calculate cross-embedding agreement (Jaccard Index)
    embed_sets = [detections_by_embedding[emb]['embed'] for emb in detections_by_embedding]
    if len(embed_sets) >= 2:
        # Calculate pairwise Jaccard indices
        jaccard_scores = []
        for i in range(len(embed_sets)):
            for j in range(i + 1, len(embed_sets)):
                intersection = len(embed_sets[i] & embed_sets[j])
                union = len(embed_sets[i] | embed_sets[j])
                jaccard = intersection / union if union > 0 else 0.0
                jaccard_scores.append(jaccard)
        avg_jaccard = np.mean(jaccard_scores) if jaccard_scores else 0.0
    else:
        avg_jaccard = 1.0
    
    return {
        'baseline_id': baseline_id,
        'total_runs': len(baseline_results),
        'models': list(detections_by_model.keys()),
        'embeddings': list(detections_by_embedding.keys()),
        'channel_c_consensus': c_consensus,
        'embedding_consensus': embed_consensus,
        't1_only_consensus': t1_only_consensus,
        'cross_embedding_agreement': avg_jaccard,
        'detections_by_model': dict(detections_by_model),
        'detections_by_embedding': dict(detections_by_embedding)
    }


def generate_summary_text(
    synthetic_df: Optional[pd.DataFrame],
    naturalistic_results: Optional[Dict],
    output_dir: Path,
    run_id: str
):
    """
    Generate comprehensive human-readable summary text file
    """
    summary_path = output_dir / 'ANALYSIS_SUMMARY.txt'
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("="*80 + "\n")
        f.write("S64 VALIDATION RESULTS - COMPREHENSIVE SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # ====================================================================
        # SYNTHETIC BASELINES
        # ====================================================================
        
        if synthetic_df is not None and not synthetic_df.empty:
            f.write("="*80 + "\n")
            f.write("PART 1: SYNTHETIC BASELINES (B1-B8)\n")
            f.write("="*80 + "\n\n")
            
            # ----------------------------------------------------------------
            # OVERALL MODEL INFERENCE DEPTH SCORES (Original Metric)
            # ----------------------------------------------------------------
            f.write("OVERALL MODEL INFERENCE DEPTH SCORES (Original Metric)\n")
            f.write("-" * 80 + "\n\n")
            
            id_summary = synthetic_df.groupby('model_name').agg({
                'inference_depth_component': 'sum',
                'c_f1': 'mean',
                'c_paths_detected': 'mean'
            }).sort_values('inference_depth_component', ascending=False)
            
            f.write(f"{'Model':<22} {'InfDepth':>9} {'C F1':>7} {'C Detected':>11}\n")
            f.write("-" * 80 + "\n")
            for model, row in id_summary.iterrows():
                f.write(
                    f"{model:<22} "
                    f"{row['inference_depth_component']:>9.2f} "
                    f"{row['c_f1']:>7.3f} "
                    f"{row['c_paths_detected']:>11.2f}\n"
                )
            
            f.write("\n" + "="*80 + "\n\n")
            
            # ----------------------------------------------------------------
            # LLM TUS (Channel C) - Standardized Metric
            # ----------------------------------------------------------------
            f.write("LLM PERFORMANCE - TUS (Channel C)\n")
            f.write("-" * 80 + "\n\n")
            
            # Aggregate LLM TUS by model (average across embeddings since they're independent)
            llm_tus_summary = synthetic_df.groupby('model_name').agg({
                'llm_tus': 'mean',  # Average across embeddings (they're independent)
                'c_f1': 'mean',
                'c_precision': 'mean',
                'c_recall': 'mean'
            })
            
            # Normalized: percentage of max 70
            llm_tus_summary['tus_pct'] = (llm_tus_summary['llm_tus'] / 70.0 * 100.0).round(2)
            llm_tus_summary = llm_tus_summary.sort_values('llm_tus', ascending=False)
            
            f.write(f"{'Model':<22} {'TUS':>8} {'TUS%':>8} {'F1':>8} {'Prec':>8} {'Recall':>8}\n")
            f.write("-" * 80 + "\n")
            
            for model, row in llm_tus_summary.iterrows():
                f.write(
                    f"{model:<22} "
                    f"{row['llm_tus']:>8.2f} "
                    f"{row['tus_pct']:>8.2f} "
                    f"{row['c_f1']:>8.3f} "
                    f"{row['c_precision']:>8.3f} "
                    f"{row['c_recall']:>8.3f}\n"
                )
            
            f.write("\n" + "="*80 + "\n\n")
            
            # ----------------------------------------------------------------
            # EMBEDDING TUS (Channel A and A+) - Same Standardized Metric
            # ----------------------------------------------------------------
            f.write("EMBEDDING PERFORMANCE - TUS (Channel A / Channel A+)\n")
            f.write("-" * 80 + "\n\n")
            
            # Aggregate Embedding TUS by embedding backend (average across models)
            embed_a_summary = synthetic_df.groupby('embedding_name').agg({
                'embed_a_tus': 'mean',
                'embed_a_f1': 'mean',
                'embed_a_precision': 'mean',
                'embed_a_recall': 'mean'
            })
            embed_aplus_summary = synthetic_df.groupby('embedding_name').agg({
                'embed_aplus_tus': 'mean',
                'embed_aplus_f1': 'mean',
                'embed_aplus_precision': 'mean',
                'embed_aplus_recall': 'mean'
            })
            
            # Normalized: percentage of max 70
            embed_a_summary['tus_pct'] = (embed_a_summary['embed_a_tus'] / 70.0 * 100.0).round(2)
            embed_aplus_summary['tus_pct'] = (embed_aplus_summary['embed_aplus_tus'] / 70.0 * 100.0).round(2)
            
            embed_a_summary = embed_a_summary.sort_values('embed_a_tus', ascending=False)
            embed_aplus_summary = embed_aplus_summary.sort_values('embed_aplus_tus', ascending=False)
            
            # Channel A table
            f.write("Channel A (whole conversation):\n")
            f.write(f"{'Embedding':<25} {'TUS_A':>8} {'TUS_A%':>8} {'F1_A':>8} {'Prec_A':>8} {'Rec_A':>8}\n")
            f.write("-" * 80 + "\n")
            for embed, row in embed_a_summary.iterrows():
                f.write(
                    f"{embed:<25} "
                    f"{row['embed_a_tus']:>8.2f} "
                    f"{row['tus_pct']:>8.2f} "
                    f"{row['embed_a_f1']:>8.3f} "
                    f"{row['embed_a_precision']:>8.3f} "
                    f"{row['embed_a_recall']:>8.3f}\n"
                )
            
            f.write("\n\nChannel A+ (user–assistant pairs):\n")
            f.write(f"{'Embedding':<25} {'TUS_A+':>8} {'TUS_A+%':>8} {'F1_A+':>8} {'Prec_A+':>8} {'Rec_A+':>8}\n")
            f.write("-" * 80 + "\n")
            for embed, row in embed_aplus_summary.iterrows():
                f.write(
                    f"{embed:<25} "
                    f"{row['embed_aplus_tus']:>8.2f} "
                    f"{row['tus_pct']:>8.2f} "
                    f"{row['embed_aplus_f1']:>8.3f} "
                    f"{row['embed_aplus_precision']:>8.3f} "
                    f"{row['embed_aplus_recall']:>8.3f}\n"
                )
            
            f.write("\n" + "="*80 + "\n\n")
            
            # Per-baseline analysis
            f.write("DETAILED BASELINE-BY-BASELINE ANALYSIS\n")
            f.write("-" * 80 + "\n\n")
            
            for baseline_id in sorted(synthetic_df['baseline'].unique()):
                baseline_data = synthetic_df[synthetic_df['baseline'] == baseline_id]
                baseline_label = baseline_data.iloc[0]['baseline_label']
                ground_truth = baseline_data.iloc[0]['ground_truth_paths']
                
                f.write(f"{baseline_id}: {baseline_label}\n")
                f.write("=" * 80 + "\n")
                
                if ground_truth:
                    f.write(f"Ground Truth: {len(ground_truth)} paths → {', '.join([f'M{p}' for p in ground_truth])}\n")
                else:
                    f.write("Ground Truth: NONE (Deception Baseline)\n")
                
                f.write(f"\nResults across {len(baseline_data)} model/embedding combinations:\n")
                f.write(f"  Average Channel C F1: {baseline_data['c_f1'].mean():.3f}\n")
                f.write(f"  Average TUS:          {baseline_data['tus'].mean():.2f}\n")
                f.write(f"  Best F1:              {baseline_data['c_f1'].max():.3f}\n")
                f.write(f"  Worst F1:             {baseline_data['c_f1'].min():.3f}\n\n")
                
                # Detection details by model
                f.write("Channel C Detections by Model:\n")
                for model in sorted(baseline_data['model_name'].unique()):
                    model_data = baseline_data[baseline_data['model_name'] == model]
                    avg_detected = model_data['c_paths_detected'].mean()
                    avg_f1 = model_data['c_f1'].mean()
                    
                    # Get all detected paths across embeddings for this model
                    all_paths = set()
                    for _, row in model_data.iterrows():
                        paths = row.get('c_paths_detected_list', [])
                        if isinstance(paths, list):
                            all_paths.update(paths)
                    
                    f.write(f"  {model:20s} → Avg {avg_detected:.1f} paths, F1: {avg_f1:.3f}")
                    if all_paths:
                        sorted_paths = sorted(list(all_paths))
                        f.write(f" | Paths: {', '.join([f'M{p}' for p in sorted_paths[:8]])}")
                        if len(sorted_paths) > 8:
                            f.write(f" ... (+{len(sorted_paths)-8})")
                    f.write("\n")
                
                f.write("\n")
                
                # Embedding backend comparison (Channel A and A+ separately)
                f.write("Performance by Embedding Backend:\n")
                for embedding in sorted(baseline_data['embedding_name'].unique()):
                    embed_data = baseline_data[baseline_data['embedding_name'] == embedding]
                    
                    # Channel A
                    avg_a_f1 = embed_data['embed_a_f1'].mean()
                    avg_a_tus = embed_data['embed_a_tus'].mean() if 'embed_a_tus' in embed_data.columns else float('nan')
                    all_a_paths = set()
                    for _, row in embed_data.iterrows():
                        paths = row.get('embed_a_paths_detected_list', [])
                        if isinstance(paths, list):
                            all_a_paths.update(paths)
                    
                    f.write(f"  {embedding:25s} [Channel A]  → F1: {avg_a_f1:.3f}, TUS: {avg_a_tus:.2f}")
                    if all_a_paths:
                        sorted_paths = sorted(list(all_a_paths))
                        f.write(f" | Paths: {', '.join([f'M{p}' for p in sorted_paths[:8]])}")
                        if len(sorted_paths) > 8:
                            f.write(f" ... (+{len(sorted_paths)-8})")
                    f.write("\n")
                    
                    # Channel A+
                    avg_ap_f1 = embed_data['embed_aplus_f1'].mean()
                    avg_ap_tus = embed_data['embed_aplus_tus'].mean() if 'embed_aplus_tus' in embed_data.columns else float('nan')
                    all_ap_paths = set()
                    for _, row in embed_data.iterrows():
                        paths = row.get('embed_aplus_paths_detected_list', [])
                        if isinstance(paths, list):
                            all_ap_paths.update(paths)
                    
                    f.write(f"  {embedding:25s} [Channel A+] → F1: {avg_ap_f1:.3f}, TUS: {avg_ap_tus:.2f}")
                    if all_ap_paths:
                        sorted_paths = sorted(list(all_ap_paths))
                        f.write(f" | Paths: {', '.join([f'M{p}' for p in sorted_paths[:8]])}")
                        if len(sorted_paths) > 8:
                            f.write(f" ... (+{len(sorted_paths)-8})")
                    f.write("\n")
                
                f.write("\n" + "-" * 80 + "\n\n")
            
            # Key insights
            f.write("="*80 + "\n")
            f.write("KEY INSIGHTS - SYNTHETIC BASELINES\n")
            f.write("="*80 + "\n\n")
            
            # Best LLM (llm_tus_summary is already sorted by TUS descending)
            best_llm = llm_tus_summary.index[0]
            best_llm_tus = llm_tus_summary.iloc[0]['llm_tus']
            best_llm_pct = llm_tus_summary.iloc[0]['tus_pct']
            f.write(f"🏆 Best LLM: {best_llm} (TUS: {best_llm_tus:.2f}, {best_llm_pct:.1f}%)\n\n")
            
            # Best Embeddings (Channel A and A+)
            best_embed_a = embed_a_summary.index[0]
            best_embed_a_tus = embed_a_summary.iloc[0]['embed_a_tus']
            best_embed_a_pct = embed_a_summary.iloc[0]['tus_pct']
            f.write(f"🔬 Best Embedding (Channel A): {best_embed_a} (TUS: {best_embed_a_tus:.2f}, {best_embed_a_pct:.1f}%)\n")
            
            best_embed_aplus = embed_aplus_summary.index[0]
            best_embed_aplus_tus = embed_aplus_summary.iloc[0]['embed_aplus_tus']
            best_embed_aplus_pct = embed_aplus_summary.iloc[0]['tus_pct']
            f.write(f"🔬 Best Embedding (Channel A+): {best_embed_aplus} (TUS: {best_embed_aplus_tus:.2f}, {best_embed_aplus_pct:.1f}%)\n\n")
            
            # Easiest/hardest baselines
            baseline_avg_f1 = synthetic_df.groupby('baseline_label')['c_f1'].mean().sort_values(ascending=False)
            f.write(f"📊 Easiest Baseline: {baseline_avg_f1.index[0]} (Avg F1: {baseline_avg_f1.iloc[0]:.3f})\n")
            f.write(f"📊 Hardest Baseline: {baseline_avg_f1.index[-1]} (Avg F1: {baseline_avg_f1.iloc[-1]:.3f})\n\n")
        
        # ====================================================================
        # NATURALISTIC BASELINES
        # ====================================================================
        
        if naturalistic_results:
            f.write("\n" + "="*80 + "\n")
            f.write("PART 2: NATURALISTIC BASELINES (B9+)\n")
            f.write("="*80 + "\n\n")
            
            f.write("Note: No ground truth available. Analysis based on consensus across\n")
            f.write("      multiple models and embedding backends.\n\n")
            
            for baseline_id, analysis in naturalistic_results.items():
                f.write(f"{baseline_id}: Naturalistic Self-Discovery Conversation\n")
                f.write("=" * 80 + "\n\n")
                
                if analysis['total_runs'] == 0:
                    f.write("⚠️  No results available yet for this baseline.\n")
                    f.write("   Run detection on this baseline to generate consensus analysis.\n\n")
                    continue
                
                f.write(f"Analysis Summary:\n")
                f.write(f"  Total Runs:              {analysis['total_runs']}\n")
                f.write(f"  Models Analyzed:         {', '.join([MODEL_NAMES.get(m, m) for m in analysis['models']])}\n")
                f.write(f"  Embedding Backends:      {', '.join([EMBEDDING_NAMES.get(e, e) for e in analysis['embeddings']])}\n")
                f.write(f"  Cross-Embedding Agreement: {analysis['cross_embedding_agreement']:.3f} (Jaccard Index)\n\n")
                
                c_consensus = analysis['channel_c_consensus']
                
                f.write("CHANNEL C CONSENSUS BREAKDOWN:\n")
                f.write("-" * 80 + "\n")
                f.write(f"  High Consensus (75%+):     {len(c_consensus['high'])} paths\n")
                f.write(f"  Moderate Consensus (50-74%): {len(c_consensus['moderate'])} paths\n")
                f.write(f"  Low Consensus (25-49%):    {len(c_consensus['low'])} paths\n")
                f.write(f"  Outliers (<25%):           {len(c_consensus['outliers'])} paths\n\n")
                
                # High consensus paths (detailed)
                if c_consensus['high']:
                    f.write("HIGH CONSENSUS PATHS (75%+ Agreement):\n")
                    f.write("-" * 80 + "\n")
                    for path, count, ratio in c_consensus['high']:
                        f.write(f"  M{path:2d}: Detected in {count:2d}/{analysis['total_runs']} runs ({ratio*100:.0f}% agreement)\n")
                    f.write("\n")
                
                # Moderate consensus paths
                if c_consensus['moderate']:
                    f.write("MODERATE CONSENSUS PATHS (50-74% Agreement):\n")
                    f.write("-" * 80 + "\n")
                    for path, count, ratio in c_consensus['moderate']:
                        f.write(f"  M{path:2d}: Detected in {count:2d}/{analysis['total_runs']} runs ({ratio*100:.0f}% agreement)\n")
                    f.write("\n")
                
                # Model-specific detections
                f.write("DETECTIONS BY MODEL (Channel C):\n")
                f.write("-" * 80 + "\n")
                for model, detections in analysis['detections_by_model'].items():
                    model_name = MODEL_NAMES.get(model, model)
                    c_paths = sorted(list(detections['c']))
                    f.write(f"  {model_name:20s}: {len(c_paths)} paths → ")
                    f.write(f"{', '.join([f'M{p}' for p in c_paths[:10]])}")
                    if len(c_paths) > 10:
                        f.write(f" ... (+{len(c_paths)-10} more)")
                    f.write("\n")
                f.write("\n")
                
                # Embedding-specific detections
                f.write("DETECTIONS BY EMBEDDING (A/A+ Combined):\n")
                f.write("-" * 80 + "\n")
                for embedding, detections in analysis['detections_by_embedding'].items():
                    embed_name = EMBEDDING_NAMES.get(embedding, embedding)
                    embed_paths = sorted(list(detections['embed']))
                    f.write(f"  {embed_name:25s}: {len(embed_paths)} paths → ")
                    f.write(f"{', '.join([f'M{p}' for p in embed_paths[:10]])}")
                    if len(embed_paths) > 10:
                        f.write(f" ... (+{len(embed_paths)-10} more)")
                    f.write("\n")
                f.write("\n")
                
                # T1-ONLY PATHS - Incomplete Transformations
                t1_consensus = analysis.get('t1_only_consensus', {})
                if t1_consensus and any(len(t1_consensus.get(cat, [])) > 0 for cat in ['high', 'moderate', 'low']):
                    f.write("="*80 + "\n")
                    f.write("T1-ONLY PATHS (INCOMPLETE TRANSFORMATIONS)\n")
                    f.write("="*80 + "\n")
                    f.write("These paths show T1 (initiation) but weak/no T2 (completion)\n")
                    f.write("Indicates transformations that were started but not yet completed\n\n")
                    
                    if t1_consensus.get('high'):
                        f.write("HIGH CONSENSUS T1-ONLY (75%+ agreement):\n")
                        f.write("-" * 80 + "\n")
                        for path, count, ratio in t1_consensus['high']:
                            f.write(f"  M{path:2d}: {count:2d}/{analysis['total_runs']} runs ({ratio*100:5.1f}%) - Initiated but not completed\n")
                        f.write("\n")
                    
                    if t1_consensus.get('moderate'):
                        f.write("MODERATE CONSENSUS T1-ONLY (50-74% agreement):\n")
                        f.write("-" * 80 + "\n")
                        for path, count, ratio in t1_consensus['moderate']:
                            f.write(f"  M{path:2d}: {count:2d}/{analysis['total_runs']} runs ({ratio*100:5.1f}%) - Initiated but not completed\n")
                        f.write("\n")
                    
                    # T1-only by model
                    f.write("T1-ONLY DETECTIONS BY MODEL:\n")
                    f.write("-" * 80 + "\n")
                    for model, detections in analysis['detections_by_model'].items():
                        model_name = MODEL_NAMES.get(model, model)
                        t1_paths = sorted(list(detections.get('t1_only', set())))
                        if t1_paths:
                            f.write(f"  {model_name:20s}: {len(t1_paths)} paths → ")
                            f.write(f"{', '.join([f'M{p}' for p in t1_paths[:8]])}")
                            if len(t1_paths) > 8:
                                f.write(f" ... (+{len(t1_paths)-8})")
                            f.write("\n")
                    f.write("\n")
                
                f.write("-" * 80 + "\n\n")
        
        # ====================================================================
        # FOOTER
        # ====================================================================
        
        f.write("="*80 + "\n")
        f.write("END OF ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"\nFor detailed data:\n")
        f.write(f"  - CSV: synthetic_baselines_analysis.csv\n")
        f.write(f"  - JSON: naturalistic_baselines_analysis.json\n")
        f.write(f"  - Visualizations: *.png files\n")
    
    print(f"\n✓ Generated summary text file: ANALYSIS_SUMMARY.txt")


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def get_next_run_id(output_base_dir: Path) -> str:
    """
    Get next incremental 3-digit run ID
    
    Scans existing run_XXX directories and returns next available number
    """
    if not output_base_dir.exists():
        return "001"
    
    # Find all existing run_XXX directories
    existing_runs = []
    for item in output_base_dir.iterdir():
        if item.is_dir() and item.name.startswith('run_'):
            try:
                run_num = int(item.name.split('_')[1])
                existing_runs.append(run_num)
            except (IndexError, ValueError):
                continue
    
    # Get next number
    if existing_runs:
        next_num = max(existing_runs) + 1
    else:
        next_num = 1
    
    return f"{next_num:03d}"


def analyze_v4_results(output_base_dir: Optional[Path] = None):
    """
    Main analysis function for v4 results
    
    Analyzes both synthetic and naturalistic baselines
    Creates output in incremental run_XXX directories
    """
    print("="*80)
    print("S64 VALIDATION RESULTS ANALYSIS - V4 Structure")
    print("="*80)
    
    # Setup paths
    script_dir = Path(__file__).parent
    v4_dir = script_dir.parent / 'v4'
    
    if not v4_dir.exists():
        print(f"\n❌ V4 directory not found: {v4_dir}")
        return
    
    # Setup output directory with incremental run ID
    if output_base_dir is None:
        output_base_dir = script_dir / 'analysis_output'
    output_base_dir.mkdir(exist_ok=True)
    
    # Get next run ID
    run_id = get_next_run_id(output_base_dir)
    output_dir = output_base_dir / f"run_{run_id}"
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n📁 Output directory: run_{run_id}")
    print(f"   Path: {output_dir}")
    
    # Load results
    print(f"\n📂 Loading results from: {v4_dir}")
    results = load_v4_results(v4_dir)
    print(f"✓ Loaded {len(results)} result files")
    
    # Categorize baselines
    synthetic_baselines, naturalistic_baselines = categorize_baselines(v4_dir)
    print(f"\n📊 Found {len(synthetic_baselines)} synthetic baselines: {', '.join(synthetic_baselines)}")
    print(f"📊 Found {len(naturalistic_baselines)} naturalistic baselines: {', '.join(naturalistic_baselines)}")
    
    # ========================================================================
    # ANALYZE SYNTHETIC BASELINES
    # ========================================================================
    
    if synthetic_baselines:
        print("\n" + "="*80)
        print("SYNTHETIC BASELINE ANALYSIS")
        print("="*80)
        
        all_synthetic_data = []
        
        for baseline_id in synthetic_baselines:
            print(f"\n📈 Analyzing {baseline_id}: {BASELINE_LABELS.get(baseline_id, baseline_id)}")
            
            df = analyze_synthetic_baseline(baseline_id, results)
            if not df.empty:
                all_synthetic_data.append(df)
                
                # Print summary
                print(f"   Results: {len(df)} model/embedding combinations")
                avg_f1 = df['c_f1'].mean()
                avg_tus = df['tus'].mean()
                print(f"   Average Channel C F1: {avg_f1:.3f}")
                print(f"   Average TUS: {avg_tus:.2f}")
        
        if all_synthetic_data:
            # Combine all synthetic data
            synthetic_df = pd.concat(all_synthetic_data, ignore_index=True)
            
            # Save to CSV
            synthetic_csv = output_dir / 'synthetic_baselines_analysis.csv'
            synthetic_df.to_csv(synthetic_csv, index=False)
            print(f"\n✓ Saved synthetic analysis to: synthetic_baselines_analysis.csv")
            
            # Generate summary by LLM model (console view)
            llm_summary = synthetic_df.groupby('model_name').agg({
                'llm_tus': 'mean',  # Average across embeddings
                'c_f1': 'mean',
                'c_precision': 'mean',
                'c_recall': 'mean'
            }).round(3)
            llm_summary['tus_pct'] = (llm_summary['llm_tus'] / 70.0 * 100.0).round(2)
            llm_summary = llm_summary.sort_values('llm_tus', ascending=False)
            
            # Generate summary by Embedding
            embed_summary = synthetic_df.groupby('embedding_name').agg({
                'embed_tus': 'mean',  # Average across models
                'embed_f1': 'mean',
                'embed_precision': 'mean',
                'embed_recall': 'mean'
            }).round(3)
            embed_summary['tus_pct'] = (embed_summary['embed_tus'] / 70.0 * 100.0).round(2)
            embed_summary = embed_summary.sort_values('embed_tus', ascending=False)
            
            print("\n" + "="*80)
            print("LLM PERFORMANCE (Channel C)")
            print("="*80)
            print(llm_summary.to_string())
            
            print("\n" + "="*80)
            print("EMBEDDING PERFORMANCE (Channel A/A+)")
            print("="*80)
            print(embed_summary.to_string())
            
            # Save summaries
            llm_summary.to_csv(output_dir / 'llm_summary.csv')
            embed_summary.to_csv(output_dir / 'embedding_summary.csv')
            print(f"\n✓ Saved llm_summary.csv and embedding_summary.csv")
    
    # ========================================================================
    # ANALYZE NATURALISTIC BASELINES
    # ========================================================================
    
    if naturalistic_baselines:
        print("\n" + "="*80)
        print("NATURALISTIC BASELINE ANALYSIS")
        print("="*80)
        
        naturalistic_results = {}
        
        for baseline_id in naturalistic_baselines:
            print(f"\n🔍 Analyzing {baseline_id} (Consensus Analysis)")
            
            analysis = analyze_naturalistic_baseline(baseline_id, results)
            naturalistic_results[baseline_id] = analysis
            
            # Print consensus summary
            print(f"   Total runs: {analysis['total_runs']}")
            print(f"   Models: {', '.join([MODEL_NAMES.get(m, m) for m in analysis['models']])}")
            print(f"   Embeddings: {', '.join([EMBEDDING_NAMES.get(e, e) for e in analysis['embeddings']])}")
            print(f"   Cross-embedding agreement: {analysis['cross_embedding_agreement']:.3f}")
            
            c_consensus = analysis['channel_c_consensus']
            print(f"\n   Channel C Consensus:")
            print(f"      High consensus: {len(c_consensus['high'])} paths")
            print(f"      Moderate: {len(c_consensus['moderate'])} paths")
            print(f"      Low: {len(c_consensus['low'])} paths")
            print(f"      Outliers: {len(c_consensus['outliers'])} paths")
            
            if c_consensus['high']:
                print(f"\n   High Consensus Paths (75%+ agreement):")
                for path, count, ratio in c_consensus['high'][:5]:  # Top 5
                    print(f"      M{path}: {count}/{analysis['total_runs']} ({ratio*100:.0f}%)")
        
        # Save naturalistic results to JSON
        naturalistic_json = output_dir / 'naturalistic_baselines_analysis.json'
        with open(naturalistic_json, 'w', encoding='utf-8') as f:
            json.dump(naturalistic_results, f, indent=2, default=str)
        print(f"\n✓ Saved naturalistic analysis to: naturalistic_baselines_analysis.json")
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nRun ID: {run_id}")
    print(f"All results saved to: {output_dir}")
    
    # Generate comprehensive summary text file
    generate_summary_text(
        synthetic_df if synthetic_baselines and all_synthetic_data else None,
        naturalistic_results if naturalistic_baselines else None,
        output_dir,
        run_id
    )
    
    return run_id, output_dir


if __name__ == '__main__':
    analyze_v4_results()
