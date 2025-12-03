#!/usr/bin/env python3
"""
S64 Validation Results Visualizer - V4 Structure

Reads pre-analyzed data from run directories and creates visualizations.
Does NOT perform calculations - uses data from analyze_results_v4.py output.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional

# Import constants only (no analysis functions)
from analyze_results_v4 import (
    BASELINE_LABELS,
    MODEL_NAMES,
    EMBEDDING_NAMES
)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


def plot_synthetic_performance(synthetic_df: pd.DataFrame, output_dir: Path):
    """
    Create comprehensive performance plots for synthetic baselines.
    Uses pre-calculated data from the CSV - no recalculation.
    LLM and Embedding performance are shown separately (independent architectures).
    """
    # 1. LLM TUS (Channel C) - Average across embeddings
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Check if llm_tus column exists, otherwise calculate from tus
    if 'llm_tus' in synthetic_df.columns:
        llm_tus = synthetic_df.groupby('model_name')['llm_tus'].mean().sort_values(ascending=False)
        max_tus = 70.0
    else:
        # Fallback for older data
        llm_tus = synthetic_df.groupby('model_name')['tus'].mean().sort_values(ascending=False)
        max_tus = 70.0
    
    llm_tus_pct = (llm_tus / max_tus * 100.0)
    
    bars = ax.barh(range(len(llm_tus_pct)), llm_tus_pct.values)
    ax.set_yticks(range(len(llm_tus_pct)))
    ax.set_yticklabels(llm_tus_pct.index)
    ax.set_xlabel('TUS (% of max 70)')
    ax.set_title('LLM Performance - TUS (Channel C)')
    ax.set_xlim(0, 105)
    ax.grid(axis='x', alpha=0.3)
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    for i, (model, score) in enumerate(llm_tus_pct.items()):
        ax.text(score + 1, i, f'{score:.1f}%', va='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'llm_tus.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Embedding TUS (Channel A and A+) - Average across models
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    # Channel A
    if 'embed_a_tus' in synthetic_df.columns:
        embed_a_tus = synthetic_df.groupby('embedding_name')['embed_a_tus'].mean().sort_values(ascending=False)
    else:
        embed_a_tus = synthetic_df.groupby('embedding_name')['embed_f1'].mean().sort_values(ascending=False) * 70
    
    embed_a_pct = (embed_a_tus / max_tus * 100.0)
    bars = axes[0].barh(range(len(embed_a_pct)), embed_a_pct.values)
    axes[0].set_yticks(range(len(embed_a_pct)))
    axes[0].set_yticklabels(embed_a_pct.index)
    axes[0].set_xlabel('TUS (% of max 70)')
    axes[0].set_title('Embedding TUS - Channel A')
    axes[0].set_xlim(0, 105)
    axes[0].grid(axis='x', alpha=0.3)
    colors = plt.cm.Set2.colors[:len(bars)]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    for i, (embed, score) in enumerate(embed_a_pct.items()):
        axes[0].text(score + 1, i, f'{score:.1f}%', va='center')
    
    # Channel A+
    if 'embed_aplus_tus' in synthetic_df.columns:
        embed_ap_tus = synthetic_df.groupby('embedding_name')['embed_aplus_tus'].mean().sort_values(ascending=False)
    else:
        embed_ap_tus = synthetic_df.groupby('embedding_name')['embed_f1'].mean().sort_values(ascending=False) * 70
    
    embed_ap_pct = (embed_ap_tus / max_tus * 100.0)
    bars = axes[1].barh(range(len(embed_ap_pct)), embed_ap_pct.values)
    axes[1].set_yticks(range(len(embed_ap_pct)))
    axes[1].set_yticklabels(embed_ap_pct.index)
    axes[1].set_xlabel('TUS (% of max 70)')
    axes[1].set_title('Embedding TUS - Channel A+')
    axes[1].set_xlim(0, 105)
    axes[1].grid(axis='x', alpha=0.3)
    colors = plt.cm.Pastel1.colors[:len(bars)]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    for i, (embed, score) in enumerate(embed_ap_pct.items()):
        axes[1].text(score + 1, i, f'{score:.1f}%', va='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'embedding_tus.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. F1 Score Heatmap (Model × Baseline) - Channel C only
    fig, ax = plt.subplots(figsize=(14, 8))
    
    heatmap_data = synthetic_df.pivot_table(
        values='c_f1',
        index='model_name',
        columns='baseline_label',
        aggfunc='mean'
    )
    
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', 
                center=0.5, vmin=0, vmax=1.0, ax=ax, cbar_kws={'label': 'F1 Score'})
    ax.set_title('LLM F1 Score by Baseline (Channel C)')
    ax.set_xlabel('Baseline')
    ax.set_ylabel('Model')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'llm_f1_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Channel Synergy (LLM ↔ Embedding Agreement) - for reference
    fig, ax = plt.subplots(figsize=(12, 6))
    synergy_data = synthetic_df.groupby('model_name')['channel_synergy'].mean().sort_values(ascending=False)
    
    bars = ax.barh(range(len(synergy_data)), synergy_data.values)
    ax.set_yticks(range(len(synergy_data)))
    ax.set_yticklabels(synergy_data.index)
    ax.set_xlabel('Channel Synergy (A/A+ ∩ C Agreement)')
    ax.set_title('LLM-Embedding Agreement (for reference only)')
    ax.set_xlim(0, 1.0)
    ax.grid(axis='x', alpha=0.3)
    
    colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    for i, (model, score) in enumerate(synergy_data.items()):
        ax.text(score + 0.02, i, f'{score:.2f}', va='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'channel_synergy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. LLM Precision-Recall (Channel C only)
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Distinct colors for LLM models
    llm_colors = {
        'Gemini 3.0 Pro': '#FF6B00',      # Orange
        'Claude Sonnet 4.5': '#8B4513',   # Brown
        'Claude Opus 4.1': '#9932CC',     # Purple
        'Claude Haiku 4.5': '#DC143C',    # Crimson
        'ChatGPT 5.1': '#228B22',         # Green
        'DeepSeek': '#1E90FF'             # Blue
    }
    
    # Add small jitter to prevent overlapping points
    jitter_amount = 0.02
    
    for model_name in synthetic_df['model_name'].unique():
        model_data = synthetic_df[synthetic_df['model_name'] == model_name]
        # Add jitter
        jitter_x = np.random.uniform(-jitter_amount, jitter_amount, len(model_data))
        jitter_y = np.random.uniform(-jitter_amount, jitter_amount, len(model_data))
        
        color = llm_colors.get(model_name, '#888888')
        ax.scatter(
            model_data['c_recall'].values + jitter_x, 
            model_data['c_precision'].values + jitter_y, 
            label=model_name, alpha=0.7, s=120, c=color, edgecolors='white', linewidth=0.5
        )
    
    ax.set_xlabel('Recall (detecting true positives)', fontsize=12)
    ax.set_ylabel('Precision (avoiding false positives)', fontsize=12)
    ax.set_title('LLM Precision-Recall (Channel C)', fontsize=14)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect Balance')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'llm_precision_recall.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Embedding Precision-Recall (Channels A and A+ shown separately)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
    
    # Distinct colors for embedding backends
    embed_colors = {
        'E5-Large (Local)': '#E74C3C',    # Red
        'Ada-002 (OpenAI)': '#3498DB',    # Blue
        'Cohere v3.0': '#2ECC71'          # Green
    }
    
    # Channel A
    for embed_name in synthetic_df['embedding_name'].unique():
        embed_data = synthetic_df[synthetic_df['embedding_name'] == embed_name]
        jitter_x = np.random.uniform(-jitter_amount, jitter_amount, len(embed_data))
        jitter_y = np.random.uniform(-jitter_amount, jitter_amount, len(embed_data))
        
        color = embed_colors.get(embed_name, '#888888')
        axes[0].scatter(
            embed_data['embed_a_recall'].values + jitter_x, 
            embed_data['embed_a_precision'].values + jitter_y, 
            label=embed_name, alpha=0.7, s=120, c=color, edgecolors='white', linewidth=0.5
        )
    
    axes[0].set_xlabel('Recall (detecting true positives)', fontsize=12)
    axes[0].set_ylabel('Precision (avoiding false positives)', fontsize=12)
    axes[0].set_title('Embedding Precision-Recall (Channel A)', fontsize=14)
    axes[0].set_xlim(-0.05, 1.05)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[0].grid(alpha=0.3)
    
    # Channel A+
    for embed_name in synthetic_df['embedding_name'].unique():
        embed_data = synthetic_df[synthetic_df['embedding_name'] == embed_name]
        jitter_x = np.random.uniform(-jitter_amount, jitter_amount, len(embed_data))
        jitter_y = np.random.uniform(-jitter_amount, jitter_amount, len(embed_data))
        
        color = embed_colors.get(embed_name, '#888888')
        axes[1].scatter(
            embed_data['embed_aplus_recall'].values + jitter_x, 
            embed_data['embed_aplus_precision'].values + jitter_y, 
            label=embed_name, alpha=0.7, s=120, c=color, edgecolors='white', linewidth=0.5
        )
    
    axes[1].set_xlabel('Recall (detecting true positives)', fontsize=12)
    axes[1].set_title('Embedding Precision-Recall (Channel A+)', fontsize=14)
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect Balance')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'embedding_precision_recall.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Generated 6 synthetic baseline visualizations")


def plot_naturalistic_consensus(naturalistic_results: dict, output_dir: Path):
    """
    Create consensus visualizations for naturalistic baselines.
    Uses pre-analyzed data from JSON.
    """
    for baseline_id, analysis in naturalistic_results.items():
        print(f"  Visualizing {baseline_id}...")
        
        # 1. Consensus Distribution (Channel C)
        c_consensus = analysis['channel_c_consensus']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        categories = ['High\n(75%+)', 'Moderate\n(50-74%)', 'Low\n(25-49%)', 'Outliers\n(<25%)']
        counts = [len(c_consensus['high']), len(c_consensus['moderate']), 
                 len(c_consensus['low']), len(c_consensus['outliers'])]
        colors = ['#2ecc71', '#f39c12', '#e74c3c', '#95a5a6']
        
        bars = ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Number of Paths')
        ax.set_title(f'{baseline_id} - Channel C Consensus Distribution')
        ax.grid(axis='y', alpha=0.3)
        
        # Add count labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{baseline_id}_consensus_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Top Consensus Paths
        if c_consensus['high']:
            fig, ax = plt.subplots(figsize=(12, max(6, len(c_consensus['high']) * 0.4)))
            
            paths = [f"M{path}" for path, _, _ in c_consensus['high']]
            agreements = [ratio * 100 for _, _, ratio in c_consensus['high']]
            
            y_pos = range(len(paths))
            bars = ax.barh(y_pos, agreements, color='#2ecc71', alpha=0.7, edgecolor='black')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(paths)
            ax.set_xlabel('Agreement (%)')
            ax.set_title(f'{baseline_id} - High Consensus Paths (75%+ Agreement)')
            ax.set_xlim(0, 105)
            ax.axvline(x=75, color='red', linestyle='--', alpha=0.5, label='75% Threshold')
            ax.legend()
            ax.grid(axis='x', alpha=0.3)
            
            # Add percentage labels
            for i, (bar, agreement) in enumerate(zip(bars, agreements)):
                ax.text(agreement + 1, i, f'{agreement:.0f}%', va='center')
            
            plt.tight_layout()
            plt.savefig(output_dir / f'{baseline_id}_high_consensus_paths.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Summary visualization
        detections_by_model = analysis['detections_by_model']
        detections_by_embedding = analysis['detections_by_embedding']
        
        if len(detections_by_model) >= 2 and len(detections_by_embedding) >= 2:
            models = sorted(detections_by_model.keys())
            embeddings = sorted(detections_by_embedding.keys())
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create summary data
            summary_text = f"{baseline_id} Analysis Summary\n\n"
            summary_text += f"Models analyzed: {len(models)}\n"
            summary_text += f"Embedding backends: {len(embeddings)}\n"
            summary_text += f"Total runs: {analysis['total_runs']}\n\n"
            summary_text += f"Cross-embedding agreement: {analysis['cross_embedding_agreement']:.2f}\n\n"
            summary_text += "High consensus paths:\n"
            for path, count, ratio in c_consensus['high'][:10]:
                summary_text += f"  M{path}: {ratio*100:.0f}%\n"
            
            ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                   fontsize=12, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_dir / f'{baseline_id}_summary.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"✓ Generated naturalistic baseline visualizations")


def get_latest_run_id(output_base_dir: Path) -> Optional[str]:
    """Get the latest run ID from analysis output"""
    if not output_base_dir.exists():
        return None
    
    existing_runs = []
    for item in output_base_dir.iterdir():
        if item.is_dir() and item.name.startswith('run_'):
            try:
                run_num = int(item.name.split('_')[1])
                existing_runs.append((run_num, item.name))
            except (IndexError, ValueError):
                continue
    
    if existing_runs:
        return max(existing_runs, key=lambda x: x[0])[1]
    
    return None


def create_all_visualizations(run_id: Optional[str] = None):
    """
    Main visualization function.
    
    Reads pre-analyzed data from run directory (CSV/JSON files).
    Does NOT re-run analysis.
    """
    print("="*80)
    print("S64 VALIDATION RESULTS VISUALIZER - V4 Structure")
    print("="*80)
    
    # Setup paths
    script_dir = Path(__file__).parent
    output_base_dir = script_dir / 'analysis_output'
    
    # Determine which run to visualize
    if run_id is None:
        run_id = get_latest_run_id(output_base_dir)
        if run_id is None:
            print("\n❌ No analysis runs found. Run analyze_results_v4.py first.")
            return
        print(f"\n📁 Using latest run: {run_id}")
    else:
        if not run_id.startswith('run_'):
            run_id = f"run_{run_id}"
        print(f"\n📁 Using specified run: {run_id}")
    
    output_dir = output_base_dir / run_id
    
    if not output_dir.exists():
        print(f"\n❌ Run directory not found: {output_dir}")
        print(f"   Available runs:")
        for item in output_base_dir.iterdir():
            if item.is_dir() and item.name.startswith('run_'):
                print(f"   - {item.name}")
        return
    
    # ========================================================================
    # LOAD PRE-ANALYZED DATA (no re-calculation!)
    # ========================================================================
    
    synthetic_csv = output_dir / 'synthetic_baselines_analysis.csv'
    naturalistic_json = output_dir / 'naturalistic_baselines_analysis.json'
    
    synthetic_df = None
    naturalistic_results = None
    
    # Load synthetic data from CSV
    if synthetic_csv.exists():
        print(f"\n📂 Loading synthetic data from: {synthetic_csv.name}")
        synthetic_df = pd.read_csv(synthetic_csv)
        print(f"✓ Loaded {len(synthetic_df)} rows")
    else:
        print(f"\n⚠️  No synthetic CSV found: {synthetic_csv.name}")
    
    # Load naturalistic data from JSON
    if naturalistic_json.exists():
        print(f"📂 Loading naturalistic data from: {naturalistic_json.name}")
        with open(naturalistic_json, 'r', encoding='utf-8') as f:
            naturalistic_results = json.load(f)
        print(f"✓ Loaded {len(naturalistic_results)} baselines")
    else:
        print(f"⚠️  No naturalistic JSON found: {naturalistic_json.name}")
    
    # ========================================================================
    # GENERATE VISUALIZATIONS
    # ========================================================================
    
    if synthetic_df is not None and not synthetic_df.empty:
        print("\n" + "="*80)
        print("VISUALIZING SYNTHETIC BASELINES")
        print("="*80)
        plot_synthetic_performance(synthetic_df, output_dir)
    
    if naturalistic_results:
        print("\n" + "="*80)
        print("VISUALIZING NATURALISTIC BASELINES")
        print("="*80)
        plot_naturalistic_consensus(naturalistic_results, output_dir)
    
    print("\n" + "="*80)
    print("✅ VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nRun ID: {run_id}")
    print(f"All visualizations saved to: {output_dir}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        create_all_visualizations(run_id=sys.argv[1])
    else:
        create_all_visualizations()
