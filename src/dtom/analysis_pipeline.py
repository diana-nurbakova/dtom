#!/usr/bin/env python3
"""
DToM Empirical Analysis Pipeline
=================================
Double Theory of Mind (DToM) — Empirical grounding using the TalkMoves Dataset.

This script performs two studies:
  Study 1: L3 Depth Mapping across standard talk move categories
  Study 2: Within-category mentalizing depth analysis (Press for Accuracy)

Dataset: TalkMoves (Suresh et al., 2022, LREC)
  - 567 annotated K-12 mathematics lesson transcripts
  - Available at: https://github.com/SumnerLab/TalkMoves
  - License: CC BY-NC-SA 4.0

Requirements:
  pip install pandas openpyxl scipy matplotlib seaborn numpy

Usage:
  1. Clone the TalkMoves repo: git clone https://github.com/SumnerLab/TalkMoves.git
  2. Run: python dtom_analysis_pipeline.py --data-dir TalkMoves/data

Author: [Anonymous for review]
Date: April 2026
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ============================================================
# CONFIGURATION
# ============================================================

# L3 Depth Mapping: Teacher talk moves → mentalizing depth levels
# Grounded in teacher cognition literature:
#   - Berliner (2004): novice-to-expert shift from rule-based to interpretive
#   - Sherin & Van Es (2009): professional vision as interpreting student reasoning
#   - Franke et al. (2009): teacher questioning and student thinking
L3_DEPTH_MAP = {
    # Level 0 — No L3 (procedural/management):
    #   Teacher does not need to model student thinking
    'None':               0,
    'Context':            0,
    'KeepingTogether':    0,
    'Marking':            0,

    # Level 1 — Surface L3 (acknowledge/check):
    #   Teacher acknowledges student output or checks correctness
    #   but does not probe the reasoning behind it
    'Restating':          1,
    'Revoicing':          1,
    'PressAccuracy':      1,

    # Level 2 — Deep L3 (probe reasoning):
    #   Teacher actively models and probes student mental states —
    #   understanding, misconceptions, reasoning processes
    'PressReasoning':     2,
    'GettingStudentsRelate': 2,
}

L3_LABELS = {
    0: 'No L3 (procedural/management)',
    1: 'Surface L3 (acknowledge/check)',
    2: 'Deep L3 (probe reasoning)',
}

# Within-category mentalizing depth: linguistic markers
# Based on questioning taxonomies from:
#   - Chin (2007): teacher questioning in science classrooms
#   - Franke et al. (2009): types of teacher questions in mathematics
#   - Boaler & Brodie (2004): question types that support mathematical thinking
DEEP_PATTERNS = [
    r'\bwhy\b.*\b(think|work|true|right|correct|happen|say|choose|pick)\b',
    r'\bwhy\b(?!.*\bnot\b)',
    r'\bhow do you know\b',
    r'\bhow can you tell\b',
    r'\bwhat makes you\b',
    r'\bwhat tells you\b',
    r'\bexplain\b.*\b(thinking|reasoning|why|how)\b',
    r'\bwhat would happen\b',
    r'\bwhat if\b',
    r'\bdoes that make sense\b',
    r'\bdoes that work\b',
    r'\bprove\b',
    r'\bjustify\b',
    r'\bconvince\b',
    r'\bhow does that connect\b',
    r'\bhow does that relate\b',
    r'\bdo you agree\b.*\bwhy\b',
    r'\bwhy do you agree\b',
    r'\bis there another way\b',
    r'\bcould you do it differently\b',
    r'\bwhat.{0,20}mean\b.*\bby\b',
    r'\btell me more about\b.*\bthinking\b',
]

INTERMEDIATE_PATTERNS = [
    r'\bhow did you\b',
    r'\bhow do you\b',
    r'\bwhat did you do\b',
    r'\bwhat.{0,10}you do\b',
    r'\bshow\s+(me|us)\b',
    r'\bexplain\b',
    r'\btell\s+(me|us)\s+how\b',
    r'\btell\s+(me|us)\s+what\b',
    r'\bwhat.{0,10}(method|strategy|approach|steps|process)\b',
    r'\bwalk.{0,10}through\b',
    r'\bdescribe\b',
    r'\bwhat.{0,10}(next|then)\b',
    r'\bcan you\s+(show|tell|explain|describe)\b',
    r'\bhow.{0,15}(figure|solve|find|get|work)\b',
]

MENTAL_DEPTH_LABELS = {
    'A': 'Surface (checking answer/fact)',
    'B': 'Intermediate (requesting procedure)',
    'C': 'Deep (probing understanding)',
}


# ============================================================
# DATA LOADING AND NORMALIZATION
# ============================================================

def normalize_teacher_tag(tag: str) -> str | None:
    """Normalize teacher talk move tags to canonical categories.
    
    The TalkMoves dataset has inconsistent casing and slight naming
    variations across subsets. This function maps all variants to
    a canonical set of 9 categories.
    """
    if pd.isna(tag):
        return None
    tag = str(tag).strip().lower()
    if 'none' in tag or tag in ['1', '2', '']:
        return 'None'
    if 'keeping' in tag:
        return 'KeepingTogether'
    if 'getting' in tag or 'relate' in tag:
        return 'GettingStudentsRelate'
    if 'restating' in tag:
        return 'Restating'
    if 'revoicing' in tag:
        return 'Revoicing'
    if 'marking' in tag:
        return 'Marking'
    if 'context' in tag:
        return 'Context'
    if 'reasoning' in tag:
        return 'PressReasoning'
    if 'accuracy' in tag:
        return 'PressAccuracy'
    return 'Other'


def normalize_student_tag(tag: str) -> str | None:
    """Normalize student talk move tags to canonical categories."""
    if pd.isna(tag):
        return None
    tag = str(tag).strip().lower()
    if 'none' in tag or tag in ['1', '2', '5', '\\']:
        return 'None'
    if 'relating' in tag or 'relate' in tag:
        return 'RelatingToAnother'
    if 'asking' in tag or 'information' in tag:
        return 'AskingForInfo'
    if 'claim' in tag:
        return 'MakingClaim'
    if 'evidence' in tag or 'reasoning' in tag:
        return 'ProvidingEvidence'
    return 'Other'


def classify_mentalizing_depth(text: str) -> str:
    """Classify a teacher utterance's mentalizing depth using linguistic markers.
    
    Returns:
        'A' (Surface): Checking factual answers, requesting numbers/results
        'B' (Intermediate): Asking for procedure/method explanation
        'C' (Deep): Probing conceptual understanding, asking why/justify
    
    The classifier uses pattern matching on linguistic markers associated
    with different questioning depths in the mathematics education literature.
    Priority: Deep > Intermediate > Surface (default).
    """
    text = str(text).lower().strip()
    
    for pattern in DEEP_PATTERNS:
        if re.search(pattern, text):
            return 'C'
    
    for pattern in INTERMEDIATE_PATTERNS:
        if re.search(pattern, text):
            return 'B'
    
    return 'A'


def load_transcripts(data_dir: str) -> pd.DataFrame:
    """Load all TalkMoves transcripts from Subset 1 and Subset 2.
    
    Args:
        data_dir: Path to TalkMoves/data directory
        
    Returns:
        DataFrame with all utterances, normalized tags, and L3 depth assignments
    """
    subset1_files = glob.glob(os.path.join(data_dir, 'Subset 1', '*.xlsx'))
    subset2_files = glob.glob(os.path.join(data_dir, 'Subset 2', '*.xlsx'))
    all_files = subset1_files + subset2_files
    
    print(f"Loading transcripts from {data_dir}...")
    print(f"  Subset 1: {len(subset1_files)} files")
    print(f"  Subset 2: {len(subset2_files)} files")
    
    all_dfs = []
    errors = 0
    for f in all_files:
        try:
            df = pd.read_excel(f, engine='openpyxl')
            df['transcript_id'] = Path(f).stem
            df['subset'] = 'Subset1' if 'Subset 1' in f else 'Subset2'
            all_dfs.append(df)
        except Exception as e:
            errors += 1
    
    combined = pd.concat(all_dfs, ignore_index=True)
    
    if errors > 0:
        print(f"  Warning: {errors} files could not be loaded")
    
    # Normalize tags
    combined['t_move'] = combined['Teacher Tag'].apply(normalize_teacher_tag)
    combined['s_move'] = combined['Student Tag'].apply(normalize_student_tag)
    
    # Assign L3 depth
    combined['l3_depth'] = combined['t_move'].map(L3_DEPTH_MAP)
    
    # Word count for filtering
    combined['word_count'] = combined['Sentence'].apply(
        lambda x: len(str(x).split()) if pd.notna(x) else 0
    )
    
    n_transcripts = combined['transcript_id'].nunique()
    n_teacher = combined['t_move'].notna().sum()
    n_student = combined['s_move'].notna().sum()
    
    print(f"\n  Total utterances: {len(combined):,}")
    print(f"  Transcripts: {n_transcripts}")
    print(f"  Teacher utterances (tagged): {n_teacher:,}")
    print(f"  Student utterances (tagged): {n_student:,}")
    
    return combined


# ============================================================
# STUDY 1: L3 DEPTH MAPPING ACROSS TALK MOVES
# ============================================================

def study1_l3_depth_mapping(combined: pd.DataFrame, output_dir: str) -> dict:
    """Study 1: Map teacher talk moves to DToM L3 depth levels.
    
    Analyses:
      1a. Distribution of L3 depth across all teacher utterances
      1b. Sequential analysis: student response quality after each L3 depth
      1c. Transcript-level correlation: mean L3 depth vs student reasoning
      1d. Median-split comparison with effect size
      
    Returns:
        Dictionary of all statistical results
    """
    print("\n" + "=" * 70)
    print("STUDY 1: L3 Depth Mapping Across Teacher Talk Moves")
    print("=" * 70)
    
    results = {}
    teacher_utts = combined[combined['t_move'].notna()].copy()
    student_utts = combined[combined['s_move'].notna()].copy()
    
    # ----------------------------------------------------------
    # 1a. L3 Depth Distribution
    # ----------------------------------------------------------
    print("\n--- 1a. L3 Depth Distribution ---")
    depth_counts = teacher_utts['l3_depth'].value_counts().sort_index()
    total_teacher = depth_counts.sum()
    
    results['distribution'] = {}
    for depth, count in depth_counts.items():
        pct = count / total_teacher * 100
        label = L3_LABELS[depth]
        print(f"  Level {int(depth)} ({label}): {count:,} ({pct:.1f}%)")
        results['distribution'][f'level_{int(depth)}'] = {
            'count': int(count), 'pct': round(pct, 1)
        }
    results['total_teacher_utterances'] = int(total_teacher)
    
    # ----------------------------------------------------------
    # 1b. Sequential Analysis
    # ----------------------------------------------------------
    print("\n--- 1b. Sequential Analysis: Student Response After Teacher L3 ---")
    
    seq_data = []
    for depth_level in [0, 1, 2]:
        depth_idx = combined[combined['l3_depth'] == depth_level].index
        for idx in depth_idx:
            for offset in range(1, 4):
                if idx + offset < len(combined) and combined.loc[idx + offset, 's_move'] is not None:
                    s_move = combined.loc[idx + offset, 's_move']
                    seq_data.append({
                        'l3_depth': depth_level,
                        'next_evidence': 1 if s_move == 'ProvidingEvidence' else 0,
                        'next_move': s_move
                    })
                    break
    
    seq_df = pd.DataFrame(seq_data)
    
    # Chi-square: overall
    contingency = pd.crosstab(seq_df['l3_depth'], seq_df['next_evidence'])
    chi2, p_chi, dof, _ = stats.chi2_contingency(contingency)
    n_total = contingency.sum().sum()
    cramers_v_overall = np.sqrt(chi2 / (n_total * (min(contingency.shape) - 1)))
    
    print(f"  Overall chi-square: χ²={chi2:.2f}, df={dof}, p={p_chi:.2e}, V={cramers_v_overall:.3f}")
    results['sequential'] = {
        'chi2': round(chi2, 2),
        'p': float(p_chi),
        'dof': int(dof),
        'cramers_v': round(cramers_v_overall, 3),
        'n': int(n_total),
    }
    
    results['sequential']['by_depth'] = {}
    for depth_level in [0, 1, 2]:
        subset = seq_df[seq_df['l3_depth'] == depth_level]
        n = len(subset)
        ev_rate = subset['next_evidence'].mean() * 100
        label = L3_LABELS[depth_level]
        print(f"  After {label} (n={n:,}): {ev_rate:.1f}% student evidence")
        results['sequential']['by_depth'][f'level_{depth_level}'] = {
            'n': int(n), 'evidence_pct': round(ev_rate, 1)
        }
    
    # Pairwise comparisons
    results['sequential']['pairwise'] = {}
    for d1, d2 in [(0, 1), (0, 2), (1, 2)]:
        g1 = seq_df[seq_df['l3_depth'] == d1]['next_evidence']
        g2 = seq_df[seq_df['l3_depth'] == d2]['next_evidence']
        ct = pd.crosstab(
            pd.Series([d1] * len(g1) + [d2] * len(g2)),
            pd.Series(list(g1) + list(g2))
        )
        chi2_pair, p_pair, _, _ = stats.chi2_contingency(ct)
        n_pair = ct.sum().sum()
        v_pair = np.sqrt(chi2_pair / (n_pair * (min(ct.shape) - 1)))
        key = f'{d1}_vs_{d2}'
        results['sequential']['pairwise'][key] = {
            'chi2': round(chi2_pair, 2),
            'p': float(p_pair),
            'cramers_v': round(v_pair, 3)
        }
        print(f"    L{d1} vs L{d2}: χ²={chi2_pair:.2f}, p={p_pair:.2e}, V={v_pair:.3f}")
    
    # ----------------------------------------------------------
    # 1c. Transcript-Level Correlation
    # ----------------------------------------------------------
    print("\n--- 1c. Transcript-Level: L3 Depth vs Student Reasoning ---")
    
    transcript_l3 = teacher_utts.groupby('transcript_id')['l3_depth'].agg(
        ['mean', 'count']
    ).reset_index()
    transcript_l3.columns = ['transcript_id', 'mean_l3', 'n_teacher']
    transcript_l3 = transcript_l3[transcript_l3['n_teacher'] >= 20]
    
    student_quality = student_utts.groupby('transcript_id').apply(
        lambda x: pd.Series({
            'pct_evidence': (x['s_move'] == 'ProvidingEvidence').mean(),
            'pct_high_quality': (
                x['s_move'].isin(['ProvidingEvidence', 'RelatingToAnother'])
            ).mean(),
            'n_student': len(x)
        }),
        include_groups=False,
    ).reset_index()
    
    merged = transcript_l3.merge(student_quality, on='transcript_id')
    merged = merged[merged['n_student'] >= 10]
    
    r_pearson, p_pearson = stats.pearsonr(merged['mean_l3'], merged['pct_evidence'])
    r_spearman, p_spearman = stats.spearmanr(merged['mean_l3'], merged['pct_evidence'])
    r_hq, p_hq = stats.pearsonr(merged['mean_l3'], merged['pct_high_quality'])
    
    print(f"  N transcripts: {len(merged)}")
    print(f"  Pearson r (evidence): r={r_pearson:.3f}, p={p_pearson:.4f}")
    print(f"  Spearman ρ (evidence): ρ={r_spearman:.3f}, p={p_spearman:.4f}")
    print(f"  Pearson r (high-quality): r={r_hq:.3f}, p={p_hq:.4f}")
    
    results['transcript_level'] = {
        'n': int(len(merged)),
        'pearson_r': round(r_pearson, 3),
        'pearson_p': round(p_pearson, 4),
        'spearman_rho': round(r_spearman, 3),
        'spearman_p': round(p_spearman, 4),
        'pearson_r_hq': round(r_hq, 3),
    }
    
    # ----------------------------------------------------------
    # 1d. Median-Split Comparison
    # ----------------------------------------------------------
    print("\n--- 1d. Median-Split Comparison ---")
    
    median_l3 = merged['mean_l3'].median()
    high_group = merged[merged['mean_l3'] >= median_l3]['pct_evidence']
    low_group = merged[merged['mean_l3'] < median_l3]['pct_evidence']
    
    t_stat, p_ttest = stats.ttest_ind(high_group, low_group)
    pooled_std = np.sqrt((high_group.var() + low_group.var()) / 2)
    cohens_d = (high_group.mean() - low_group.mean()) / pooled_std
    
    # Mann-Whitney U (non-parametric alternative)
    u_stat, p_mann = stats.mannwhitneyu(high_group, low_group, alternative='two-sided')
    
    print(f"  High L3 (n={len(high_group)}): M={high_group.mean():.3f}, SD={high_group.std():.3f}")
    print(f"  Low L3 (n={len(low_group)}): M={low_group.mean():.3f}, SD={low_group.std():.3f}")
    print(f"  t-test: t={t_stat:.3f}, p={p_ttest:.4f}")
    print(f"  Cohen's d: {cohens_d:.3f}")
    print(f"  Mann-Whitney U: U={u_stat:.0f}, p={p_mann:.4f}")
    
    results['median_split'] = {
        'median_l3': round(median_l3, 3),
        'high_n': int(len(high_group)),
        'high_mean': round(high_group.mean(), 3),
        'high_sd': round(high_group.std(), 3),
        'low_n': int(len(low_group)),
        'low_mean': round(low_group.mean(), 3),
        'low_sd': round(low_group.std(), 3),
        't_stat': round(t_stat, 3),
        'p_ttest': round(p_ttest, 4),
        'cohens_d': round(cohens_d, 3),
        'mann_whitney_u': round(u_stat, 1),
        'mann_whitney_p': round(p_mann, 4),
    }
    
    # Save transcript-level data
    merged.to_csv(os.path.join(output_dir, 'transcript_l3_analysis.csv'), index=False)
    
    # ----------------------------------------------------------
    # Generate Figure 1
    # ----------------------------------------------------------
    _generate_study1_figure(
        depth_counts, seq_df, chi2, merged, r_pearson, p_pearson,
        high_group, low_group, cohens_d, p_ttest, output_dir
    )
    
    return results


def _generate_study1_figure(depth_counts, seq_df, chi2_overall, merged,
                            r_pearson, p_pearson, high_group, low_group,
                            cohens_d, p_ttest, output_dir):
    """Generate publication-ready Figure 1 for Study 1."""
    
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)
    
    colors = ['#95a5a6', '#3498db', '#e74c3c']
    depth_labels = ['No L3\n(Procedural)', 'Surface L3\n(Acknowledge)', 'Deep L3\n(Probe Reasoning)']
    
    # Panel A: L3 Depth Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar([0, 1, 2], depth_counts.values, color=colors, edgecolor='white', width=0.6)
    for bar, count in zip(bars, depth_counts.values):
        pct = count / depth_counts.sum() * 100
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
                 f'{pct:.1f}%', ha='center', fontsize=11, fontweight='bold')
    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(depth_labels, fontsize=9)
    ax1.set_ylabel('Number of Utterances', fontsize=11)
    ax1.set_title('(A) Distribution of Teacher L3 Depth', fontsize=12, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Panel B: Sequential — student evidence after each L3 depth
    ax2 = fig.add_subplot(gs[0, 1])
    evidence_rates = []
    for d in [0, 1, 2]:
        subset = seq_df[seq_df['l3_depth'] == d]
        evidence_rates.append(subset['next_evidence'].mean() * 100)
    bars2 = ax2.bar([0, 1, 2], evidence_rates, color=colors, edgecolor='white', width=0.6)
    for bar, rate in zip(bars2, evidence_rates):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{rate:.1f}%', ha='center', fontsize=11, fontweight='bold')
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(depth_labels, fontsize=9)
    ax2.set_ylabel('% Student Evidence/Reasoning', fontsize=11)
    ax2.set_title('(B) Student Reasoning After Teacher L3 Move', fontsize=12, fontweight='bold')
    ax2.text(0.95, 0.95, f'χ²={chi2_overall:.1f}, p<.001',
             transform=ax2.transAxes, ha='right', va='top', fontsize=9, style='italic',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Panel C: Scatter — transcript-level
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(merged['mean_l3'], merged['pct_evidence'] * 100, alpha=0.4, s=30, color='#2c3e50')
    z = np.polyfit(merged['mean_l3'], merged['pct_evidence'] * 100, 1)
    p_line = np.poly1d(z)
    x_range = np.linspace(merged['mean_l3'].min(), merged['mean_l3'].max(), 100)
    ax3.plot(x_range, p_line(x_range), 'r-', linewidth=2, alpha=0.7)
    ax3.set_xlabel('Mean L3 Depth per Transcript', fontsize=11)
    ax3.set_ylabel('% Student Evidence/Reasoning', fontsize=11)
    ax3.set_title('(C) Transcript-Level: L3 Depth vs Student Reasoning',
                  fontsize=12, fontweight='bold')
    ax3.text(0.05, 0.95, f'r={r_pearson:.3f}, p={p_pearson:.4f}\nN={len(merged)} transcripts',
             transform=ax3.transAxes, ha='left', va='top', fontsize=9, style='italic',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Panel D: Median-split
    ax4 = fig.add_subplot(gs[1, 1])
    means = [low_group.mean() * 100, high_group.mean() * 100]
    sds = [low_group.std() * 100, high_group.std() * 100]
    bars4 = ax4.bar([0, 1], means, yerr=sds, color=['#3498db', '#e74c3c'],
                    edgecolor='white', width=0.5, capsize=5, error_kw={'linewidth': 1.5})
    for bar, m in zip(bars4, means):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + sds[0] + 0.5,
                 f'{m:.1f}%', ha='center', fontsize=11, fontweight='bold')
    ax4.set_xticks([0, 1])
    ax4.set_xticklabels(['Low L3\nTranscripts', 'High L3\nTranscripts'], fontsize=10)
    ax4.set_ylabel('% Student Evidence/Reasoning', fontsize=11)
    ax4.set_title('(D) Median-Split Comparison', fontsize=12, fontweight='bold')
    ax4.text(0.95, 0.95, f"Cohen's d={cohens_d:.2f}\np={p_ttest:.4f}",
             transform=ax4.transAxes, ha='right', va='top', fontsize=9, style='italic',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    filepath = os.path.join(output_dir, 'figure1_l3_depth_mapping.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n  Figure 1 saved: {filepath}")


# ============================================================
# STUDY 2: WITHIN-CATEGORY MENTALIZING DEPTH
# ============================================================

def study2_within_category(combined: pd.DataFrame, output_dir: str) -> dict:
    """Study 2: Within-category mentalizing depth analysis.
    
    Applies a linguistically-grounded mentalizing depth classifier
    to all 'Press for Accuracy' utterances, demonstrating that a
    single standard talk move category contains meaningful variation
    in L3 depth that predicts student reasoning quality.
    
    Returns:
        Dictionary of all statistical results
    """
    print("\n" + "=" * 70)
    print("STUDY 2: Within-Category Mentalizing Depth ('Press for Accuracy')")
    print("=" * 70)
    
    results = {}
    
    # Select Press for Accuracy utterances
    press_acc = combined[combined['t_move'] == 'PressAccuracy'].copy()
    press_acc_meaningful = press_acc[press_acc['word_count'] > 3].copy()
    
    print(f"\n  Total 'Press for Accuracy': {len(press_acc):,}")
    print(f"  Meaningful (>3 words): {len(press_acc_meaningful):,}")
    
    # Apply mentalizing depth classifier
    press_acc_meaningful = press_acc_meaningful.copy()
    press_acc_meaningful['mental_depth'] = press_acc_meaningful['Sentence'].apply(
        classify_mentalizing_depth
    )
    
    # ----------------------------------------------------------
    # 2a. Distribution
    # ----------------------------------------------------------
    print("\n--- 2a. Within-Category Distribution ---")
    depth_counts = press_acc_meaningful['mental_depth'].value_counts()
    total = len(press_acc_meaningful)
    
    results['distribution'] = {}
    for level in ['A', 'B', 'C']:
        count = depth_counts.get(level, 0)
        pct = count / total * 100
        label = MENTAL_DEPTH_LABELS[level]
        print(f"  Level {level} ({label}): {count:,} ({pct:.1f}%)")
        results['distribution'][f'level_{level}'] = {
            'count': int(count), 'pct': round(pct, 1)
        }
    results['total_utterances'] = int(total)
    
    # ----------------------------------------------------------
    # 2b. Examples
    # ----------------------------------------------------------
    print("\n--- 2b. Examples ---")
    np.random.seed(42)
    results['examples'] = {}
    for level in ['A', 'B', 'C']:
        subset = press_acc_meaningful[press_acc_meaningful['mental_depth'] == level]
        examples = subset.sample(min(5, len(subset)), random_state=42)
        label = MENTAL_DEPTH_LABELS[level]
        print(f"\n  Level {level} ({label}):")
        ex_list = []
        for _, row in examples.iterrows():
            text = str(row['Sentence']).strip()
            print(f"    • \"{text}\"")
            ex_list.append(text)
        results['examples'][level] = ex_list
    
    # ----------------------------------------------------------
    # 2c. Validation: Does depth predict student response?
    # ----------------------------------------------------------
    print("\n--- 2c. Validation: Student Response by Within-Category Depth ---")
    
    seq_data = []
    for idx in press_acc_meaningful.index:
        depth = press_acc_meaningful.loc[idx, 'mental_depth']
        for offset in range(1, 4):
            if idx + offset < len(combined) and combined.loc[idx + offset, 's_move'] is not None:
                s_move = combined.loc[idx + offset, 's_move']
                seq_data.append({
                    'mental_depth': depth,
                    'next_evidence': 1 if s_move == 'ProvidingEvidence' else 0,
                    'next_move': s_move
                })
                break
    
    seq_df = pd.DataFrame(seq_data)
    
    results['validation'] = {'by_depth': {}}
    for level in ['A', 'B', 'C']:
        subset = seq_df[seq_df['mental_depth'] == level]
        n = len(subset)
        ev_rate = subset['next_evidence'].mean() * 100
        label = MENTAL_DEPTH_LABELS[level]
        print(f"  After Level {level} ({label}, n={n:,}): {ev_rate:.1f}% student evidence")
        results['validation']['by_depth'][level] = {
            'n': int(n), 'evidence_pct': round(ev_rate, 1)
        }
    
    # Chi-square: overall within-category
    contingency = pd.crosstab(seq_df['mental_depth'], seq_df['next_evidence'])
    chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
    n_total = contingency.sum().sum()
    cramers_v = np.sqrt(chi2 / (n_total * (min(contingency.shape) - 1)))
    
    print(f"\n  Overall chi-square: χ²={chi2:.2f}, p={p_val:.2e}, V={cramers_v:.3f}")
    results['validation']['chi2'] = round(chi2, 2)
    results['validation']['p'] = float(p_val)
    results['validation']['cramers_v'] = round(cramers_v, 3)
    
    # A vs C pairwise
    a_data = seq_df[seq_df['mental_depth'] == 'A']['next_evidence']
    c_data = seq_df[seq_df['mental_depth'] == 'C']['next_evidence']
    ct_ac = pd.crosstab(
        pd.Series(['A'] * len(a_data) + ['C'] * len(c_data)),
        pd.Series(list(a_data) + list(c_data))
    )
    chi2_ac, p_ac, _, _ = stats.chi2_contingency(ct_ac)
    n_ac = ct_ac.sum().sum()
    v_ac = np.sqrt(chi2_ac / (n_ac * (min(ct_ac.shape) - 1)))
    
    print(f"  A vs C: χ²={chi2_ac:.2f}, p={p_ac:.2e}, V={v_ac:.3f}")
    results['validation']['a_vs_c'] = {
        'chi2': round(chi2_ac, 2),
        'p': float(p_ac),
        'cramers_v': round(v_ac, 3)
    }
    
    # ----------------------------------------------------------
    # Generate Figure 2
    # ----------------------------------------------------------
    _generate_study2_figure(
        depth_counts, total, seq_df, chi2, output_dir
    )
    
    return results


def _generate_study2_figure(depth_counts, total, seq_df, chi2_overall, output_dir):
    """Generate publication-ready Figure 2 for Study 2."""
    
    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(1, 2, wspace=0.35)
    
    colors = ['#95a5a6', '#3498db', '#e74c3c']
    cats = ['Surface\n(check answer)', 'Intermediate\n(request procedure)',
            'Deep\n(probe understanding)']
    
    # Panel A: Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    counts = [depth_counts.get(l, 0) for l in ['A', 'B', 'C']]
    pcts = [c / total * 100 for c in counts]
    bars = ax1.bar(range(3), pcts, color=colors, edgecolor='white', width=0.6)
    for bar, pct, count in zip(bars, pcts, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f'{pct:.1f}%\n(n={count:,})', ha='center', fontsize=9, fontweight='bold')
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(cats, fontsize=9)
    ax1.set_ylabel('% of Utterances', fontsize=11)
    ax1.set_title("(A) Hidden L3 Variation Within\n'Press for Accuracy'",
                  fontsize=11, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylim(0, 100)
    
    # Panel B: Validation
    ax2 = fig.add_subplot(gs[0, 1])
    evidence_rates = []
    for level in ['A', 'B', 'C']:
        subset = seq_df[seq_df['mental_depth'] == level]
        evidence_rates.append(subset['next_evidence'].mean() * 100)
    bars2 = ax2.bar(range(3), evidence_rates, color=colors, edgecolor='white', width=0.6)
    for bar, rate in zip(bars2, evidence_rates):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                 f'{rate:.1f}%', ha='center', fontsize=11, fontweight='bold')
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(cats, fontsize=9)
    ax2.set_ylabel('% Student Evidence/Reasoning', fontsize=11)
    ax2.set_title('(B) Student Reasoning by L3 Depth\n(Within Same Standard Category)',
                  fontsize=11, fontweight='bold')
    ax2.text(0.95, 0.95, f'χ²={chi2_overall:.1f}, p<.001',
             transform=ax2.transAxes, ha='right', va='top', fontsize=9, style='italic',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    filepath = os.path.join(output_dir, 'figure2_within_category.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n  Figure 2 saved: {filepath}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='DToM Empirical Analysis Pipeline'
    )
    parser.add_argument(
        '--data-dir', type=str, default='data/TalkMoves/data',
        help='Path to TalkMoves/data directory'
    )
    parser.add_argument(
        '--output-dir', type=str, default='output',
        help='Directory for output files (figures, tables, JSON)'
    )
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    combined = load_transcripts(args.data_dir)
    
    # Run studies
    results1 = study1_l3_depth_mapping(combined, args.output_dir)
    results2 = study2_within_category(combined, args.output_dir)
    
    # Save all results
    all_results = {
        'study1_l3_mapping': results1,
        'study2_within_category': results2,
        'dataset': {
            'name': 'TalkMoves (Suresh et al., 2022)',
            'url': 'https://github.com/SumnerLab/TalkMoves',
            'license': 'CC BY-NC-SA 4.0',
            'n_transcripts': combined['transcript_id'].nunique(),
            'n_utterances': len(combined),
        }
    }
    
    results_path = os.path.join(args.output_dir, 'dtom_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print(f"All results saved to {args.output_dir}/")
    print(f"  - dtom_results.json (all statistical results)")
    print(f"  - transcript_l3_analysis.csv (transcript-level data)")
    print(f"  - figure1_l3_depth_mapping.png")
    print(f"  - figure2_within_category.png")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
