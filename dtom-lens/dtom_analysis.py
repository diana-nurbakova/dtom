"""
DToM Analysis Module
====================
Analysis functions for the DToM Lens Streamlit application.

Provides sequential analysis, transcript-level correlation,
within-category analysis, and statistical tests.
"""

import numpy as np
import pandas as pd
from scipy import stats

from dtom_classifier import (
    L3_DEPTH_MAP,
    L3_LABELS,
    MENTAL_DEPTH_LABELS,
    classify_mentalizing_depth,
    normalize_student_tag,
    normalize_teacher_tag,
)


def analyze_sequential(df: pd.DataFrame) -> dict:
    """Turn-level sequential analysis: teacher depth -> next student response.

    Args:
        df: Full transcript DataFrame with 'l3_depth' and 's_move' columns.

    Returns:
        Dictionary with chi-square results and per-depth evidence rates.
    """
    seq_data = []
    for depth_level in [0, 1, 2]:
        depth_idx = df[df['l3_depth'] == depth_level].index
        for idx in depth_idx:
            for offset in range(1, 4):
                next_idx = idx + offset
                if next_idx < len(df) and df.loc[next_idx, 's_move'] is not None:
                    s_move = df.loc[next_idx, 's_move']
                    seq_data.append({
                        'l3_depth': depth_level,
                        'next_evidence': 1 if s_move == 'ProvidingEvidence' else 0,
                        'next_move': s_move,
                    })
                    break

    if not seq_data:
        return {}

    seq_df = pd.DataFrame(seq_data)

    # Chi-square
    contingency = pd.crosstab(seq_df['l3_depth'], seq_df['next_evidence'])
    chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
    n_total = contingency.sum().sum()
    cramers_v = np.sqrt(chi2 / (n_total * (min(contingency.shape) - 1)))

    result = {
        'chi2': round(chi2, 2),
        'p': float(p_val),
        'cramers_v': round(cramers_v, 3),
        'n': int(n_total),
        'by_depth': {},
    }

    for depth_level in [0, 1, 2]:
        subset = seq_df[seq_df['l3_depth'] == depth_level]
        n = len(subset)
        ev_rate = subset['next_evidence'].mean() * 100 if n > 0 else 0
        result['by_depth'][f'level_{depth_level}'] = {
            'n': int(n),
            'evidence_pct': round(ev_rate, 1),
        }

    return result


def analyze_transcript_level(df: pd.DataFrame) -> dict:
    """Transcript-level correlation: mean L3 depth vs student evidence.

    Args:
        df: Full transcript DataFrame with 'transcript_id', 'l3_depth',
            't_move', 's_move' columns.

    Returns:
        Dictionary with correlation results and per-transcript data.
    """
    teacher_utts = df[df['t_move'].notna()]
    student_utts = df[df['s_move'].notna()]

    transcript_l3 = teacher_utts.groupby('transcript_id')['l3_depth'].agg(
        ['mean', 'count']
    ).reset_index()
    transcript_l3.columns = ['transcript_id', 'mean_l3', 'n_teacher']
    transcript_l3 = transcript_l3[transcript_l3['n_teacher'] >= 20]

    student_quality = student_utts.groupby('transcript_id').apply(
        lambda x: pd.Series({
            'pct_evidence': (x['s_move'] == 'ProvidingEvidence').mean(),
            'n_student': len(x),
        }),
        include_groups=False,
    ).reset_index()

    merged = transcript_l3.merge(student_quality, on='transcript_id')
    merged = merged[merged['n_student'] >= 10]

    if len(merged) < 3:
        return {}

    r_pearson, p_pearson = stats.pearsonr(merged['mean_l3'], merged['pct_evidence'])

    # Median split
    median_l3 = merged['mean_l3'].median()
    high_group = merged[merged['mean_l3'] >= median_l3]['pct_evidence']
    low_group = merged[merged['mean_l3'] < median_l3]['pct_evidence']

    t_stat, p_ttest = stats.ttest_ind(high_group, low_group)
    pooled_std = np.sqrt((high_group.var() + low_group.var()) / 2)
    cohens_d = (high_group.mean() - low_group.mean()) / pooled_std if pooled_std > 0 else 0

    return {
        'n': int(len(merged)),
        'pearson_r': round(r_pearson, 3),
        'pearson_p': float(p_pearson),
        'median_split': {
            'high_n': int(len(high_group)),
            'high_mean': round(high_group.mean(), 3),
            'low_n': int(len(low_group)),
            'low_mean': round(low_group.mean(), 3),
            'cohens_d': round(cohens_d, 3),
            'p_ttest': float(p_ttest),
        },
        'transcript_data': merged.to_dict('records'),
    }


def analyze_within_category(df: pd.DataFrame, category: str) -> dict:
    """Within-category mentalizing depth analysis.

    Args:
        df: Full transcript DataFrame with 't_move', 'Sentence', 's_move'.
        category: Talk move category to analyze (e.g. 'PressAccuracy').

    Returns:
        Dictionary with distribution, examples, and validation stats.
    """
    cat_utts = df[(df['t_move'] == category) & (df['word_count'] > 3)].copy()

    if len(cat_utts) == 0:
        return {}

    cat_utts['mental_depth'] = cat_utts['Sentence'].apply(classify_mentalizing_depth)

    total = len(cat_utts)
    result = {
        'category': category,
        'total': total,
        'distribution': {},
        'examples': {},
        'validation': {},
    }

    # Distribution
    depth_counts = cat_utts['mental_depth'].value_counts()
    for level in ['A', 'B', 'C']:
        count = depth_counts.get(level, 0)
        pct = count / total * 100
        result['distribution'][level] = {
            'count': int(count),
            'pct': round(pct, 1),
        }

    # Examples (up to 5 per level)
    for level in ['A', 'B', 'C']:
        subset = cat_utts[cat_utts['mental_depth'] == level]
        if len(subset) > 0:
            examples = subset.sample(min(5, len(subset)), random_state=42)
            result['examples'][level] = examples['Sentence'].tolist()
        else:
            result['examples'][level] = []

    # Sequential validation
    seq_data = []
    for idx in cat_utts.index:
        depth = cat_utts.loc[idx, 'mental_depth']
        for offset in range(1, 4):
            next_idx = idx + offset
            if next_idx < len(df) and df.loc[next_idx, 's_move'] is not None:
                s_move = df.loc[next_idx, 's_move']
                seq_data.append({
                    'mental_depth': depth,
                    'next_evidence': 1 if s_move == 'ProvidingEvidence' else 0,
                })
                break

    if seq_data:
        seq_df = pd.DataFrame(seq_data)
        result['validation']['by_depth'] = {}
        for level in ['A', 'B', 'C']:
            subset = seq_df[seq_df['mental_depth'] == level]
            n = len(subset)
            ev_rate = subset['next_evidence'].mean() * 100 if n > 0 else 0
            result['validation']['by_depth'][level] = {
                'n': int(n),
                'evidence_pct': round(ev_rate, 1),
            }

        # Chi-square if enough data
        contingency = pd.crosstab(seq_df['mental_depth'], seq_df['next_evidence'])
        if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
            chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
            n_total = contingency.sum().sum()
            cramers_v = np.sqrt(chi2 / (n_total * (min(contingency.shape) - 1)))
            result['validation']['chi2'] = round(chi2, 2)
            result['validation']['p'] = float(p_val)
            result['validation']['cramers_v'] = round(cramers_v, 3)

    return result


def get_transcript_summary(df: pd.DataFrame) -> dict:
    """Compute summary metrics for a single processed transcript."""
    total = len(df)
    teacher = df['t_move'].notna().sum()
    student = df['s_move'].notna().sum() if 's_move' in df.columns else 0
    mean_l3 = df['l3_depth'].mean() if 'l3_depth' in df.columns else None

    deep_l3_pct = 0.0
    if 'l3_depth' in df.columns and teacher > 0:
        deep_l3_pct = (df['l3_depth'] == 2).sum() / teacher * 100

    depth_counts = {}
    if 'mental_depth' in df.columns:
        vc = df['mental_depth'].value_counts()
        for level in ['A', 'B', 'C']:
            depth_counts[level] = int(vc.get(level, 0))

    return {
        'total_utterances': int(total),
        'teacher_utterances': int(teacher),
        'student_utterances': int(student),
        'mean_l3_depth': round(mean_l3, 3) if mean_l3 is not None else None,
        'deep_l3_pct': round(deep_l3_pct, 1),
        'depth_counts': depth_counts,
    }
