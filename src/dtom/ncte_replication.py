#!/usr/bin/env python3
"""
DToM Replication on NCTE Dataset
=================================
Cross-dataset validation of DToM L3 depth classification using the
NCTE Classroom Transcript Dataset (Demszky & Hill, 2022).

Studies:
  R1: Mentalizing depth distribution (replication of TalkMoves Study 2)
  R2: Sequential and transcript-level analysis (replication of TalkMoves Study 1)
  R3: Convergence with NCTE paired annotations (extension)

Run locally — data cannot be shared externally.

Usage:
    python -m dtom.ncte_replication --data-dir data/NCTE
    # or via main.py:
    uv run main.py --ncte --ncte-data-dir data/NCTE

Expects in data-dir:
    - ncte_single_utterances.csv   (all utterances)
    - paired_annotations.csv       (annotated subset, optional)
    - student_reasoning.csv        (annotated subset, optional)

Outputs:
    - ncte_replication_results.json (all statistics)
    - Console summary ready to paste into paper
"""

import argparse
import json
import os
import sys

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from scipy import stats

from dtom.analysis_pipeline import (
    DEEP_PATTERNS,
    INTERMEDIATE_PATTERNS,
    classify_mentalizing_depth,
)


# ============================================================
# DATA LOADING
# ============================================================

def load_single_utterances(data_dir: str) -> pd.DataFrame:
    """Load ncte_single_utterances.csv.

    Expected columns: speaker, text, cleaned_text, num_words,
    turn_idx, OBSID, NCTETID, comb_idx.
    """
    path = os.path.join(data_dir, 'ncte_single_utterances.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")

    print("Loading ncte_single_utterances.csv...")
    df = pd.read_csv(path)
    print(f"  Total utterances: {len(df):,}")
    print(f"  Unique transcripts (OBSID): {df['OBSID'].nunique()}")
    print(f"  Speakers: {df['speaker'].value_counts().to_dict()}")
    return df


def load_paired_annotations(data_dir: str) -> pd.DataFrame | None:
    """Load paired_annotations.csv if available."""
    path = os.path.join(data_dir, 'paired_annotations.csv')
    if not os.path.exists(path):
        print("  paired_annotations.csv not found — skipping Study R3")
        return None
    df = pd.read_csv(path)
    print(f"  Paired annotations loaded: {len(df):,} exchanges")
    return df


def load_student_reasoning(data_dir: str) -> pd.DataFrame | None:
    """Load student_reasoning.csv if available."""
    path = os.path.join(data_dir, 'student_reasoning.csv')
    if not os.path.exists(path):
        print("  student_reasoning.csv not found — skipping reasoning validation")
        return None
    df = pd.read_csv(path)
    print(f"  Student reasoning annotations loaded: {len(df):,} utterances")
    return df


# ============================================================
# STUDY R1: MENTALIZING DEPTH DISTRIBUTION
# ============================================================

def study_r1_distribution(su: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Study R1: Apply mentalizing depth classifier to all teacher utterances.

    Replicates TalkMoves Study 2 distribution analysis on NCTE data.

    Returns:
        (teacher_df with mental_depth column, results dict)
    """
    print("\n" + "=" * 70)
    print("STUDY R1: Mentalizing Depth Distribution (Replication)")
    print("=" * 70)

    results = {}

    # Filter: teacher utterances with >3 words (matching TalkMoves pipeline)
    teacher = su[su['speaker'] == 'teacher'].copy()
    teacher = teacher[teacher['num_words'] > 3].copy()

    # Use cleaned_text if available, fall back to text
    text_col = 'cleaned_text' if 'cleaned_text' in teacher.columns else 'text'
    teacher['mental_depth'] = teacher[text_col].apply(classify_mentalizing_depth)

    total = len(teacher)
    print(f"\n  Teacher utterances (>3 words): {total:,}")

    labels = {'A': 'Surface', 'B': 'Intermediate', 'C': 'Deep'}
    depth_counts = teacher['mental_depth'].value_counts()

    results['n_teacher'] = int(total)
    results['distribution'] = {}
    print("\n  --- Distribution ---")
    for level in ['A', 'B', 'C']:
        count = depth_counts.get(level, 0)
        pct = count / total * 100
        print(f"    Level {level} ({labels[level]}): {count:,} ({pct:.1f}%)")
        results['distribution'][level] = {
            'count': int(count), 'pct': round(pct, 1),
        }

    # Replication criterion check
    surface_pct = results['distribution']['A']['pct']
    deep_pct = results['distribution']['C']['pct']
    if surface_pct > 75 and deep_pct < 5:
        verdict = "FULL REPLICATION"
    elif surface_pct > 65 and deep_pct < 10:
        verdict = "PARTIAL REPLICATION"
    else:
        verdict = "NON-REPLICATION"
    results['replication_verdict'] = verdict
    print(f"\n  Replication criterion: {verdict}")
    print(f"    (Surface >75%: {'YES' if surface_pct > 75 else 'NO'} [{surface_pct:.1f}%], "
          f"Deep <5%: {'YES' if deep_pct < 5 else 'NO'} [{deep_pct:.1f}%])")

    return teacher, results


# ============================================================
# STUDY R2: SEQUENTIAL AND TRANSCRIPT-LEVEL ANALYSIS
# ============================================================

def study_r2_sequential(
    su: pd.DataFrame,
    teacher: pd.DataFrame,
    sr: pd.DataFrame | None,
) -> dict:
    """Study R2: Sequential and transcript-level analysis.

    Replicates TalkMoves Study 1 on NCTE data using:
      - Proxy 1: Response elaboration (word count ≥10)
      - Proxy 2: Student reasoning annotations (where available)
      - Transcript-level correlation and median-split

    Returns:
        Dictionary of all statistical results
    """
    print("\n" + "=" * 70)
    print("STUDY R2: Sequential & Transcript-Level Analysis (Replication)")
    print("=" * 70)

    results = {}
    labels = {'A': 'Surface', 'B': 'Intermediate', 'C': 'Deep'}

    # ----------------------------------------------------------
    # R2a. Sequential analysis (turn-level)
    # ----------------------------------------------------------
    print("\n  --- R2a. Sequential Analysis (Proxy 1: word count) ---")

    # Sort by transcript and turn for sequential pairing
    su_sorted = su.sort_values(['OBSID', 'turn_idx']).reset_index(drop=True)

    # Build a lookup of classified teacher utterances
    teacher_idx_set = set(teacher.index)
    teacher_depth_map = teacher['mental_depth'].to_dict()

    # For each teacher utterance, find the next student utterance in same transcript
    su_sorted_obsid = su_sorted['OBSID'].values
    su_sorted_speaker = su_sorted['speaker'].values
    su_sorted_nwords = su_sorted['num_words'].values

    # Remap teacher classification to sorted frame
    text_col = 'cleaned_text' if 'cleaned_text' in su_sorted.columns else 'text'
    teacher_mask = (
        (su_sorted['speaker'] == 'teacher') & (su_sorted['num_words'] > 3)
    )
    teacher_sorted_idx = su_sorted[teacher_mask].index
    su_sorted.loc[teacher_sorted_idx, 'mental_depth'] = (
        su_sorted.loc[teacher_sorted_idx, text_col].apply(classify_mentalizing_depth)
    )

    seq_data = []
    for idx in teacher_sorted_idx:
        depth = su_sorted.at[idx, 'mental_depth']
        obsid = su_sorted_obsid[idx]
        for offset in range(1, 5):
            nxt_idx = idx + offset
            if nxt_idx >= len(su_sorted):
                break
            if su_sorted_obsid[nxt_idx] != obsid:
                break
            if su_sorted_speaker[nxt_idx] in ('student', 'multiple students'):
                seq_data.append({
                    'mental_depth': depth,
                    'next_words': int(su_sorted_nwords[nxt_idx]),
                    'is_elaborate': 1 if su_sorted_nwords[nxt_idx] >= 10 else 0,
                    'OBSID': obsid,
                    'teacher_turn_idx': int(su_sorted.at[idx, 'turn_idx']),
                    'student_turn_idx': int(su_sorted.at[nxt_idx, 'turn_idx']),
                    'student_comb_idx': su_sorted.at[nxt_idx, 'comb_idx']
                        if 'comb_idx' in su_sorted.columns else None,
                })
                break

    seq_df = pd.DataFrame(seq_data)
    print(f"    Teacher→student pairs: {len(seq_df):,}")

    results['sequential'] = {'by_depth': {}}
    for level in ['A', 'B', 'C']:
        subset = seq_df[seq_df['mental_depth'] == level]
        n = len(subset)
        if n == 0:
            continue
        mean_words = subset['next_words'].mean()
        elaborate_pct = subset['is_elaborate'].mean() * 100
        print(f"    After {level} ({labels[level]}, n={n:,}):")
        print(f"      Mean student words: {mean_words:.1f}")
        print(f"      Elaborate (≥10 words): {elaborate_pct:.1f}%")
        results['sequential']['by_depth'][level] = {
            'n': int(n),
            'mean_words': round(mean_words, 1),
            'elaborate_pct': round(elaborate_pct, 1),
        }

    # Chi-square: depth × elaborate
    contingency = pd.crosstab(seq_df['mental_depth'], seq_df['is_elaborate'])
    chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
    n_total = contingency.sum().sum()
    cramers_v = np.sqrt(chi2 / (n_total * (min(contingency.shape) - 1)))
    print(f"\n    Chi-square (depth × elaborate): χ²={chi2:.2f}, p={p_val:.2e}, V={cramers_v:.3f}")
    results['sequential']['chi2'] = round(chi2, 2)
    results['sequential']['p'] = float(p_val)
    results['sequential']['dof'] = int(dof)
    results['sequential']['cramers_v'] = round(cramers_v, 3)
    results['sequential']['n'] = int(n_total)

    # Kruskal-Wallis on response word count
    groups = [
        seq_df[seq_df['mental_depth'] == level]['next_words']
        for level in ['A', 'B', 'C']
        if len(seq_df[seq_df['mental_depth'] == level]) > 0
    ]
    h_stat, p_kw = stats.kruskal(*groups)
    print(f"    Kruskal-Wallis (depth → word count): H={h_stat:.2f}, p={p_kw:.2e}")
    results['sequential']['kruskal_h'] = round(h_stat, 2)
    results['sequential']['kruskal_p'] = float(p_kw)

    # Pairwise comparisons (A vs C with Cramér's V)
    results['sequential']['pairwise'] = {}
    for d1, d2 in [('A', 'B'), ('A', 'C'), ('B', 'C')]:
        g1 = seq_df[seq_df['mental_depth'] == d1]['is_elaborate']
        g2 = seq_df[seq_df['mental_depth'] == d2]['is_elaborate']
        if len(g1) == 0 or len(g2) == 0:
            continue
        ct = pd.crosstab(
            pd.Series([d1] * len(g1) + [d2] * len(g2)),
            pd.Series(list(g1) + list(g2)),
        )
        chi2_pair, p_pair, _, _ = stats.chi2_contingency(ct)
        n_pair = ct.sum().sum()
        v_pair = np.sqrt(chi2_pair / (n_pair * (min(ct.shape) - 1)))
        key = f'{d1}_vs_{d2}'
        results['sequential']['pairwise'][key] = {
            'chi2': round(chi2_pair, 2),
            'p': float(p_pair),
            'cramers_v': round(v_pair, 3),
        }
        print(f"    {d1} vs {d2}: χ²={chi2_pair:.2f}, p={p_pair:.2e}, V={v_pair:.3f}")

    # ----------------------------------------------------------
    # R2b. Student reasoning validation (Proxy 2, where available)
    # ----------------------------------------------------------
    if sr is not None:
        print("\n  --- R2b. Student Reasoning Validation (Proxy 2) ---")

        sr_results = {}
        sr_n = len(sr)
        sr_reasoning_pct = (sr['student_reasoning'] == 1).mean() * 100
        print(f"    Annotated student utterances: {sr_n:,}")
        print(f"    With reasoning: {(sr['student_reasoning'] == 1).sum()} ({sr_reasoning_pct:.1f}%)")
        sr_results['n'] = sr_n
        sr_results['reasoning_pct'] = round(sr_reasoning_pct, 1)

        # Link student reasoning annotations to teacher depth via comb_idx
        if 'comb_idx' in sr.columns and 'student_comb_idx' in seq_df.columns:
            sr_set = set(sr[sr['student_reasoning'] == 1]['comb_idx'].astype(str))
            sr_all_set = set(sr['comb_idx'].astype(str))

            seq_df['student_comb_idx_str'] = seq_df['student_comb_idx'].astype(str)
            annotated = seq_df[seq_df['student_comb_idx_str'].isin(sr_all_set)].copy()

            if len(annotated) > 0:
                annotated['has_reasoning'] = annotated['student_comb_idx_str'].isin(sr_set).astype(int)
                print(f"    Linked pairs (teacher depth → annotated student): {len(annotated):,}")

                for level in ['A', 'B', 'C']:
                    subset = annotated[annotated['mental_depth'] == level]
                    n = len(subset)
                    if n == 0:
                        continue
                    reasoning_rate = subset['has_reasoning'].mean() * 100
                    print(f"      After {level}: {reasoning_rate:.1f}% reasoning (n={n})")

                # Chi-square on annotated subset
                if annotated['mental_depth'].nunique() > 1 and annotated['has_reasoning'].nunique() > 1:
                    ct_sr = pd.crosstab(annotated['mental_depth'], annotated['has_reasoning'])
                    chi2_sr, p_sr, dof_sr, _ = stats.chi2_contingency(ct_sr)
                    n_sr = ct_sr.sum().sum()
                    v_sr = np.sqrt(chi2_sr / (n_sr * (min(ct_sr.shape) - 1)))
                    print(f"      Chi-square: χ²={chi2_sr:.2f}, p={p_sr:.2e}, V={v_sr:.3f}")
                    sr_results['chi2'] = round(chi2_sr, 2)
                    sr_results['p'] = float(p_sr)
                    sr_results['cramers_v'] = round(v_sr, 3)
                    sr_results['n_linked'] = int(len(annotated))
            else:
                print("    Could not link student reasoning annotations to sequential pairs")

        results['student_reasoning_validation'] = sr_results

    # ----------------------------------------------------------
    # R2c. Transcript-level correlation
    # ----------------------------------------------------------
    print("\n  --- R2c. Transcript-Level Correlation ---")

    depth_numeric = {'A': 0, 'B': 1, 'C': 2}
    teacher_with_depth = teacher.copy()
    teacher_with_depth['depth_num'] = teacher_with_depth['mental_depth'].map(depth_numeric)

    transcript_depth = teacher_with_depth.groupby('OBSID').agg(
        mean_depth=('depth_num', 'mean'),
        n_teacher=('depth_num', 'count'),
    ).reset_index()
    transcript_depth = transcript_depth[transcript_depth['n_teacher'] >= 20]

    student_elab = su[
        su['speaker'].isin(['student', 'multiple students'])
    ].groupby('OBSID').agg(
        pct_elaborate=('num_words', lambda x: (x >= 10).mean()),
        mean_student_words=('num_words', 'mean'),
        n_student=('num_words', 'count'),
    ).reset_index()

    merged = transcript_depth.merge(student_elab, on='OBSID')
    merged = merged[merged['n_student'] >= 10]

    print(f"    N transcripts (≥20 teacher, ≥10 student): {len(merged)}")

    r_pearson, p_pearson = stats.pearsonr(merged['mean_depth'], merged['pct_elaborate'])
    r_spearman, p_spearman = stats.spearmanr(merged['mean_depth'], merged['pct_elaborate'])
    r_words, p_words = stats.pearsonr(merged['mean_depth'], merged['mean_student_words'])

    print(f"    Pearson r (depth vs % elaborate): r={r_pearson:.3f}, p={p_pearson:.4f}")
    print(f"    Spearman ρ: ρ={r_spearman:.3f}, p={p_spearman:.4f}")
    print(f"    Pearson r (depth vs mean student words): r={r_words:.3f}, p={p_words:.4f}")

    results['transcript_level'] = {
        'n': int(len(merged)),
        'pearson_r_elaborate': round(r_pearson, 3),
        'pearson_p_elaborate': float(p_pearson),
        'spearman_rho': round(r_spearman, 3),
        'spearman_p': float(p_spearman),
        'pearson_r_words': round(r_words, 3),
        'pearson_p_words': float(p_words),
    }

    # ----------------------------------------------------------
    # R2d. Median-split comparison
    # ----------------------------------------------------------
    print("\n  --- R2d. Median-Split Comparison ---")

    median_d = merged['mean_depth'].median()
    high = merged[merged['mean_depth'] >= median_d]['pct_elaborate']
    low = merged[merged['mean_depth'] < median_d]['pct_elaborate']

    t_stat, p_t = stats.ttest_ind(high, low)
    pooled_std = np.sqrt((high.var() + low.var()) / 2)
    cohens_d = (high.mean() - low.mean()) / pooled_std if pooled_std > 0 else 0

    print(f"    High depth (n={len(high)}): mean elaborate = {high.mean():.3f}")
    print(f"    Low depth  (n={len(low)}):  mean elaborate = {low.mean():.3f}")
    print(f"    t={t_stat:.3f}, p={p_t:.4f}, Cohen's d={cohens_d:.3f}")

    results['median_split'] = {
        'high_n': int(len(high)),
        'high_mean': round(float(high.mean()), 3),
        'low_n': int(len(low)),
        'low_mean': round(float(low.mean()), 3),
        't_stat': round(t_stat, 3),
        'p': round(float(p_t), 4),
        'cohens_d': round(cohens_d, 3),
    }

    # Replication criterion check
    r_val = abs(r_pearson)
    d_val = abs(cohens_d)
    if p_pearson < 0.01 and r_val > 0.25 and d_val > 0.3:
        verdict = "FULL REPLICATION"
    elif p_pearson < 0.05 and r_val > 0.15 and d_val > 0.15:
        verdict = "PARTIAL REPLICATION"
    else:
        verdict = "NON-REPLICATION"
    results['replication_verdict'] = verdict
    print(f"\n    Replication criterion: {verdict}")

    return results


# ============================================================
# STUDY R3: CONVERGENCE WITH PAIRED ANNOTATIONS
# ============================================================

def study_r3_convergence(
    su: pd.DataFrame,
    pa: pd.DataFrame,
) -> dict:
    """Study R3: Convergence between rule-based classifier and paired annotations.

    Maps paired annotation labels (high_uptake, focusing_question) to L3 depth,
    then compares with the rule-based classifier applied to the same utterances.

    Returns:
        Dictionary of convergence statistics
    """
    print("\n" + "=" * 70)
    print("STUDY R3: Convergence with Paired Annotations (Extension)")
    print("=" * 70)

    results = {'n': len(pa)}

    # ----------------------------------------------------------
    # R3a. L3 depth from paired annotations
    # ----------------------------------------------------------
    def map_annotation_l3(row):
        """Map uptake/focusing to 3-level L3 depth."""
        if row['focusing_question'] == 1:
            return 'Deep_L3'
        elif row['high_uptake'] == 1:
            return 'Mid_L3'
        else:
            return 'No_L3'

    pa = pa.copy()
    pa['annotation_l3'] = pa.apply(map_annotation_l3, axis=1)

    print(f"\n  Paired annotation L3 distribution (n={len(pa):,}):")
    results['annotation_distribution'] = {}
    for level in ['No_L3', 'Mid_L3', 'Deep_L3']:
        count = (pa['annotation_l3'] == level).sum()
        pct = count / len(pa) * 100
        print(f"    {level}: {count} ({pct:.1f}%)")
        results['annotation_distribution'][level] = {
            'count': int(count), 'pct': round(pct, 1),
        }

    # ----------------------------------------------------------
    # R3b. Apply rule-based classifier to teacher text in paired annotations
    # ----------------------------------------------------------
    pa['classifier_depth'] = pa['teacher_text'].apply(classify_mentalizing_depth)

    # Harmonize classifier depth to 3-level scale for comparison
    classifier_l3_map = {'A': 'No_L3', 'B': 'Mid_L3', 'C': 'Deep_L3'}
    pa['classifier_l3'] = pa['classifier_depth'].map(classifier_l3_map)

    print(f"\n  Rule-based classifier L3 distribution on paired annotation texts:")
    results['classifier_distribution'] = {}
    for level in ['No_L3', 'Mid_L3', 'Deep_L3']:
        count = (pa['classifier_l3'] == level).sum()
        pct = count / len(pa) * 100
        print(f"    {level}: {count} ({pct:.1f}%)")
        results['classifier_distribution'][level] = {
            'count': int(count), 'pct': round(pct, 1),
        }

    # ----------------------------------------------------------
    # R3c. Cross-tabulation and Cohen's kappa
    # ----------------------------------------------------------
    print("\n  --- Cross-tabulation ---")
    level_order = ['No_L3', 'Mid_L3', 'Deep_L3']
    ct = pd.crosstab(
        pa['annotation_l3'], pa['classifier_l3'],
        rownames=['Annotation'], colnames=['Classifier'],
    ).reindex(index=level_order, columns=level_order, fill_value=0)
    print(ct.to_string())

    results['cross_tabulation'] = ct.to_dict()

    # Cohen's kappa
    from sklearn.metrics import cohen_kappa_score
    kappa = cohen_kappa_score(pa['annotation_l3'], pa['classifier_l3'])
    print(f"\n  Cohen's κ: {kappa:.3f}")
    results['cohens_kappa'] = round(kappa, 3)

    # Interpretation
    if kappa >= 0.6:
        kappa_interp = "substantial agreement"
    elif kappa >= 0.4:
        kappa_interp = "moderate agreement"
    elif kappa >= 0.2:
        kappa_interp = "fair agreement"
    else:
        kappa_interp = "slight agreement"
    results['kappa_interpretation'] = kappa_interp
    print(f"  Interpretation: {kappa_interp}")

    # ----------------------------------------------------------
    # R3d. Key prediction: Deep (C) → focusing_question=1
    # ----------------------------------------------------------
    print("\n  --- Key prediction: Deep classifier → focusing_question ---")
    for depth in ['A', 'B', 'C']:
        subset = pa[pa['classifier_depth'] == depth]
        n = len(subset)
        if n == 0:
            continue
        fq_rate = subset['focusing_question'].mean() * 100
        hu_rate = subset['high_uptake'].mean() * 100
        print(f"    {depth}: focusing_question={fq_rate:.1f}%, high_uptake={hu_rate:.1f}% (n={n})")
        results[f'depth_{depth}_focusing'] = round(fq_rate, 1)
        results[f'depth_{depth}_uptake'] = round(hu_rate, 1)

    # Chi-square: classifier depth × focusing_question
    ct_fq = pd.crosstab(pa['classifier_depth'], pa['focusing_question'])
    if ct_fq.shape[0] > 1 and ct_fq.shape[1] > 1:
        chi2_fq, p_fq, _, _ = stats.chi2_contingency(ct_fq)
        n_fq = ct_fq.sum().sum()
        v_fq = np.sqrt(chi2_fq / (n_fq * (min(ct_fq.shape) - 1)))
        print(f"    Chi-square (depth × focusing): χ²={chi2_fq:.2f}, p={p_fq:.2e}, V={v_fq:.3f}")
        results['focusing_chi2'] = round(chi2_fq, 2)
        results['focusing_p'] = float(p_fq)
        results['focusing_cramers_v'] = round(v_fq, 3)

    return results


# ============================================================
# PAPER-READY SUMMARY
# ============================================================

def print_paper_summary(all_results: dict):
    """Print a paper-ready summary paragraph."""
    r1 = all_results.get('study_r1', {})
    r2 = all_results.get('study_r2', {})
    dataset = all_results.get('dataset', {})

    dist = r1.get('distribution', {})
    seq = r2.get('sequential', {})
    tl = r2.get('transcript_level', {})
    ms = r2.get('median_split', {})

    print(f"""
{'=' * 70}
PAPER-READY SUMMARY (Section 4.1, replication paragraph)
{'=' * 70}

To test generalizability, we applied the identical mentalizing depth
classifier to the NCTE Transcript Dataset (Demszky & Hill, 2022;
N={dataset.get('n_transcripts', '?')} transcripts,
{dataset.get('total_utterances', '?'):,} utterances).

The distribution replicated:
  {dist.get('A', {}).get('pct', '?')}% surface,
  {dist.get('B', {}).get('pct', '?')}% intermediate,
  {dist.get('C', {}).get('pct', '?')}% deep.

Sequential pattern (depth -> elaborate student response):
  chi2={seq.get('chi2', '?')}, p={seq.get('p', '?'):.2e},
  Cramer's V={seq.get('cramers_v', '?')}
  Kruskal-Wallis H={seq.get('kruskal_h', '?')}, p={seq.get('kruskal_p', '?'):.2e}

Transcript-level correlation:
  r={tl.get('pearson_r_elaborate', '?')}, p={tl.get('pearson_p_elaborate', '?'):.4f},
  N={tl.get('n', '?')}

Median split: Cohen's d={ms.get('cohens_d', '?')}

Study R1 verdict: {r1.get('replication_verdict', '?')}
Study R2 verdict: {r2.get('replication_verdict', '?')}
{'=' * 70}
""")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='DToM NCTE Replication')
    parser.add_argument(
        '--data-dir', type=str, required=True,
        help='Path to directory containing NCTE CSV files',
    )
    parser.add_argument(
        '--output-dir', type=str, default='ncte_output',
        help='Directory for output files',
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    # Load data
    su = load_single_utterances(args.data_dir)
    pa = load_paired_annotations(args.data_dir)
    sr = load_student_reasoning(args.data_dir)

    all_results['dataset'] = {
        'total_utterances': len(su),
        'n_transcripts': int(su['OBSID'].nunique()),
        'speakers': {k: int(v) for k, v in su['speaker'].value_counts().items()},
    }

    # Study R1: Distribution
    teacher, r1_results = study_r1_distribution(su)
    all_results['study_r1'] = r1_results

    # Study R2: Sequential + Transcript-level
    r2_results = study_r2_sequential(su, teacher, sr)
    all_results['study_r2'] = r2_results

    # Study R3: Convergence (if paired annotations available)
    if pa is not None:
        try:
            r3_results = study_r3_convergence(su, pa)
            all_results['study_r3'] = r3_results
        except ImportError:
            print("\n  WARNING: scikit-learn not installed — skipping Cohen's kappa")
            print("  Install with: pip install scikit-learn")
        except Exception as e:
            print(f"\n  WARNING: Study R3 failed: {e}")

    # Save results
    results_path = os.path.join(args.output_dir, 'ncte_replication_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Paper summary
    print_paper_summary(all_results)


if __name__ == '__main__':
    main()
