#!/usr/bin/env python3
"""
DToM Study 3 — Follow-up Analyses (no new API calls)
====================================================

Reads the saved Study 3 outputs and computes:
  1. A clean side-by-side comparison table (fixes the embedded-newline CSV bug)
  2. Linear-weighted (ordinal) Cohen's kappa + bootstrap 95% CI on both
     unweighted and weighted kappa
  3. Decomposition of the GPT-4o evidence-gradient anomaly: split GPT-4o "C"
     into agreed-C (rule-based also C/B) vs lifted-from-A C, and compare
     student-evidence rates.

Inputs  (from --output-dir):
    llm_samples.json          (rule_based_level, next_student_move, context, ...)
    llm_coding_results.json   (id, reason, level)

Outputs (to --output-dir):
    llm_comparison.csv         (regenerated, properly quoted)
    llm_followup_analysis.json (weighted kappa, CIs, gradient decomposition)

Usage:
    python -m dtom.llm_followup --output-dir output
"""

import argparse
import csv
import json
import os
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd

LABELS = ['A', 'B', 'C']
DEPTH = {'A': 0, 'B': 1, 'C': 2}
RANDOM_SEED = 42
N_BOOTSTRAP = 10000


def _normalize_level(level_str):
    if not level_str:
        return None
    level_str = str(level_str).strip()
    if level_str and level_str[0] in ('A', 'B', 'C'):
        return level_str[0]
    return None


def load_comparison(output_dir: str) -> pd.DataFrame:
    with open(os.path.join(output_dir, 'llm_samples.json'), encoding='utf-8') as f:
        samples = json.load(f)
    with open(os.path.join(output_dir, 'llm_coding_results.json'), encoding='utf-8') as f:
        llm_results = json.load(f)

    llm_level = {}
    llm_reason = {}
    for r in llm_results:
        lvl = _normalize_level(r.get('level'))
        if lvl is not None and 'id' in r:
            llm_level[r['id']] = lvl
            llm_reason[r['id']] = r.get('reason', '')

    rows = []
    for s in samples:
        if s['id'] in llm_level:
            rows.append({
                'id': s['id'],
                'utterance': s['utterance'],
                'context': s['context'],
                'rule_based': s['rule_based_level'],
                'llm': llm_level[s['id']],
                'llm_reason': llm_reason.get(s['id'], ''),
                'next_student_move': s.get('next_student_move'),
            })
    df = pd.DataFrame(rows)
    df['rb_num'] = df['rule_based'].map(DEPTH)
    df['llm_num'] = df['llm'].map(DEPTH)
    return df


def weighted_kappa(rb, llm, weights):
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(rb, llm, labels=LABELS, weights=weights)


def bootstrap_kappa_ci(df, weights, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED):
    """Percentile bootstrap 95% CI for (weighted) Cohen's kappa."""
    from sklearn.metrics import cohen_kappa_score
    rng = np.random.default_rng(seed)
    rb = df['rule_based'].to_numpy()
    llm = df['llm'].to_numpy()
    n = len(df)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        rb_b, llm_b = rb[idx], llm[idx]
        # need at least 2 categories present in each to define kappa
        try:
            k = cohen_kappa_score(rb_b, llm_b, labels=LABELS, weights=weights)
            if not np.isnan(k):
                vals.append(k)
        except Exception:
            pass
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), float(vals.std())


def evidence_rate(subset):
    if len(subset) == 0:
        return None
    return round((subset['next_student_move'] == 'ProvidingEvidence').mean() * 100, 1)


def main():
    parser = argparse.ArgumentParser(description='DToM Study 3 follow-up analyses')
    parser.add_argument('--output-dir', type=str, default='output')
    args = parser.parse_args()

    df = load_comparison(args.output_dir)
    n = len(df)
    print(f"Loaded {n} matched utterances")

    # 1. Regenerate a clean, properly-quoted CSV
    csv_path = os.path.join(args.output_dir, 'llm_comparison.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8',
              quoting=csv.QUOTE_ALL, lineterminator='\n')
    reread = pd.read_csv(csv_path)
    print(f"Regenerated {csv_path}: {len(reread)} rows on re-read "
          f"({'OK' if len(reread) == n else 'MISMATCH'})")

    # 2. Weighted kappa + bootstrap CIs
    k_unw = weighted_kappa(df['rule_based'], df['llm'], None)
    k_lin = weighted_kappa(df['rule_based'], df['llm'], 'linear')
    k_quad = weighted_kappa(df['rule_based'], df['llm'], 'quadratic')
    lo_u, hi_u, se_u = bootstrap_kappa_ci(df, None)
    lo_l, hi_l, se_l = bootstrap_kappa_ci(df, 'linear')

    print(f"\n{'=' * 60}\nKAPPA (n={n})\n{'=' * 60}")
    print(f"  Unweighted      kappa = {k_unw:.3f}  95% CI [{lo_u:.3f}, {hi_u:.3f}]  SE={se_u:.3f}")
    print(f"  Linear-weighted kappa = {k_lin:.3f}  95% CI [{lo_l:.3f}, {hi_l:.3f}]  SE={se_l:.3f}")
    print(f"  Quadratic-wtd   kappa = {k_quad:.3f}")

    # 3. Gradient anomaly decomposition
    print(f"\n{'=' * 60}\nGRADIENT DECOMPOSITION\n{'=' * 60}")
    overall_ev = evidence_rate(df)
    print(f"  Overall evidence rate: {overall_ev}% (n={n})")

    gpt_c = df[df['llm'] == 'C']
    lifted_c = gpt_c[gpt_c['rule_based'] == 'A']          # A -> C lift
    nonlift_c = gpt_c[gpt_c['rule_based'] != 'A']          # rule-based B or C -> C
    print(f"\n  GPT-4o 'C' (n={len(gpt_c)}): evidence={evidence_rate(gpt_c)}%")
    print(f"    - lifted from rule-based A (n={len(lifted_c)}): evidence={evidence_rate(lifted_c)}%")
    print(f"    - rule-based B or C        (n={len(nonlift_c)}): evidence={evidence_rate(nonlift_c)}%")

    gpt_b = df[df['llm'] == 'B']
    lifted_b = gpt_b[gpt_b['rule_based'] == 'A']
    print(f"\n  GPT-4o 'B' (n={len(gpt_b)}): evidence={evidence_rate(gpt_b)}%")
    print(f"    - lifted from rule-based A (n={len(lifted_b)}): evidence={evidence_rate(lifted_b)}%")

    results = {
        'n': n,
        'kappa': {
            'unweighted': {'value': round(k_unw, 3), 'ci95': [round(lo_u, 3), round(hi_u, 3)], 'se': round(se_u, 3)},
            'linear_weighted': {'value': round(k_lin, 3), 'ci95': [round(lo_l, 3), round(hi_l, 3)], 'se': round(se_l, 3)},
            'quadratic_weighted': {'value': round(k_quad, 3)},
            'n_bootstrap': N_BOOTSTRAP,
        },
        'gradient_decomposition': {
            'overall_evidence_pct': overall_ev,
            'gpt_C': {
                'n': len(gpt_c), 'evidence_pct': evidence_rate(gpt_c),
                'lifted_from_A': {'n': len(lifted_c), 'evidence_pct': evidence_rate(lifted_c)},
                'rulebased_B_or_C': {'n': len(nonlift_c), 'evidence_pct': evidence_rate(nonlift_c)},
            },
            'gpt_B': {
                'n': len(gpt_b), 'evidence_pct': evidence_rate(gpt_b),
                'lifted_from_A': {'n': len(lifted_b), 'evidence_pct': evidence_rate(lifted_b)},
            },
        },
    }
    out_path = os.path.join(args.output_dir, 'llm_followup_analysis.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
