#!/usr/bin/env python3
"""
DToM Study 3 — Consensus (majority-vote) depth label
=====================================================

Combines the three independent classifiers (rule-based, GPT-4o, Claude) into a
single robust consensus depth per utterance and re-tests the student-evidence
gradient on it.

Consensus rule (3 ordinal raters A<B<C): the MEDIAN of the three depths.
  - equals the majority label whenever >=2 raters agree (2-1 or 3-0)
  - breaks the only possible tie (all three distinct = {A,B,C}) to B, the
    middle level — the natural ordinal compromise.

Inputs (from --output-dir):
    llm_samples.json                 (rule_based_level, next_student_move)
    llm_coding_results.json          (GPT-4o: id, level)
    llm_coding_results_claude.json   (Claude: id, level)

Outputs (to --output-dir):
    llm_consensus.csv                (per-utterance: 3 votes + consensus)
    llm_consensus_analysis.json      (distribution, agreement, evidence gradient)

Usage:
    python -m dtom.llm_consensus --output-dir output
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

from dtom.llm_classifier import _normalize_level

LABELS = ['A', 'B', 'C']
DEPTH = {'A': 0, 'B': 1, 'C': 2}
NUM2LET = {0: 'A', 1: 'B', 2: 'C'}


def level_map(path):
    with open(path, encoding='utf-8') as f:
        results = json.load(f)
    m = {}
    for r in results:
        lvl = _normalize_level(r.get('level'))
        if lvl is not None and 'id' in r:
            m[r['id']] = lvl
    return m


def evidence_rate(subset):
    if len(subset) == 0:
        return None
    return round((subset['next_student_move'] == 'ProvidingEvidence').mean() * 100, 1)


def agreement_pattern(rb, gpt, cl):
    """How many of the 3 votes match the consensus / unanimity descriptor."""
    votes = [rb, gpt, cl]
    counts = {l: votes.count(l) for l in LABELS}
    top = max(counts.values())
    if top == 3:
        return 'unanimous'
    if top == 2:
        return 'majority_2_1'
    return 'split_1_1_1'


def main():
    parser = argparse.ArgumentParser(description='DToM Study 3 consensus depth')
    parser.add_argument('--output-dir', type=str, default='output')
    args = parser.parse_args()
    od = args.output_dir

    with open(os.path.join(od, 'llm_samples.json'), encoding='utf-8') as f:
        samples = json.load(f)
    gpt = level_map(os.path.join(od, 'llm_coding_results.json'))
    claude = level_map(os.path.join(od, 'llm_coding_results_claude.json'))

    rows = []
    for s in samples:
        i = s['id']
        if i in gpt and i in claude:
            rb = s['rule_based_level']
            g, c = gpt[i], claude[i]
            cons_num = int(np.median([DEPTH[rb], DEPTH[g], DEPTH[c]]))
            rows.append({
                'id': i, 'utterance': s['utterance'],
                'rule_based': rb, 'gpt': g, 'claude': c,
                'consensus': NUM2LET[cons_num],
                'pattern': agreement_pattern(rb, g, c),
                'next_student_move': s.get('next_student_move'),
            })
    df = pd.DataFrame(rows)
    n = len(df)
    print(f"Consensus computed for {n} utterances\n")

    # Vote-pattern breakdown
    print(f"{'=' * 60}\nVOTE PATTERNS\n{'=' * 60}")
    pat = df['pattern'].value_counts().to_dict()
    for p in ['unanimous', 'majority_2_1', 'split_1_1_1']:
        k = pat.get(p, 0)
        print(f"  {p:14s}: {k} ({k / n * 100:.1f}%)")

    # Consensus distribution + how each method tracks consensus
    print(f"\n{'=' * 60}\nCONSENSUS DISTRIBUTION\n{'=' * 60}")
    cons_dist = {l: int((df['consensus'] == l).sum()) for l in LABELS}
    nonsurf = (cons_dist['B'] + cons_dist['C']) / n * 100
    print(f"  A={cons_dist['A']} B={cons_dist['B']} C={cons_dist['C']}  (non-surface {nonsurf:.1f}%)")
    print(f"\n  Agreement of each method with consensus:")
    for col, name in [('rule_based', 'Rule-based'), ('gpt', 'GPT-4o'), ('claude', 'Claude')]:
        acc = (df[col] == df['consensus']).mean() * 100
        print(f"    {name:11s}: {acc:.1f}%")

    # Evidence gradient on consensus
    print(f"\n{'=' * 60}\nSTUDENT-EVIDENCE GRADIENT ON CONSENSUS\n{'=' * 60}")
    overall = evidence_rate(df)
    print(f"  Overall: {overall}% (n={n})")
    grad = {}
    for l in LABELS:
        sub = df[df['consensus'] == l]
        r = evidence_rate(sub)
        grad[l] = {'n': len(sub), 'evidence_pct': r}
        if r is not None:
            print(f"    {l} (n={len(sub)}): {r}%")
    rates = [grad[l]['evidence_pct'] for l in LABELS if grad[l]['evidence_pct'] is not None]
    monotonic = all(x < y for x, y in zip(rates, rates[1:]))
    print(f"\n  Monotonic A<B<C: {monotonic}")

    # Optional chi-square if all cells >= 5
    chi = None
    if all(grad[l]['n'] >= 5 for l in LABELS):
        from scipy.stats import chi2_contingency
        table = []
        for l in LABELS:
            sub = df[df['consensus'] == l]
            ev = int((sub['next_student_move'] == 'ProvidingEvidence').sum())
            table.append([ev, len(sub) - ev])
        chi2, p, dof, _ = chi2_contingency(np.array(table))
        chi = {'chi2': round(float(chi2), 3), 'dof': int(dof), 'p': round(float(p), 4)}
        print(f"  Chi-square (evidence x consensus level): "
              f"chi2={chi['chi2']}, dof={chi['dof']}, p={chi['p']}")
    else:
        print(f"  Chi-square skipped (a consensus cell has n<5)")

    # Save
    df.to_csv(os.path.join(od, 'llm_consensus.csv'), index=False,
              encoding='utf-8', quoting=csv.QUOTE_ALL, lineterminator='\n')
    results = {
        'n': n,
        'vote_patterns': {p: int(pat.get(p, 0)) for p in ['unanimous', 'majority_2_1', 'split_1_1_1']},
        'consensus_distribution': cons_dist,
        'consensus_non_surface_pct': round(nonsurf, 1),
        'method_agreement_with_consensus': {
            col: round((df[col] == df['consensus']).mean() * 100, 1)
            for col in ['rule_based', 'gpt', 'claude']
        },
        'evidence_gradient': grad,
        'evidence_gradient_monotonic': bool(monotonic),
        'chi_square': chi,
    }
    with open(os.path.join(od, 'llm_consensus_analysis.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {od}/llm_consensus.csv and {od}/llm_consensus_analysis.json")


if __name__ == '__main__':
    main()
