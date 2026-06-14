#!/usr/bin/env python3
"""
DToM Study 3 — Build consolidated convergence payload for the dtom-lens app
===========================================================================

Assembles the Study 3 + validation outputs into a single JSON the Streamlit
app can render, recomputing the reproducibility and surface-vs-non-surface
contrasts from the raw files so nothing is hard-coded.

Inputs (from --output-dir):
    llm_agreement_results.json       (GPT-4o vs rule-based; evidence rates)
    llm_followup_analysis.json       (weighted kappa + CIs)
    llm_threeway_analysis.json       (pairwise kappa, 3-way distribution, lifts)
    llm_consensus.csv                (per-utterance 3 votes + consensus)
    llm_consensus_analysis.json      (consensus distribution + gradient)
    llm_coding_results.json          (GPT run 1)
    llm_coding_results_run2.json     (GPT run 2)
    llm_coding_results_claude.json   (Claude)

Output:
    --dest (default dtom-lens/data/precomputed/llm_convergence.json)

Usage:
    python -m dtom.llm_payload --output-dir output
"""

import argparse
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


def _load(od, name):
    with open(os.path.join(od, name), encoding='utf-8') as f:
        return json.load(f)


def _level_map(od, name):
    m = {}
    for r in _load(od, name):
        lvl = _normalize_level(r.get('level'))
        if lvl is not None and 'id' in r:
            m[r['id']] = lvl
    return m


def main():
    parser = argparse.ArgumentParser(description='Build dtom-lens convergence payload')
    parser.add_argument('--output-dir', type=str, default='output')
    parser.add_argument(
        '--dest', type=str,
        default=os.path.join('dtom-lens', 'data', 'precomputed', 'llm_convergence.json'),
    )
    args = parser.parse_args()
    od = args.output_dir

    agreement = _load(od, 'llm_agreement_results.json')
    followup = _load(od, 'llm_followup_analysis.json')
    threeway = _load(od, 'llm_threeway_analysis.json')
    consensus = _load(od, 'llm_consensus_analysis.json')

    agr = agreement.get('agreement', {})

    # --- Reproducibility (recompute from the two GPT runs) ---
    m1 = _level_map(od, 'llm_coding_results.json')
    m2 = _level_map(od, 'llm_coding_results_run2.json')
    ids = sorted(set(m1) & set(m2))
    same = sum(m1[i] == m2[i] for i in ids)
    from sklearn.metrics import cohen_kappa_score
    repro_kappa = round(float(cohen_kappa_score(
        [m1[i] for i in ids], [m2[i] for i in ids], labels=LABELS)), 3)
    reproducibility = {
        'n': len(ids),
        'identical': same,
        'identical_pct': round(same / len(ids) * 100, 1),
        'changed': len(ids) - same,
        'kappa': repro_kappa,
    }

    # --- Surface vs non-surface contrast on consensus (recompute from CSV) ---
    df = pd.read_csv(os.path.join(od, 'llm_consensus.csv'))
    df['ev'] = df['next_student_move'] == 'ProvidingEvidence'
    A = df[df['consensus'] == 'A']
    NS = df[df['consensus'] != 'A']
    from scipy.stats import fisher_exact, chi2_contingency
    tab = np.array([[int(A['ev'].sum()), len(A) - int(A['ev'].sum())],
                    [int(NS['ev'].sum()), len(NS) - int(NS['ev'].sum())]])
    odds, fisher_p = fisher_exact(tab)
    chi2, chi_p, _, _ = chi2_contingency(tab)
    surface_contrast = {
        'surface_pct': round(A['ev'].mean() * 100, 1),
        'nonsurface_pct': round(NS['ev'].mean() * 100, 1),
        'odds_ratio': round(1 / odds, 2) if odds else None,
        'fisher_p': round(float(fisher_p), 4),
        'chi2_p': round(float(chi_p), 4),
    }

    payload = {
        'n': agr.get('n', 200),
        'models': {
            'gpt': agreement.get('model', 'gpt-4o'),
            'claude': threeway.get('claude_model', 'claude-sonnet-4-6'),
        },
        'prompt_version': agreement.get('prompt_version', 'v2_concept_anchored'),
        # Primary single-classifier result (GPT-4o vs rule-based)
        'primary': {
            'model': agreement.get('model', 'gpt-4o'),
            'cohens_kappa': agr.get('cohens_kappa'),
            'agreement': agr.get('agreement'),
            'distribution': agr.get('distribution', {}),
            'disagreement': agr.get('disagreement', {}),
            'evidence_rates': agr.get('evidence_rates', {}),
        },
        'weighted_kappa': followup.get('kappa', {}),
        'threeway': {
            'pairwise_kappa': threeway.get('pairwise_kappa', {}),
            'distribution': threeway.get('distribution', {}),
            'unanimous': threeway.get('unanimous'),
            'claude_on_gpt_lifts': threeway.get('claude_on_gpt_lifts', {}),
            'claude_evidence_by_level': threeway.get('claude_evidence_by_level', {}),
        },
        'consensus': {
            'distribution': consensus.get('consensus_distribution', {}),
            'non_surface_pct': consensus.get('consensus_non_surface_pct'),
            'vote_patterns': consensus.get('vote_patterns', {}),
            'method_agreement': consensus.get('method_agreement_with_consensus', {}),
            'evidence_gradient': consensus.get('evidence_gradient', {}),
            'evidence_gradient_monotonic': consensus.get('evidence_gradient_monotonic'),
            'chi_square_3level': consensus.get('chi_square'),
            'surface_vs_nonsurface': surface_contrast,
        },
        'reproducibility': reproducibility,
    }

    os.makedirs(os.path.dirname(args.dest), exist_ok=True)
    with open(args.dest, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.dest}")
    print(json.dumps(payload, indent=2)[:1500])


if __name__ == '__main__':
    main()
