#!/usr/bin/env python3
"""
DToM Study 3 — Validation passes (new API calls)
================================================

Two independent validation passes on the SAME 200-utterance sample that
Study 3 (GPT-4o) already coded. Reads output/llm_samples.json so the sample
is identical (no re-sampling).

Modes
-----
  reproduce : Re-code the 200 utterances with GPT-4o (temp=0, seed=42) a second
              time and measure run-to-run label stability (spec limitation #1).
              -> output/llm_coding_results_run2.json

  claude    : Independently classify the 200 utterances with Claude using the
              IDENTICAL v2 prompt (different vendor = independent replication).
              Computes three-way agreement (rule-based / GPT-4o / Claude),
              pairwise kappas, and how Claude adjudicates the 67 A->C lifts.
              -> output/llm_coding_results_claude.json
              -> output/llm_threeway_analysis.json

Usage:
    python -m dtom.llm_validation --mode claude --output-dir output
    python -m dtom.llm_validation --mode reproduce --output-dir output
    python -m dtom.llm_validation --mode both --output-dir output
"""

import argparse
import json
import os
import sys
import time

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from dtom.llm_classifier import (
    SYSTEM_PROMPT, CODING_PROMPT_TEMPLATE, MODEL as GPT_MODEL,
    BATCH_SIZE, RANDOM_SEED, _normalize_level,
)

LABELS = ['A', 'B', 'C']
DEPTH = {'A': 0, 'B': 1, 'C': 2}
CLAUDE_MODEL = "claude-sonnet-4-6"


def _format_batch(batch):
    txt = ""
    for s in batch:
        txt += f"\n--- ID: {s['id']} ---\n"
        if s['context']:
            txt += f"Context:\n{s['context']}\n"
        txt += f"Teacher utterance: {s['utterance']}\n"
    return txt


def _parse(response_text):
    response_text = response_text.strip()
    if response_text.startswith('```'):
        response_text = response_text.split('\n', 1)[1]
        if response_text.endswith('```'):
            response_text = response_text[:-3].strip()
    return json.loads(response_text)


def code_gpt(samples, api_key):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    out = []
    for start in range(0, len(samples), BATCH_SIZE):
        batch = samples[start:start + BATCH_SIZE]
        prompt = CODING_PROMPT_TEMPLATE.format(utterances=_format_batch(batch))
        resp = client.chat.completions.create(
            model=GPT_MODEL, temperature=0, seed=RANDOM_SEED, max_tokens=2000,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": prompt}],
        )
        out.extend(_parse(resp.choices[0].message.content))
        print(f"  GPT batch {start // BATCH_SIZE + 1}: ok")
        if start + BATCH_SIZE < len(samples):
            time.sleep(1)
    return out


def code_claude(samples, api_key):
    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)
    out = []
    for start in range(0, len(samples), BATCH_SIZE):
        batch = samples[start:start + BATCH_SIZE]
        prompt = CODING_PROMPT_TEMPLATE.format(utterances=_format_batch(batch))
        resp = client.messages.create(
            model=CLAUDE_MODEL, max_tokens=3000, system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        out.extend(_parse(resp.content[0].text))
        print(f"  Claude batch {start // BATCH_SIZE + 1}: ok")
        if start + BATCH_SIZE < len(samples):
            time.sleep(1)
    return out


def level_map(results):
    m = {}
    for r in results:
        lvl = _normalize_level(r.get('level'))
        if lvl is not None and 'id' in r:
            m[r['id']] = lvl
    return m


def kappa(a, b):
    from sklearn.metrics import cohen_kappa_score
    return round(float(cohen_kappa_score(a, b, labels=LABELS)), 3)


def run_reproduce(samples, output_dir):
    api_key = os.environ['OPENAI_API_KEY']
    print(f"\nReproducibility re-run: GPT-4o on {len(samples)} utterances...")
    run2 = code_gpt(samples, api_key)
    with open(os.path.join(output_dir, 'llm_coding_results_run2.json'), 'w', encoding='utf-8') as f:
        json.dump(run2, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, 'llm_coding_results.json'), encoding='utf-8') as f:
        run1 = json.load(f)
    m1, m2 = level_map(run1), level_map(run2)
    ids = sorted(set(m1) & set(m2))
    same = sum(m1[i] == m2[i] for i in ids)
    changes = [(i, m1[i], m2[i]) for i in ids if m1[i] != m2[i]]
    print(f"\n{'=' * 60}\nREPRODUCIBILITY (run1 vs run2)\n{'=' * 60}")
    print(f"  Matched ids: {len(ids)}")
    print(f"  Identical labels: {same} ({same / len(ids) * 100:.1f}%)")
    print(f"  Changed labels: {len(changes)} ({len(changes) / len(ids) * 100:.1f}%)")
    print(f"  run1-vs-run2 kappa: {kappa([m1[i] for i in ids], [m2[i] for i in ids])}")
    for i, a, b in changes[:15]:
        print(f"    id {i}: {a} -> {b}")
    return {
        'n': len(ids), 'identical': same,
        'identical_pct': round(same / len(ids) * 100, 1),
        'changed': len(changes),
        'kappa_run1_run2': kappa([m1[i] for i in ids], [m2[i] for i in ids]),
        'changes': [{'id': i, 'run1': a, 'run2': b} for i, a, b in changes],
    }


def run_claude(samples, output_dir):
    api_key = os.environ['ANTHROPIC_API_KEY']
    print(f"\nIndependent Claude classification: {len(samples)} utterances...")
    cl = code_claude(samples, api_key)
    with open(os.path.join(output_dir, 'llm_coding_results_claude.json'), 'w', encoding='utf-8') as f:
        json.dump(cl, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, 'llm_coding_results.json'), encoding='utf-8') as f:
        gpt = json.load(f)
    mg, mc = level_map(gpt), level_map(cl)

    rows = []
    for s in samples:
        i = s['id']
        if i in mg and i in mc:
            rows.append({
                'id': i, 'utterance': s['utterance'],
                'rule_based': s['rule_based_level'], 'gpt': mg[i], 'claude': mc[i],
                'next_student_move': s.get('next_student_move'),
            })
    df = pd.DataFrame(rows)
    n = len(df)

    print(f"\n{'=' * 60}\nTHREE-WAY AGREEMENT (n={n})\n{'=' * 60}")
    pk = {
        'rulebased_vs_gpt': kappa(df['rule_based'], df['gpt']),
        'rulebased_vs_claude': kappa(df['rule_based'], df['claude']),
        'gpt_vs_claude': kappa(df['gpt'], df['claude']),
    }
    for k, v in pk.items():
        print(f"  kappa {k}: {v}")

    print(f"\n  Distribution (non-surface = B+C):")
    dist = {}
    for col, name in [('rule_based', 'Rule-based'), ('gpt', 'GPT-4o'), ('claude', 'Claude')]:
        d = {l: int((df[col] == l).sum()) for l in LABELS}
        dist[col] = d
        nonsurf = (d['B'] + d['C']) / n * 100
        print(f"    {name:11s}: A={d['A']} B={d['B']} C={d['C']}  (non-surface {nonsurf:.1f}%)")

    all3 = int(((df['rule_based'] == df['gpt']) & (df['gpt'] == df['claude'])).sum())
    print(f"\n  Unanimous (all three agree): {all3} ({all3 / n * 100:.1f}%)")

    # How Claude adjudicates the 67 GPT A->C lifts
    lifts = df[(df['rule_based'] == 'A') & (df['gpt'] == 'C')]
    cl_on_lifts = {l: int((lifts['claude'] == l).sum()) for l in LABELS}
    print(f"\n  Claude's verdict on the {len(lifts)} GPT A->C lifts:")
    print(f"    sides with GPT (C):        {cl_on_lifts['C']}")
    print(f"    intermediate (B):          {cl_on_lifts['B']}")
    print(f"    sides with rule-based (A): {cl_on_lifts['A']}")

    # Evidence rate by Claude level (does Claude's depth track the gradient?)
    print(f"\n  Student-evidence rate by Claude level:")
    ev = {}
    for l in LABELS:
        sub = df[df['claude'] == l]
        if len(sub):
            r = round((sub['next_student_move'] == 'ProvidingEvidence').mean() * 100, 1)
            ev[l] = {'n': len(sub), 'evidence_pct': r}
            print(f"    {l} (n={len(sub)}): {r}%")

    results = {
        'n': n, 'claude_model': CLAUDE_MODEL,
        'pairwise_kappa': pk,
        'distribution': dist,
        'unanimous': all3,
        'claude_on_gpt_lifts': {'n_lifts': len(lifts), 'verdict': cl_on_lifts},
        'claude_evidence_by_level': ev,
    }
    with open(os.path.join(output_dir, 'llm_threeway_analysis.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {output_dir}/llm_threeway_analysis.json")
    return results


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description='DToM Study 3 validation passes')
    parser.add_argument('--mode', choices=['reproduce', 'claude', 'both'], required=True)
    parser.add_argument('--output-dir', type=str, default='output')
    args = parser.parse_args()

    with open(os.path.join(args.output_dir, 'llm_samples.json'), encoding='utf-8') as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} samples from {args.output_dir}/llm_samples.json")

    if args.mode in ('claude', 'both'):
        run_claude(samples, args.output_dir)
    if args.mode in ('reproduce', 'both'):
        run_reproduce(samples, args.output_dir)


if __name__ == '__main__':
    main()
