#!/usr/bin/env python3
"""
DToM Within-Category Analysis: LLM-Based Mentalizing Depth Classifier
=====================================================================

This script uses the Anthropic API (Claude) to classify "Press for Accuracy"
teacher utterances by mentalizing depth, providing convergent validity for
the rule-based classifier in the main pipeline.

Run LOCALLY (requires ANTHROPIC_API_KEY environment variable).

Usage:
    export ANTHROPIC_API_KEY="your-key-here"
    python dtom_llm_classifier.py --data-dir TalkMoves/data --output-dir output

The script:
  1. Samples 200 "Press for Accuracy" utterances with surrounding context
  2. Sends them to Claude in batches of 20 for mentalizing depth coding
  3. Computes inter-method agreement (Cohen's κ) with the rule-based classifier
  4. Generates a comparison figure and summary statistics

Requirements:
    pip install anthropic pandas openpyxl scipy matplotlib numpy
"""

import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import stats

# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_SIZE = 200
BATCH_SIZE = 20
RANDOM_SEED = 42
MODEL = "claude-sonnet-4-20250514"

SYSTEM_PROMPT = """You are an expert educational researcher specializing in 
mathematics classroom discourse analysis. You have deep knowledge of teacher 
questioning taxonomies (Chin 2007, Franke et al. 2009, Boaler & Brodie 2004) 
and Theory of Mind in educational contexts."""

CODING_PROMPT_TEMPLATE = """Below are teacher utterances from K-12 math classrooms. 
All are labeled "Press for Accuracy" in the standard TalkMoves coding scheme.

Your task: Classify each utterance's MENTALIZING DEPTH — how deeply the teacher 
engages with or probes the student's mental model of the mathematics.

Use this 3-level scale:

- **Level A (Surface):** Checking a factual answer, requesting a number/result, 
  confirming correctness. The teacher does NOT need to model what the student is 
  thinking to produce this question.
  Examples: "What did you get?", "Is that right?", "How much is 4 minus 30?"

- **Level B (Intermediate):** Asking the student to show or explain a procedure, 
  requesting "how" without probing conceptual understanding. The teacher needs a 
  basic model of what the student DID but not WHY they did it.
  Examples: "How did you solve that?", "Show me your work", "What method did you use?"

- **Level C (Deep):** Probing the student's conceptual understanding, asking WHY 
  something works mathematically, requesting justification or proof, asking the 
  student to evaluate alternatives, or connecting to underlying reasoning. The 
  teacher must model what the student UNDERSTANDS or MISUNDERSTANDS.
  Examples: "Why does that work?", "How do you know that's the right approach?", 
  "What would happen if we changed the denominator?", "Can you prove it?"

IMPORTANT: Consider the conversational context when provided. The same words 
("What do you think?") can be surface-level (after a factual question) or deep 
(after a conceptual discussion). Use context to judge depth.

For each utterance, respond with ONLY a JSON array. Each element:
{{"id": <number>, "level": "A" or "B" or "C", "reason": "<brief 5-10 word justification>"}}

Here are the utterances:

{utterances}

Respond with ONLY the JSON array, no other text or markdown formatting."""


# ============================================================
# RULE-BASED CLASSIFIER (for comparison)
# ============================================================

DEEP_PATTERNS = [
    r'\bwhy\b.*\b(think|work|true|right|correct|happen|say|choose|pick)\b',
    r'\bwhy\b(?!.*\bnot\b)',
    r'\bhow do you know\b', r'\bhow can you tell\b',
    r'\bwhat makes you\b', r'\bwhat tells you\b',
    r'\bexplain\b.*\b(thinking|reasoning|why|how)\b',
    r'\bwhat would happen\b', r'\bwhat if\b',
    r'\bdoes that make sense\b', r'\bdoes that work\b',
    r'\bprove\b', r'\bjustify\b', r'\bconvince\b',
    r'\bhow does that connect\b', r'\bhow does that relate\b',
    r'\bdo you agree\b.*\bwhy\b', r'\bwhy do you agree\b',
    r'\bis there another way\b', r'\bcould you do it differently\b',
    r'\bwhat.{0,20}mean\b.*\bby\b',
    r'\btell me more about\b.*\bthinking\b',
]

INTERMEDIATE_PATTERNS = [
    r'\bhow did you\b', r'\bhow do you\b',
    r'\bwhat did you do\b', r'\bwhat.{0,10}you do\b',
    r'\bshow\s+(me|us)\b', r'\bexplain\b',
    r'\btell\s+(me|us)\s+how\b', r'\btell\s+(me|us)\s+what\b',
    r'\bwhat.{0,10}(method|strategy|approach|steps|process)\b',
    r'\bwalk.{0,10}through\b', r'\bdescribe\b',
    r'\bwhat.{0,10}(next|then)\b',
    r'\bcan you\s+(show|tell|explain|describe)\b',
    r'\bhow.{0,15}(figure|solve|find|get|work)\b',
]


def classify_rule_based(text: str) -> str:
    text = str(text).lower().strip()
    for pattern in DEEP_PATTERNS:
        if re.search(pattern, text):
            return 'C'
    for pattern in INTERMEDIATE_PATTERNS:
        if re.search(pattern, text):
            return 'B'
    return 'A'


# ============================================================
# DATA LOADING
# ============================================================

def normalize_teacher_tag(tag):
    if pd.isna(tag):
        return None
    tag = str(tag).strip().lower()
    if 'accuracy' in tag:
        return 'PressAccuracy'
    return 'Other'


def normalize_student_tag(tag):
    if pd.isna(tag):
        return None
    tag = str(tag).strip().lower()
    if 'evidence' in tag or 'reasoning' in tag:
        return 'ProvidingEvidence'
    if 'claim' in tag:
        return 'MakingClaim'
    if 'relating' in tag or 'relate' in tag:
        return 'RelatingToAnother'
    if 'none' in tag or tag in ['1', '2', '5', '\\']:
        return 'None'
    return 'Other'


def load_and_sample(data_dir: str) -> tuple[pd.DataFrame, list[dict]]:
    """Load transcripts and sample Press for Accuracy utterances with context."""
    files = glob.glob(os.path.join(data_dir, 'Subset 1', '*.xlsx')) + \
            glob.glob(os.path.join(data_dir, 'Subset 2', '*.xlsx'))

    all_dfs = []
    for f in files:
        try:
            df = pd.read_excel(f, engine='openpyxl')
            df['transcript_id'] = Path(f).stem
            all_dfs.append(df)
        except:
            pass

    combined = pd.concat(all_dfs, ignore_index=True)
    combined['t_move'] = combined['Teacher Tag'].apply(normalize_teacher_tag)
    combined['s_move'] = combined['Student Tag'].apply(normalize_student_tag)
    combined['word_count'] = combined['Sentence'].apply(
        lambda x: len(str(x).split()) if pd.notna(x) else 0
    )

    press_acc = combined[
        (combined['t_move'] == 'PressAccuracy') &
        (combined['word_count'] > 3)
    ]

    print(f"Total Press for Accuracy (>3 words): {len(press_acc):,}")

    np.random.seed(RANDOM_SEED)
    sample_indices = np.random.choice(
        press_acc.index, size=min(SAMPLE_SIZE, len(press_acc)), replace=False
    )

    samples = []
    for idx in sample_indices:
        row = combined.loc[idx]

        # Get 2 preceding utterances for context
        context_rows = []
        for offset in range(1, 4):
            if idx - offset >= 0:
                prev = combined.loc[idx - offset]
                if pd.notna(prev['Sentence']) and str(prev['Sentence']).strip():
                    speaker = prev['Speaker'] if pd.notna(prev['Speaker']) else '?'
                    context_rows.append(f"[{speaker}]: {prev['Sentence']}")
        context_rows.reverse()

        # Get next student response
        next_student = None
        next_student_move = None
        for offset in range(1, 4):
            if idx + offset < len(combined):
                nxt = combined.loc[idx + offset]
                if pd.notna(nxt.get('Student Tag')):
                    next_student = str(nxt['Sentence']).strip()
                    next_student_move = normalize_student_tag(nxt['Student Tag'])
                    break

        utterance_text = str(row['Sentence']).strip()
        samples.append({
            'id': len(samples),
            'index': int(idx),
            'utterance': utterance_text,
            'context': '\n'.join(context_rows[-2:]) if context_rows else '',
            'next_student': next_student or '',
            'next_student_move': next_student_move,
            'transcript': row['transcript_id'],
            'rule_based_level': classify_rule_based(utterance_text),
        })

    print(f"Sampled {len(samples)} utterances for LLM coding")
    return combined, samples


# ============================================================
# LLM CLASSIFICATION
# ============================================================

def classify_with_llm(samples: list[dict], api_key: str) -> list[dict]:
    """Send samples to Claude API in batches for mentalizing depth coding."""
    try:
        from anthropic import Anthropic
    except ImportError:
        print("ERROR: Install anthropic package: pip install anthropic")
        return []

    client = Anthropic(api_key=api_key)
    all_results = []

    for batch_start in range(0, len(samples), BATCH_SIZE):
        batch = samples[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(samples) + BATCH_SIZE - 1) // BATCH_SIZE

        # Format utterances for the prompt
        utterance_text = ""
        for s in batch:
            utterance_text += f"\n--- ID: {s['id']} ---\n"
            if s['context']:
                utterance_text += f"Context:\n{s['context']}\n"
            utterance_text += f"Teacher utterance: {s['utterance']}\n"

        prompt = CODING_PROMPT_TEMPLATE.format(utterances=utterance_text)

        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=2000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text.strip()
            # Clean markdown fences if present
            if response_text.startswith('```'):
                response_text = response_text.split('\n', 1)[1]
                if response_text.endswith('```'):
                    response_text = response_text[:-3].strip()

            batch_results = json.loads(response_text)
            all_results.extend(batch_results)
            print(f"  Batch {batch_num}/{total_batches}: coded {len(batch_results)} utterances")

        except json.JSONDecodeError as e:
            print(f"  Batch {batch_num}/{total_batches}: JSON parse error — {e}")
            print(f"  Raw response: {response_text[:200]}...")
        except Exception as e:
            print(f"  Batch {batch_num}/{total_batches}: API error — {e}")

        # Rate limiting: pause between batches
        if batch_start + BATCH_SIZE < len(samples):
            time.sleep(1)

    return all_results


# ============================================================
# INTER-METHOD AGREEMENT ANALYSIS
# ============================================================

def compute_agreement(samples: list[dict], llm_results: list[dict]) -> dict:
    """Compute Cohen's kappa and agreement statistics between classifiers."""

    # Map LLM results by ID
    llm_by_id = {r['id']: r['level'] for r in llm_results}

    # Build comparison data
    comparison = []
    for s in samples:
        if s['id'] in llm_by_id:
            comparison.append({
                'id': s['id'],
                'utterance': s['utterance'],
                'rule_based': s['rule_based_level'],
                'llm': llm_by_id[s['id']],
                'next_student_move': s.get('next_student_move'),
            })

    if not comparison:
        print("WARNING: No matched samples for agreement analysis")
        return {}

    comp_df = pd.DataFrame(comparison)
    n = len(comp_df)

    # Overall agreement
    agreement = (comp_df['rule_based'] == comp_df['llm']).mean()

    # Cohen's kappa
    labels = ['A', 'B', 'C']
    try:
        from sklearn.metrics import cohen_kappa_score
        kappa = cohen_kappa_score(comp_df['rule_based'], comp_df['llm'], labels=labels)
    except ImportError:
        kappa = _manual_kappa(comp_df['rule_based'], comp_df['llm'], labels)

    print(f"\n{'=' * 60}")
    print(f"INTER-METHOD AGREEMENT (n={n})")
    print(f"{'=' * 60}")
    print(f"  Overall agreement: {agreement:.1%}")
    print(f"  Cohen's κ: {kappa:.3f}")

    # Confusion matrix
    print(f"\n  Confusion matrix (rows=rule-based, cols=LLM):")
    for rb_level in labels:
        counts = []
        for llm_level in labels:
            c = ((comp_df['rule_based'] == rb_level) & (comp_df['llm'] == llm_level)).sum()
            counts.append(c)
        print(f"    {rb_level}: {counts}")

    # Distribution comparison
    print(f"\n  Distribution comparison:")
    for level in labels:
        rb_pct = (comp_df['rule_based'] == level).mean() * 100
        llm_pct = (comp_df['llm'] == level).mean() * 100
        print(f"    Level {level}: Rule-based={rb_pct:.1f}%, LLM={llm_pct:.1f}%")

    # Validation: do BOTH classifiers' depth predict student evidence?
    print(f"\n  Validation — student evidence rates:")
    for method in ['rule_based', 'llm']:
        print(f"    By {method}:")
        for level in labels:
            subset = comp_df[comp_df[method] == level]
            if len(subset) > 0:
                ev_rate = (subset['next_student_move'] == 'ProvidingEvidence').mean() * 100
                print(f"      Level {level} (n={len(subset)}): {ev_rate:.1f}% evidence")

    results = {
        'n': n,
        'agreement': round(agreement, 3),
        'cohens_kappa': round(kappa, 3),
        'distribution': {
            'rule_based': {l: int((comp_df['rule_based'] == l).sum()) for l in labels},
            'llm': {l: int((comp_df['llm'] == l).sum()) for l in labels},
        }
    }

    return results


def _manual_kappa(y1, y2, labels):
    """Compute Cohen's kappa without sklearn."""
    n = len(y1)
    # Observed agreement
    po = sum(a == b for a, b in zip(y1, y2)) / n
    # Expected agreement
    pe = sum(
        (sum(a == l for a in y1) / n) * (sum(b == l for b in y2) / n)
        for l in labels
    )
    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)


# ============================================================
# MAIN
# ============================================================

def main():
    global SAMPLE_SIZE
    load_dotenv()

    parser = argparse.ArgumentParser(
        description='DToM LLM-based mentalizing depth classifier'
    )
    parser.add_argument('--data-dir', type=str, default='data/TalkMoves/data')
    parser.add_argument('--output-dir', type=str, default='output')
    parser.add_argument(
        '--api-key', type=str, default=None,
        help='Anthropic API key (or set ANTHROPIC_API_KEY env var)'
    )
    parser.add_argument(
        '--sample-size', type=int, default=SAMPLE_SIZE,
        help=f'Number of utterances to sample (default: {SAMPLE_SIZE})'
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    api_key = args.api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY or pass --api-key")
        return

    SAMPLE_SIZE = args.sample_size

    # Load and sample
    combined, samples = load_and_sample(args.data_dir)

    # Save samples for reference
    samples_path = os.path.join(args.output_dir, 'llm_samples.json')
    with open(samples_path, 'w') as f:
        json.dump(samples, f, indent=2)
    print(f"Samples saved to {samples_path}")

    # Run LLM classification
    print(f"\nRunning LLM classification ({len(samples)} utterances, "
          f"batches of {BATCH_SIZE})...")
    llm_results = classify_with_llm(samples, api_key)

    if not llm_results:
        print("ERROR: No LLM results obtained. Check API key and connection.")
        return

    # Save LLM results
    llm_path = os.path.join(args.output_dir, 'llm_coding_results.json')
    with open(llm_path, 'w') as f:
        json.dump(llm_results, f, indent=2)
    print(f"LLM results saved to {llm_path}")

    # Compute agreement
    agreement_results = compute_agreement(samples, llm_results)

    # Save all results
    all_results = {
        'n_samples': len(samples),
        'n_coded': len(llm_results),
        'model': MODEL,
        'agreement': agreement_results,
    }
    results_path = os.path.join(args.output_dir, 'llm_agreement_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to {args.output_dir}/")
    print(f"  - llm_samples.json ({len(samples)} sampled utterances)")
    print(f"  - llm_coding_results.json ({len(llm_results)} coded)")
    print(f"  - llm_agreement_results.json (agreement statistics)")


if __name__ == '__main__':
    main()
