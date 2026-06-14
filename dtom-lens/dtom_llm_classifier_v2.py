#!/usr/bin/env python3
"""
DToM Within-Category Analysis: LLM-Based Mentalizing Depth Classifier
=====================================================================

Uses GPT-4o to classify "Press for Accuracy" teacher utterances by
mentalizing depth, providing convergent validity for the rule-based
classifier in the main pipeline.

Usage:
    export OPENAI_API_KEY="your-key-here"
    python dtom_llm_classifier.py --data-dir TalkMoves/data --output-dir output

Requirements:
    pip install openai pandas openpyxl scipy numpy
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
from scipy import stats

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional for the standalone script
    load_dotenv = None

# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_SIZE = 200
BATCH_SIZE = 20
RANDOM_SEED = 42
MODEL = "gpt-4o"

SYSTEM_PROMPT = """You are classifying teacher utterances from K-12 mathematics classrooms. \
Each utterance has already been labeled "Press for Accuracy" by human annotators \
using a standard discourse coding scheme. Your task is to assess the MENTALIZING \
DEPTH of each utterance: how deeply the teacher is reasoning about what the student \
understands or misunderstands when producing this question.

You will receive each utterance with its conversational context (the preceding \
2 turns). Use the context to inform your judgment. The same words can reflect \
different depths depending on what came before."""

CODING_PROMPT_TEMPLATE = """Classify each teacher utterance below by MENTALIZING DEPTH: \
the degree to which the teacher must reason about the student's understanding \
to produce the utterance.

THREE LEVELS:

LEVEL A — SURFACE
The teacher checks or requests a factual answer. Producing this utterance does \
not require the teacher to consider what the student thinks, knows, or \
misunderstands. The question has a single expected answer.
Indicator: the teacher could ask this question without having heard anything \
the student said.

LEVEL B — INTERMEDIATE
The teacher asks the student to explain a process or show their work. Producing \
this utterance requires the teacher to attend to what the student DID, but not \
to WHY they did it or what they understand.
Indicator: the teacher is curious about the student's procedure but not yet \
probing the reasoning behind it.

LEVEL C — DEEP
The teacher probes the student's conceptual understanding, asks for \
justification, invites the student to evaluate alternatives, or connects the \
student's thinking to a broader mathematical idea. Producing this utterance \
requires the teacher to form a model of what the student understands or \
misunderstands.
Indicator: the teacher's question would change depending on what they believe \
the student is thinking.

CONTEXT MATTERS: An utterance like "What do you think?" is Surface after \
"What is 3 times 5?" but Deep after a student has proposed a novel solution \
strategy. Use the preceding turns to judge.

BORDERLINE CASES: When uncertain between two levels, choose the level that \
better reflects the cognitive demand on the teacher. If the teacher could \
produce the utterance without any model of the student's thinking, it is \
Surface. If the teacher needs to track procedure, it is Intermediate. If the \
teacher needs to reason about understanding, it is Deep.

EXAMPLES:

--- Example 1 ---
Context:
[Teacher]: Open your books to page forty-two.
[Student]: Is it number three?
Teacher utterance: Yes, number three. What answer did you get?
{{"id": "ex1", "reason": "Teacher requests a factual answer; no reasoning about student thinking is needed.", "level": "A — SURFACE"}}

--- Example 2 ---
Context:
[Teacher]: So we have twelve divided by four.
[Student]: Three.
Teacher utterance: Three. And how about this next one, twenty divided by five?
{{"id": "ex2", "reason": "Teacher moves to the next problem after a correct answer; checking another fact.", "level": "A — SURFACE"}}

--- Example 3 ---
Context:
[Student]: I got fourteen.
Teacher utterance: Okay, fourteen. Tell us what you did to get that.
{{"id": "ex3", "reason": "Teacher asks the student to recount their procedure, attending to what they did but not why.", "level": "B — INTERMEDIATE"}}

--- Example 4 ---
Context:
[Student]: We put them in groups of two.
Teacher utterance: You put them in groups of two. Can you show us that on the board?
{{"id": "ex4", "reason": "Teacher asks for demonstration of the procedure; tracks student action without probing reasoning.", "level": "B — INTERMEDIATE"}}

--- Example 5 ---
Context:
[Student]: I think you have to add them because they are parts of the same thing.
Teacher utterance: Interesting. What made you think they are parts of the same thing?
{{"id": "ex5", "reason": "Teacher probes the conceptual basis of the student's claim; needs a model of what the student understands about part-whole relationships.", "level": "C — DEEP"}}

--- Example 6 ---
Context:
[Student]: I got a different answer than Maria.
Teacher utterance: You did. So if someone in the class looked at both your answers, how could they figure out which one makes sense?
{{"id": "ex6", "reason": "Teacher invites the student to evaluate competing solutions; requires reasoning about what would be convincing to another person.", "level": "C — DEEP"}}

END OF EXAMPLES.

Now classify the following utterances.

For each utterance, respond with ONLY a JSON array. Each element:
{{"id": <the integer ID shown above each utterance>, "reason": "<1 sentence explaining your judgment, referencing context if relevant>", "level": "A — SURFACE" or "B — INTERMEDIATE" or "C — DEEP"}}

Note: provide the reason BEFORE the level.

UTTERANCES:

{utterances}

Respond with ONLY the JSON array. No other text, no markdown formatting."""


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
    mapping = {
        'press for accuracy': 'PressAccuracy',
        'press_for_accuracy': 'PressAccuracy',
    }
    for key, val in mapping.items():
        if key in tag:
            return val
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


def load_and_sample(data_dir: str) -> tuple:
    """Load transcripts and sample Press for Accuracy utterances with context."""
    files = glob.glob(os.path.join(data_dir, 'Subset 1', '*.xlsx')) + \
            glob.glob(os.path.join(data_dir, 'Subset 2', '*.xlsx'))

    all_dfs = []
    for f in files:
        try:
            df = pd.read_excel(f, engine='openpyxl')
            df['transcript_id'] = Path(f).stem
            all_dfs.append(df)
        except Exception:
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

def classify_with_llm(samples: list, api_key: str) -> list:
    """Send samples to GPT-4o in batches for mentalizing depth coding."""
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: Install openai package: pip install openai")
        return []

    client = OpenAI(api_key=api_key)
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
            response = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                seed=RANDOM_SEED,
                max_tokens=2000,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
            )

            response_text = response.choices[0].message.content.strip()

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

        # Rate limiting
        if batch_start + BATCH_SIZE < len(samples):
            time.sleep(1)

    return all_results


# ============================================================
# INTER-METHOD AGREEMENT ANALYSIS
# ============================================================

def compute_agreement(samples: list, llm_results: list) -> dict:
    """Compute Cohen's kappa and agreement statistics."""

    # Map LLM results by ID, normalizing level labels
    def normalize_level(level_str):
        """Extract letter from labels like 'A — SURFACE' or just 'A'."""
        if not level_str:
            return None
        level_str = str(level_str).strip()
        if level_str[0] in ('A', 'B', 'C'):
            return level_str[0]
        return None

    llm_by_id = {}
    for r in llm_results:
        level = normalize_level(r.get('level'))
        if level:
            llm_by_id[r['id']] = level

    # Build comparison data
    comparison = []
    for s in samples:
        if s['id'] in llm_by_id:
            comparison.append({
                'id': s['id'],
                'utterance': s['utterance'],
                'context': s['context'],
                'rule_based': s['rule_based_level'],
                'llm': llm_by_id[s['id']],
                'llm_reason': next(
                    (r.get('reason', '') for r in llm_results if r['id'] == s['id']),
                    ''
                ),
                'next_student_move': s.get('next_student_move'),
            })

    if not comparison:
        print("WARNING: No matched samples for agreement analysis")
        return {}

    comp_df = pd.DataFrame(comparison)
    n = len(comp_df)

    # Overall agreement
    agreement = (comp_df['rule_based'] == comp_df['llm']).mean()

    # Cohen's kappa (manual computation to avoid sklearn dependency)
    labels = ['A', 'B', 'C']
    po = agreement
    pe = sum(
        (comp_df['rule_based'] == l).mean() * (comp_df['llm'] == l).mean()
        for l in labels
    )
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0

    print(f"\n{'=' * 60}")
    print(f"INTER-METHOD AGREEMENT (n={n})")
    print(f"{'=' * 60}")
    print(f"  Overall agreement: {agreement:.1%}")
    print(f"  Cohen's κ: {kappa:.3f}")

    # Distribution comparison
    print(f"\n  Distribution:")
    for level in labels:
        rb_pct = (comp_df['rule_based'] == level).mean() * 100
        llm_pct = (comp_df['llm'] == level).mean() * 100
        print(f"    Level {level}: Rule-based={rb_pct:.1f}%, GPT-4o={llm_pct:.1f}%")

    # Confusion matrix
    print(f"\n  Confusion matrix (rows=rule-based, cols=GPT-4o):")
    print(f"         {'A':>6} {'B':>6} {'C':>6}")
    confusion = {}
    for rb_level in labels:
        counts = []
        for llm_level in labels:
            c = int(((comp_df['rule_based'] == rb_level) & (comp_df['llm'] == llm_level)).sum())
            counts.append(c)
        confusion[rb_level] = {l: c for l, c in zip(labels, counts)}
        print(f"    {rb_level}:  {counts[0]:>6} {counts[1]:>6} {counts[2]:>6}")

    # Disagreement direction
    deeper = ((comp_df['rule_based'] < comp_df['llm'].map({'A': 0, 'B': 1, 'C': 2}.get)) |
              ((comp_df['rule_based'] == 'A') & (comp_df['llm'].isin(['B', 'C']))) |
              ((comp_df['rule_based'] == 'B') & (comp_df['llm'] == 'C')))

    # Simpler approach
    depth_map = {'A': 0, 'B': 1, 'C': 2}
    comp_df['rb_num'] = comp_df['rule_based'].map(depth_map)
    comp_df['llm_num'] = comp_df['llm'].map(depth_map)

    n_agree = (comp_df['rb_num'] == comp_df['llm_num']).sum()
    n_llm_deeper = (comp_df['llm_num'] > comp_df['rb_num']).sum()
    n_llm_shallower = (comp_df['llm_num'] < comp_df['rb_num']).sum()

    print(f"\n  Disagreement direction:")
    print(f"    Agree: {n_agree} ({n_agree/n*100:.1f}%)")
    print(f"    GPT-4o codes deeper: {n_llm_deeper} ({n_llm_deeper/n*100:.1f}%)")
    print(f"    GPT-4o codes shallower: {n_llm_shallower} ({n_llm_shallower/n*100:.1f}%)")

    # Validation: do BOTH classifiers' depth predict student evidence?
    print(f"\n  Student evidence rates by depth:")
    for method, label in [('rule_based', 'Rule-based'), ('llm', 'GPT-4o')]:
        print(f"    {label}:")
        for level in labels:
            subset = comp_df[comp_df[method] == level]
            if len(subset) > 0:
                ev_rate = (subset['next_student_move'] == 'ProvidingEvidence').mean() * 100
                print(f"      Level {level} (n={len(subset)}): {ev_rate:.1f}% evidence")

    # Compile results
    results = {
        'n': n,
        'model': MODEL,
        'temperature': 0,
        'seed': RANDOM_SEED,
        'agreement': round(agreement, 3),
        'cohens_kappa': round(kappa, 3),
        'distribution': {
            'rule_based': {l: int((comp_df['rule_based'] == l).sum()) for l in labels},
            'llm': {l: int((comp_df['llm'] == l).sum()) for l in labels},
        },
        'confusion_matrix': confusion,
        'disagreement': {
            'agree': int(n_agree),
            'llm_deeper': int(n_llm_deeper),
            'llm_shallower': int(n_llm_shallower),
        },
        'evidence_rates': {},
    }

    for method in ['rule_based', 'llm']:
        results['evidence_rates'][method] = {}
        for level in labels:
            subset = comp_df[comp_df[method] == level]
            if len(subset) > 0:
                results['evidence_rates'][method][level] = {
                    'n': int(len(subset)),
                    'evidence_pct': round(
                        (subset['next_student_move'] == 'ProvidingEvidence').mean() * 100, 1
                    ),
                }

    return results, comp_df


# ============================================================
# MAIN
# ============================================================

def main():
    if load_dotenv is not None:
        load_dotenv()

    parser = argparse.ArgumentParser(
        description='DToM LLM-based mentalizing depth classifier (GPT-4o)'
    )
    parser.add_argument('--data-dir', type=str, default='data/TalkMoves/data')
    parser.add_argument('--output-dir', type=str, default='output')
    parser.add_argument(
        '--api-key', type=str, default=None,
        help='OpenAI API key (or set OPENAI_API_KEY env var)'
    )
    parser.add_argument(
        '--sample-size', type=int, default=SAMPLE_SIZE,
        help=f'Number of utterances to sample (default: {SAMPLE_SIZE})'
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY or pass --api-key")
        return

    global SAMPLE_SIZE
    SAMPLE_SIZE = args.sample_size

    # Load and sample
    combined, samples = load_and_sample(args.data_dir)

    # Save samples for reference
    samples_path = os.path.join(args.output_dir, 'llm_samples.json')
    with open(samples_path, 'w') as f:
        json.dump(samples, f, indent=2)
    print(f"Samples saved to {samples_path}")

    # Run LLM classification
    print(f"\nRunning GPT-4o classification ({len(samples)} utterances, "
          f"batches of {BATCH_SIZE}, temperature=0, seed={RANDOM_SEED})...")
    llm_results = classify_with_llm(samples, api_key)

    if not llm_results:
        print("ERROR: No LLM results obtained. Check API key and connection.")
        return

    # Save raw LLM results (with reasons)
    llm_path = os.path.join(args.output_dir, 'llm_coding_results.json')
    with open(llm_path, 'w') as f:
        json.dump(llm_results, f, indent=2)
    print(f"LLM results saved to {llm_path}")

    # Compute agreement
    agreement_results, comp_df = compute_agreement(samples, llm_results)

    # Save comparison dataframe (for qualitative analysis of disagreements)
    comp_path = os.path.join(args.output_dir, 'llm_comparison.csv')
    comp_df.to_csv(comp_path, index=False)
    print(f"Comparison data saved to {comp_path}")

    # Save all results
    all_results = {
        'n_samples': len(samples),
        'n_coded': len(llm_results),
        'model': MODEL,
        'prompt_version': 'v2_concept_anchored',
        'agreement': agreement_results,
    }
    results_path = os.path.join(args.output_dir, 'llm_agreement_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"All results saved to {args.output_dir}/")
    print(f"  - llm_samples.json ({len(samples)} sampled utterances)")
    print(f"  - llm_coding_results.json ({len(llm_results)} coded, with reasons)")
    print(f"  - llm_comparison.csv (side-by-side for qualitative analysis)")
    print(f"  - llm_agreement_results.json (agreement statistics)")
    print(f"{'=' * 60}")

    # Print paper-ready summary
    ar = agreement_results
    print(f"""
╔══════════════════════════════════════════════════════════╗
║  PAPER-READY SUMMARY                                    ║
╠══════════════════════════════════════════════════════════╣

  Model: {MODEL}, temperature=0, seed={RANDOM_SEED}
  Prompt: v2 (concept-anchored, reason-before-label)
  Sample: {ar['n']} "Press for Accuracy" utterances

  Agreement: {ar['agreement']:.1%}
  Cohen's κ: {ar['cohens_kappa']:.3f}

  Distribution:
    Rule-based: A={ar['distribution']['rule_based']['A']}, B={ar['distribution']['rule_based']['B']}, C={ar['distribution']['rule_based']['C']}
    GPT-4o:     A={ar['distribution']['llm']['A']}, B={ar['distribution']['llm']['B']}, C={ar['distribution']['llm']['C']}

  Disagreement direction:
    Agree: {ar['disagreement']['agree']} ({ar['disagreement']['agree']/ar['n']*100:.1f}%)
    GPT-4o deeper: {ar['disagreement']['llm_deeper']} ({ar['disagreement']['llm_deeper']/ar['n']*100:.1f}%)
    GPT-4o shallower: {ar['disagreement']['llm_shallower']} ({ar['disagreement']['llm_shallower']/ar['n']*100:.1f}%)

╚══════════════════════════════════════════════════════════╝
""")


if __name__ == '__main__':
    main()
