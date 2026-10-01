#!/usr/bin/env python3
"""
DToM — Additional Experiments (WI-IAT 2026 revision)
====================================================

Four supplementary analyses specified in
`specs/additional_experiments_spec.md`:

  Option 1  Confusion matrix: rule-based vs 2-of-3 LLM consensus (3 levels)
  Option 2  Qualitative analysis of GPT-4o's A->C lift reasons
            (corroborated vs reverted by Claude)
  Option 3  Within-category depth across all Level-1 categories
            (Press for Accuracy, Revoicing, Restating)
  Option 4  L3 depth distribution by grade level (TalkMoves Subset-1)
  Option 5  Pattern-ablation sensitivity of Study 2
            (`specs/pattern_ablation_spec.md`)
  Option 5b Two-tier pattern ablation (`specs/pattern_ablation_spec_v2.md`)
  Option 6  Matched transcript-level effect size, 2 x 2 IV x DV on TalkMoves
            (`specs/matched_effect_size_spec.md`)

Options 1 & 2 reuse the Study-3 LLM output files in `output/`.
Options 3-6 re-run the rule-based pipeline on the TalkMoves corpus
(option 6 also reads the NCTE utterances for its reference values).

Usage:
  uv run python -m dtom.additional_experiments --option all
  uv run python -m dtom.additional_experiments --option 1
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from scipy import stats

from dtom.analysis_pipeline import (
    DEEP_PATTERNS,
    INTERMEDIATE_PATTERNS,
    L3_DEPTH_MAP,
    MENTAL_DEPTH_LABELS,
    classify_mentalizing_depth,
    load_transcripts,
)

LEVELS = ["A", "B", "C"]


# ============================================================
# OPTION 1 — Confusion matrix: rule-based x consensus
# ============================================================

def option1_confusion_matrix(output_dir: str) -> dict:
    """3x3 confusion matrix between the rule-based classifier and the
    two-of-three LLM consensus on the 200-utterance subsample."""
    print("\n" + "=" * 70)
    print("OPTION 1: Confusion Matrix (Rule-based x Consensus)")
    print("=" * 70)

    df = pd.read_csv(os.path.join(output_dir, "llm_consensus.csv"))
    df["rule_based"] = pd.Categorical(df["rule_based"], categories=LEVELS)
    df["consensus"] = pd.Categorical(df["consensus"], categories=LEVELS)

    matrix = pd.crosstab(
        df["rule_based"], df["consensus"], margins=True, margins_name="Total",
        dropna=False,
    )
    print("\nRows = rule-based, Cols = consensus\n")
    print(matrix.to_string())

    # Where do the rule-based Surface (A) utterances go under consensus?
    rb_A = df[df["rule_based"] == "A"]
    n_A = len(rb_A)
    a_to_a = int((rb_A["consensus"] == "A").sum())
    a_to_b = int((rb_A["consensus"] == "B").sum())
    a_to_c = int((rb_A["consensus"] == "C").sum())

    print(f"\nOf {n_A} rule-based Surface (A) utterances, consensus assigns:")
    print(f"  A (stays Surface):       {a_to_a}  ({a_to_a / n_A * 100:.1f}%)")
    print(f"  B (-> Intermediate):     {a_to_b}  ({a_to_b / n_A * 100:.1f}%)")
    print(f"  C (-> Deep):             {a_to_c}  ({a_to_c / n_A * 100:.1f}%)")

    # Raw agreement on the diagonal (excl. margins)
    diag = sum(
        int(((df["rule_based"] == lv) & (df["consensus"] == lv)).sum())
        for lv in LEVELS
    )
    raw_agreement = diag / len(df) * 100
    print(f"\nDiagonal (exact 3-level agreement): {diag}/{len(df)} = {raw_agreement:.1f}%")

    results = {
        "n": int(len(df)),
        "matrix": {
            rb: {cs: int(matrix.loc[rb, cs]) for cs in LEVELS}
            for rb in LEVELS
        },
        "row_totals": {lv: int((df["rule_based"] == lv).sum()) for lv in LEVELS},
        "col_totals": {lv: int((df["consensus"] == lv).sum()) for lv in LEVELS},
        "rule_based_A_reclassification": {
            "n_rule_A": n_A,
            "stays_A": {"n": a_to_a, "pct": round(a_to_a / n_A * 100, 1)},
            "to_B_intermediate": {"n": a_to_b, "pct": round(a_to_b / n_A * 100, 1)},
            "to_C_deep": {"n": a_to_c, "pct": round(a_to_c / n_A * 100, 1)},
        },
        "exact_agreement_pct": round(raw_agreement, 1),
    }

    out = os.path.join(output_dir, "additional_option1_confusion.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")
    return results


# ============================================================
# OPTION 2 — Qualitative analysis of A->C lift reasons
# ============================================================

REASON_CATEGORIES = OrderedDict([
    ("contextual_probing", "Contextual probing"),
    ("implicit_reasoning_demand", "Implicit reasoning demand"),
    ("keyword_over_extension", "Keyword over-extension"),
    ("pragmatic_reinterpretation", "Pragmatic reinterpretation"),
])


def option2_extract_lifts(output_dir: str) -> pd.DataFrame:
    """Extract the 67 A->C lift utterances (rule-based=A, GPT-4o=C) with
    GPT-4o and Claude reasons + labels, ready for qualitative coding."""
    print("\n" + "=" * 70)
    print("OPTION 2 (extract): GPT-4o A->C lift utterances")
    print("=" * 70)

    consensus = pd.read_csv(os.path.join(output_dir, "llm_consensus.csv"))
    gpt = pd.DataFrame(json.load(
        open(os.path.join(output_dir, "llm_coding_results.json"), encoding="utf-8")))
    claude = pd.DataFrame(json.load(
        open(os.path.join(output_dir, "llm_coding_results_claude.json"), encoding="utf-8")))

    gpt = gpt.rename(columns={"reason": "gpt_reason", "level": "gpt_level"})
    claude = claude.rename(columns={"reason": "claude_reason", "level": "claude_level"})

    df = consensus.merge(gpt[["id", "gpt_reason", "gpt_level"]], on="id")
    df = df.merge(claude[["id", "claude_reason", "claude_level"]], on="id")

    lifts = df[(df["rule_based"] == "A") & (df["gpt"] == "C")].copy()
    lifts["status"] = np.where(
        lifts["claude"] == "A", "reverted", "corroborated")

    print(f"\nA->C lifts: {len(lifts)}")
    print(f"  corroborated (Claude non-A): {int((lifts['status']=='corroborated').sum())}")
    print(f"  reverted    (Claude = A):    {int((lifts['status']=='reverted').sum())}")

    cols = ["id", "utterance", "rule_based", "gpt", "claude", "consensus",
            "status", "next_student_move", "gpt_reason", "claude_reason"]
    lifts_out = lifts[cols].sort_values("id")
    out = os.path.join(output_dir, "additional_option2_lifts.csv")
    lifts_out.to_csv(out, index=False)
    print(f"Saved: {out}")
    return lifts_out


def option2_summarize(output_dir: str, coding_path: str) -> dict:
    """Summarize the qualitative coding of the 67 lift reasons.

    `coding_path` is a JSON file mapping utterance id -> reason-category key
    (one of REASON_CATEGORIES), produced by manual/LLM categorisation of the
    GPT-4o reasons extracted by `option2_extract_lifts`.
    """
    print("\n" + "=" * 70)
    print("OPTION 2 (summarize): Reason categories, corroborated vs reverted")
    print("=" * 70)

    lifts = pd.read_csv(os.path.join(output_dir, "additional_option2_lifts.csv"))
    coding = json.load(open(coding_path, encoding="utf-8"))
    coding = {int(k): v for k, v in coding.items() if not k.startswith("_")}
    lifts["category"] = lifts["id"].map(coding)

    missing = lifts[lifts["category"].isna()]
    if len(missing):
        raise ValueError(f"Uncoded ids: {missing['id'].tolist()}")
    bad = set(lifts["category"]) - set(REASON_CATEGORIES)
    if bad:
        raise ValueError(f"Unknown categories: {bad}")

    n_corr = int((lifts["status"] == "corroborated").sum())
    n_rev = int((lifts["status"] == "reverted").sum())

    table = {}
    print(f"\n{'Reason type':<30}{'Corroborated':<18}{'Reverted':<12}")
    print(f"{'':<30}{'(n=' + str(n_corr) + ')':<18}{'(n=' + str(n_rev) + ')':<12}")
    for key, label in REASON_CATEGORIES.items():
        c = int(((lifts["status"] == "corroborated") & (lifts["category"] == key)).sum())
        r = int(((lifts["status"] == "reverted") & (lifts["category"] == key)).sum())
        c_pct = c / n_corr * 100 if n_corr else 0
        r_pct = r / n_rev * 100 if n_rev else 0
        print(f"{label:<30}{f'{c} ({c_pct:.0f}%)':<18}{f'{r} ({r_pct:.0f}%)':<12}")
        table[key] = {
            "label": label,
            "corroborated": {"n": c, "pct": round(c_pct, 1)},
            "reverted": {"n": r, "pct": round(r_pct, 1)},
        }

    results = {
        "n_lifts": int(len(lifts)),
        "n_corroborated": n_corr,
        "n_reverted": n_rev,
        "categories": table,
    }
    out = os.path.join(output_dir, "additional_option2_reasons.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")
    return results


# ============================================================
# OPTION 3 — Within-category depth across Level-1 categories
# ============================================================

def _evidence_gradient(combined, subset):
    """Link each utterance to the next student move (within 3 turns) and
    compute the ProvidingEvidence rate per within-category depth level."""
    seq = []
    for idx in subset.index:
        depth = subset.loc[idx, "mental_depth"]
        for offset in range(1, 4):
            j = idx + offset
            if j < len(combined) and combined.loc[j, "s_move"] is not None:
                seq.append({
                    "mental_depth": depth,
                    "next_evidence": 1 if combined.loc[j, "s_move"] == "ProvidingEvidence" else 0,
                })
                break
    return pd.DataFrame(seq)


def option3_within_category(combined: pd.DataFrame, output_dir: str) -> dict:
    """Apply the 35-pattern depth classifier to each Level-1 category and
    test whether the within-category depth gradient generalises."""
    print("\n" + "=" * 70)
    print("OPTION 3: Within-Category Depth Across Level-1 Categories")
    print("=" * 70)

    categories = ["PressAccuracy", "Revoicing", "Restating"]
    results = {}
    for cat in categories:
        subset = combined[combined["t_move"] == cat].copy()
        subset = subset[subset["word_count"] > 3].copy()
        subset["mental_depth"] = subset["Sentence"].apply(classify_mentalizing_depth)
        total = len(subset)

        dist = {lv: int((subset["mental_depth"] == lv).sum()) for lv in LEVELS}
        dist_pct = {lv: round(dist[lv] / total * 100, 1) for lv in LEVELS}

        seq_df = _evidence_gradient(combined, subset)
        grad = {}
        for lv in LEVELS:
            s = seq_df[seq_df["mental_depth"] == lv]
            grad[lv] = {
                "n": int(len(s)),
                "evidence_pct": round(s["next_evidence"].mean() * 100, 1) if len(s) else None,
            }

        # Chi-square over depth x next_evidence
        ct = pd.crosstab(seq_df["mental_depth"], seq_df["next_evidence"])
        chi2 = p = v = None
        if ct.shape[0] >= 2 and ct.shape[1] >= 2:
            chi2, p, _, _ = stats.chi2_contingency(ct)
            n_ct = ct.sum().sum()
            v = float(np.sqrt(chi2 / (n_ct * (min(ct.shape) - 1))))

        monotonic = (
            grad["A"]["evidence_pct"] is not None
            and grad["B"]["evidence_pct"] is not None
            and grad["C"]["evidence_pct"] is not None
            and grad["A"]["evidence_pct"] <= grad["B"]["evidence_pct"] <= grad["C"]["evidence_pct"]
        )

        grad_str = " -> ".join(
            f"{grad[lv]['evidence_pct']}%" if grad[lv]["evidence_pct"] is not None else "n/a"
            for lv in LEVELS
        )
        print(f"\n{cat}: N={total:,}  "
              f"A={dist_pct['A']}% B={dist_pct['B']}% C={dist_pct['C']}%  "
              f"gradient {grad_str}  "
              f"chi2={chi2:.1f} p={p:.2e}" if chi2 is not None else
              f"\n{cat}: N={total:,}  A={dist_pct['A']}% B={dist_pct['B']}% C={dist_pct['C']}%")

        results[cat] = {
            "n": total,
            "distribution_pct": dist_pct,
            "distribution_n": dist,
            "evidence_gradient": grad,
            "evidence_gradient_monotonic": bool(monotonic),
            "chi_square": {
                "chi2": round(chi2, 2) if chi2 is not None else None,
                "p": float(p) if p is not None else None,
                "cramers_v": round(v, 3) if v is not None else None,
            },
        }

    out = os.path.join(output_dir, "additional_option3_within_category.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")
    return results


# ============================================================
# OPTION 4 — L3 depth distribution by grade level
# ============================================================

GRADE_BANDS = OrderedDict([
    ("3-4", {"1st", "2nd", "3rd", "4th"}),
    ("5-6", {"5th", "6th"}),
    ("7-8", {"7th", "8th", "MS"}),
    ("9-12", {"9th", "10th", "11th", "12th"}),
])


def _grade_band(raw: str):
    raw = str(raw).strip()
    for band, members in GRADE_BANDS.items():
        if raw in members:
            return band
    return None


def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def option4_grade_level(combined: pd.DataFrame, data_dir: str, output_dir: str) -> dict:
    """L3 depth distribution by grade band, using the TalkMoves Datasheet
    grade metadata (available for the Subset-1 transcripts)."""
    print("\n" + "=" * 70)
    print("OPTION 4: L3 Depth Distribution by Grade Level")
    print("=" * 70)

    ds_path = os.path.join(data_dir, "Datasheet for Public Release.xlsx")
    ds = pd.read_excel(ds_path)
    ds.columns = [c.strip() for c in ds.columns]
    grade_col = [c for c in ds.columns if c.startswith("Grade")][0]
    ds = ds[["Name of File", grade_col]].dropna(subset=["Name of File"])
    ds["key"] = ds["Name of File"].apply(_norm_name)
    ds["band"] = ds[grade_col].apply(_grade_band)
    grade_lookup = dict(zip(ds["key"], ds["band"]))

    teacher = combined[combined["t_move"].notna()].copy()
    teacher["key"] = teacher["transcript_id"].apply(_norm_name)
    teacher["band"] = teacher["key"].map(grade_lookup)

    matched_transcripts = teacher.loc[teacher["band"].notna(), "transcript_id"].nunique()
    total_transcripts = teacher["transcript_id"].nunique()
    print(f"\nTranscripts with grade metadata: {matched_transcripts}/{total_transcripts}")

    teacher = teacher[teacher["band"].notna()].copy()

    # Link next student evidence for deep (L3=2) utterances
    seq = []
    for idx in teacher.index:
        for offset in range(1, 4):
            j = idx + offset
            if j < len(combined) and combined.loc[j, "s_move"] is not None:
                seq.append((idx, 1 if combined.loc[j, "s_move"] == "ProvidingEvidence" else 0))
                break
    ev_map = dict(seq)
    teacher["next_evidence"] = teacher.index.map(lambda i: ev_map.get(i, np.nan))

    results = {
        "n_transcripts_with_grade": int(matched_transcripts),
        "n_transcripts_total": int(total_transcripts),
        "coverage_note": "Grade metadata available only for Subset-1 transcripts "
                         "(Datasheet for Public Release.xlsx); Subset-2 'talkback' "
                         "transcripts have no grade label.",
        "bands": {},
    }

    print(f"\n{'Grade':<8}{'Transcripts':<13}{'N utt':<9}{'%Deep L3':<11}"
          f"{'Mean depth':<13}{'Evid(Deep)':<11}")
    for band in GRADE_BANDS:
        sub = teacher[teacher["band"] == band]
        if len(sub) == 0:
            continue
        n_tr = sub["transcript_id"].nunique()
        n_utt = len(sub)
        pct_deep = (sub["l3_depth"] == 2).mean() * 100
        # mean depth per transcript, then mean across transcripts
        mean_depth = sub.groupby("transcript_id")["l3_depth"].mean().mean()
        deep_ev = sub.loc[sub["l3_depth"] == 2, "next_evidence"]
        deep_ev_rate = deep_ev.mean() * 100 if deep_ev.notna().any() else None

        print(f"{band:<8}{n_tr:<13}{n_utt:<9}{pct_deep:<11.1f}{mean_depth:<13.3f}"
              f"{(f'{deep_ev_rate:.1f}%' if deep_ev_rate is not None else 'n/a'):<11}")

        results["bands"][band] = {
            "n_transcripts": int(n_tr),
            "n_utterances": int(n_utt),
            "pct_deep_l3": round(pct_deep, 1),
            "mean_depth": round(float(mean_depth), 3),
            "evidence_rate_deep_pct": round(deep_ev_rate, 1) if deep_ev_rate is not None else None,
        }

    # Trend test: mean transcript depth across ordinal grade bands
    band_order = {b: i for i, b in enumerate(GRADE_BANDS)}
    tr_depth = teacher.groupby("transcript_id").agg(
        band=("band", "first"), mean_depth=("l3_depth", "mean")).reset_index()
    tr_depth["band_rank"] = tr_depth["band"].map(band_order)
    rho, p_rho = stats.spearmanr(tr_depth["band_rank"], tr_depth["mean_depth"])
    print(f"\nSpearman (grade band vs transcript mean L3 depth): "
          f"rho={rho:.3f}, p={p_rho:.4f}, n={len(tr_depth)} transcripts")
    results["grade_depth_trend"] = {
        "spearman_rho": round(float(rho), 3),
        "p": round(float(p_rho), 4),
        "n_transcripts": int(len(tr_depth)),
    }

    out = os.path.join(output_dir, "additional_option4_grade_level.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")
    return results


# ============================================================
# OPTION 5 — Pattern-ablation sensitivity of Study 2
# ============================================================

ABLATION_SEED = 42
RANDOM_SIZES = (5, 10, 15)
RANDOM_DRAWS = 200
DROP_TOPK = (1, 2, 3, 5)


def _ablation_stat_block(labels: np.ndarray, ev: np.ndarray, linked: np.ndarray,
                         n_deep: int) -> dict:
    """Study-2 statistics for one labelling of the Press-for-Accuracy set.

    `labels` holds 'A'/'B'/'C' per utterance, `ev` the next-student-turn
    evidence flag and `linked` whether a student turn was found within 3 turns.
    """
    total = len(labels)
    dist_n = {lv: int((labels == lv).sum()) for lv in LEVELS}
    dist_pct = {lv: round(dist_n[lv] / total * 100, 1) for lv in LEVELS}

    lab_l, ev_l = labels[linked], ev[linked]
    evidence_n, evidence_pct, ct = {}, {}, []
    for lv in LEVELS:
        m = lab_l == lv
        n, k = int(m.sum()), int(ev_l[m].sum())
        evidence_n[lv] = n
        evidence_pct[lv] = round(k / n * 100, 1) if n else None
        if n:
            ct.append([n - k, k])

    monotonic = all(evidence_pct[lv] is not None for lv in LEVELS) and (
        evidence_pct["A"] < evidence_pct["B"] < evidence_pct["C"])

    chi2 = p = v = None
    ct = np.array(ct)
    if ct.shape[0] >= 2 and (ct.sum(axis=0) > 0).all():
        chi2, p, _, _ = stats.chi2_contingency(ct)
        v = float(np.sqrt(chi2 / (ct.sum() * (min(ct.shape) - 1))))

    # Surface vs non-surface: odds of evidence after non-surface / after surface
    surf = lab_l == "A"
    a, b = int(ev_l[~surf].sum()), int((~surf).sum() - ev_l[~surf].sum())
    c, d = int(ev_l[surf].sum()), int(surf.sum() - ev_l[surf].sum())
    odds = ci = binary_p = None
    if min(a, b, c, d) > 0:
        odds = a * d / (b * c)
        se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        ci = [round(float(np.exp(np.log(odds) - 1.96 * se)), 3),
              round(float(np.exp(np.log(odds) + 1.96 * se)), 3)]
        _, binary_p = stats.fisher_exact([[a, b], [c, d]])

    return {
        "n_deep_patterns": n_deep,
        "dist_pct": dist_pct,
        "dist_n": dist_n,
        "evidence_pct": evidence_pct,
        "evidence_n": evidence_n,
        "monotonic": bool(monotonic),
        "chi2": round(float(chi2), 2) if chi2 is not None else None,
        "p": float(p) if p is not None else None,
        "cramers_v": round(v, 3) if v is not None else None,
        "binary_or": round(float(odds), 3) if odds is not None else None,
        "binary_or_ci95": ci,
        "binary_p": float(binary_p) if binary_p is not None else None,
        "n_nonsurface": int((labels != "A").sum()),
        "nonsurface_composition": {
            "deep": round(dist_n["C"] / max(total - dist_n["A"], 1) * 100, 1),
            "intermediate": round(dist_n["B"] / max(total - dist_n["A"], 1) * 100, 1),
        },
        "degenerate": bool(evidence_n["B"] < 30 or evidence_n["C"] < 30),
    }


def _summarise_runs(runs: list, baseline_or: float) -> dict:
    ors = np.array([r["binary_or"] for r in runs if r["binary_or"] is not None])
    return {
        "n_runs": len(runs),
        "pct_monotonic": round(float(np.mean([r["monotonic"] for r in runs]) * 100), 1),
        "pct_binary_p_lt_001": round(float(np.mean(
            [r["binary_p"] is not None and r["binary_p"] < 0.001 for r in runs]) * 100), 1),
        "binary_or": {
            "mean": round(float(ors.mean()), 3),
            "median": round(float(np.median(ors)), 3),
            "p2.5": round(float(np.percentile(ors, 2.5)), 3),
            "p97.5": round(float(np.percentile(ors, 97.5)), 3),
            "min": round(float(ors.min()), 3),
            "max": round(float(ors.max()), 3),
            "pct_within_30pct_of_baseline": round(float(np.mean(
                np.abs(ors / baseline_or - 1) <= 0.30) * 100), 1),
        },
        "pct_deep": {
            "median": float(np.median([r["dist_pct"]["C"] for r in runs])),
            "min": float(min(r["dist_pct"]["C"] for r in runs)),
            "max": float(max(r["dist_pct"]["C"] for r in runs)),
        },
    }


def _ablation_setup(combined: pd.DataFrame):
    """Precompute everything the ablations share on the Press-for-Accuracy set.

    Returns (n_utt, deep_match, interm_match, run). The match matrices are
    pattern x utterance; `run(deep_set, interm_set)` classifies under the given
    pattern sets and returns (stat block, labels). Classification is "any deep
    match -> C, else any intermediate match -> B, else A", identical to
    classify_mentalizing_depth's priority order; the full-set baseline is
    checked against the main classifier before returning.
    """
    subset = combined[(combined["t_move"] == "PressAccuracy")
                      & (combined["word_count"] > 3)].copy()
    texts = subset["Sentence"].astype(str).str.lower().str.strip().tolist()
    n_utt = len(texts)

    # Next student move within 3 turns (independent of the depth labels)
    s_move = combined["s_move"].tolist()
    linked = np.zeros(n_utt, dtype=bool)
    ev = np.zeros(n_utt, dtype=bool)
    for i, idx in enumerate(subset.index):
        for offset in range(1, 4):
            j = idx + offset
            if j < len(s_move) and s_move[j] is not None:
                linked[i] = True
                ev[i] = s_move[j] == "ProvidingEvidence"
                break

    deep_match = np.array([[bool(re.search(p, t)) for t in texts] for p in DEEP_PATTERNS])
    interm_match = np.array([[bool(re.search(p, t)) for t in texts]
                             for p in INTERMEDIATE_PATTERNS])
    deep_idx = {p: i for i, p in enumerate(DEEP_PATTERNS)}
    interm_idx = {p: i for i, p in enumerate(INTERMEDIATE_PATTERNS)}
    none = np.zeros(n_utt, dtype=bool)

    def run(deep_set, interm_set=INTERMEDIATE_PATTERNS):
        rows = [deep_idx[p] for p in deep_set]
        deep_any = deep_match[rows].any(axis=0) if rows else none
        rows = [interm_idx[p] for p in interm_set]
        interm_any = interm_match[rows].any(axis=0) if rows else none
        labels = np.where(deep_any, "C", np.where(interm_any, "B", "A"))
        return _ablation_stat_block(labels, ev, linked, len(deep_set)), labels

    reference = subset["Sentence"].apply(classify_mentalizing_depth).to_numpy()
    assert (run(DEEP_PATTERNS)[1] == reference).all(), \
        "vectorised baseline != main classifier"
    return n_utt, deep_match, interm_match, run


def option5_pattern_ablation(combined: pd.DataFrame, output_dir: str) -> dict:
    """Re-run Study 2 on Press for Accuracy under reduced deep-pattern sets:
    split-half, leave-one-out, drop-top-k and random subsets."""
    print("\n" + "=" * 70)
    print("OPTION 5: Pattern-Ablation Sensitivity (Study 2)")
    print("=" * 70)

    n_utt, deep_match, _, run = _ablation_setup(combined)
    pat_idx = {p: i for i, p in enumerate(DEEP_PATTERNS)}
    baseline = run(DEEP_PATTERNS)[0]

    n_deep = len(DEEP_PATTERNS)
    print(f"\nPress for Accuracy (>3 words): N={n_utt:,}  "
          f"deep patterns={n_deep}  intermediate patterns={len(INTERMEDIATE_PATTERNS)}")

    # Frequency = independent match count in this set; ties keep list order
    freq = {p: int(deep_match[pat_idx[p]].sum()) for p in DEEP_PATTERNS}
    ranked = sorted(DEEP_PATTERNS, key=lambda p: -freq[p])
    # Utterances whose *only* deep match is this pattern
    unique = {p: int((deep_match[pat_idx[p]] & (deep_match.sum(axis=0) == 1)).sum())
              for p in DEEP_PATTERNS}

    def line(name, r):
        e = r["evidence_pct"]
        ci = r["binary_or_ci95"] or [None, None]
        print(f"  {name:<24}{r['n_deep_patterns']:>3}  deep={r['dist_pct']['C']:>4}%  "
              f"ev {e['A']}->{e['B']}->{e['C']}  mono={'Y' if r['monotonic'] else 'N'}  "
              f"chi2={r['chi2']}  V={r['cramers_v']}  "
              f"OR={r['binary_or']} [{ci[0]}, {ci[1]}]")

    print("\nBaseline")
    line("all patterns", baseline)
    base_or = baseline["binary_or"]

    # A. Split-half, frequency-balanced alternating assignment
    D1, D2 = ranked[0::2], ranked[1::2]
    split = {"D1": run(D1)[0], "D2": run(D2)[0], "D1_patterns": D1, "D2_patterns": D2,
             "D1_coverage": int(deep_match[[pat_idx[p] for p in D1]].any(axis=0).sum()),
             "D2_coverage": int(deep_match[[pat_idx[p] for p in D2]].any(axis=0).sum())}
    print("\nA. Split-half")
    line("D1", split["D1"])
    line("D2", split["D2"])

    # B. Leave-one-out
    loo = {}
    for p in DEEP_PATTERNS:
        r = run([q for q in DEEP_PATTERNS if q != p])[0]
        r["delta_binary_or_pct"] = round((r["binary_or"] / base_or - 1) * 100, 2)
        r["pattern_freq"] = freq[p]
        r["pattern_unique_matches"] = unique[p]
        loo[p] = r
    loo_summary = {"n_runs": len(loo), "pct_monotonic": round(
        float(np.mean([r["monotonic"] for r in loo.values()]) * 100), 1)}
    for key, get in [
        ("binary_or", lambda r: r["binary_or"]),
        ("chi2", lambda r: r["chi2"]),
        ("cramers_v", lambda r: r["cramers_v"]),
        ("pct_deep", lambda r: r["dist_pct"]["C"]),
        ("evidence_pct_A", lambda r: r["evidence_pct"]["A"]),
        ("evidence_pct_B", lambda r: r["evidence_pct"]["B"]),
        ("evidence_pct_C", lambda r: r["evidence_pct"]["C"]),
    ]:
        vals = {p: get(r) for p, r in loo.items()}
        base_val = get(baseline)
        most = max(vals, key=lambda p: abs(vals[p] - base_val))
        loo_summary[key] = {"min": min(vals.values()), "max": max(vals.values()),
                            "baseline": base_val, "most_influential_pattern": most,
                            "value_without_it": vals[most]}
    print("\nB. Leave-one-out")
    for key in ("binary_or", "cramers_v", "evidence_pct_C"):
        s = loo_summary[key]
        print(f"  {key:<16} range [{s['min']}, {s['max']}]  baseline {s['baseline']}  "
              f"most influential: {s['most_influential_pattern']!r} -> {s['value_without_it']}")
    print(f"  monotonic in {loo_summary['pct_monotonic']}% of runs")

    # C. Drop top-k most frequent
    drop = {}
    print("\nC. Drop top-k most frequent")
    for k in DROP_TOPK:
        r = run(ranked[k:])[0]
        r["dropped_patterns"] = ranked[:k]
        drop[str(k)] = r
        line(f"drop top-{k}", r)

    # D. Random subsets
    rng = np.random.default_rng(ABLATION_SEED)
    random_runs, random_summary = {}, {}
    print(f"\nD. Random subsets (B={RANDOM_DRAWS}, seed={ABLATION_SEED})")
    for m in RANDOM_SIZES:
        runs = []
        for _ in range(RANDOM_DRAWS):
            draw = [str(p) for p in rng.choice(DEEP_PATTERNS, m, replace=False)]
            r = run(draw)[0]
            r["patterns"] = draw
            runs.append(r)
        random_runs[str(m)] = runs
        random_summary[str(m)] = s = _summarise_runs(runs, base_or)
        print(f"  m={m:<3} monotonic {s['pct_monotonic']}%  "
              f"OR median {s['binary_or']['median']} "
              f"[{s['binary_or']['p2.5']}, {s['binary_or']['p97.5']}]  "
              f"within +/-30% of baseline {s['binary_or']['pct_within_30pct_of_baseline']}%  "
              f"p<.001 {s['pct_binary_p_lt_001']}%")

    results = {
        "n_utterances": n_utt,
        "n_deep_patterns": n_deep,
        "n_intermediate_patterns": len(INTERMEDIATE_PATTERNS),
        "seed": ABLATION_SEED,
        "deep_pattern_frequency": {p: {"matches": freq[p], "unique_matches": unique[p]}
                                   for p in ranked},
        "baseline": baseline,
        "split_half": split,
        "loo": loo,
        "loo_summary": loo_summary,
        "drop_topk": drop,
        "random_summary": random_summary,
        "random": random_runs,
    }

    out = os.path.join(output_dir, "additional_option5_ablation.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")

    loo_df = pd.DataFrame([{
        "left_out_pattern": p,
        "pattern_freq": r["pattern_freq"],
        "pattern_unique_matches": r["pattern_unique_matches"],
        "pct_deep": r["dist_pct"]["C"],
        "evidence_pct_A": r["evidence_pct"]["A"],
        "evidence_pct_B": r["evidence_pct"]["B"],
        "evidence_pct_C": r["evidence_pct"]["C"],
        "evidence_n_C": r["evidence_n"]["C"],
        "monotonic": r["monotonic"],
        "chi2": r["chi2"],
        "cramers_v": r["cramers_v"],
        "binary_or": r["binary_or"],
        "binary_or_ci_low": r["binary_or_ci95"][0],
        "binary_or_ci_high": r["binary_or_ci95"][1],
        "delta_binary_or_pct": r["delta_binary_or_pct"],
    } for p, r in loo.items()])
    loo_df = loo_df.reindex(loo_df["delta_binary_or_pct"].abs()
                            .sort_values(ascending=False).index)
    out_csv = os.path.join(output_dir, "additional_option5_loo.csv")
    loo_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    return results


# ============================================================
# OPTION 5b — Pattern ablation over both tiers (spec v2)
# ============================================================

INTERM_RANDOM_SIZES = (4, 7, 10)
BASELINE_OR = (4.593, [4.071, 5.181])   # v1 baseline, must reproduce exactly


def _split_half(patterns: list, freq: dict):
    """Frequency-balanced alternating split; ties keep list order."""
    ranked = sorted(patterns, key=lambda p: -freq[p])
    return ranked, ranked[0::2], ranked[1::2]


def option5b_two_tier_ablation(combined: pd.DataFrame, output_dir: str) -> dict:
    """Spec v2: ablate the intermediate tier (E), split all 36 patterns into
    two complete two-tier classifiers (F, primary), and remove each tier
    entirely (G)."""
    print("\n" + "=" * 70)
    print("OPTION 5b: Two-Tier Pattern Ablation (Study 2, spec v2)")
    print("=" * 70)

    n_utt, deep_match, interm_match, run = _ablation_setup(combined)
    D, I = DEEP_PATTERNS, INTERMEDIATE_PATTERNS

    # --- Prerequisites: counts, frequencies, dead patterns
    deep_freq = {p: int(deep_match[i].sum()) for i, p in enumerate(D)}
    interm_freq = {p: int(interm_match[i].sum()) for i, p in enumerate(I)}
    interm_unique = {p: int((interm_match[i] & (interm_match.sum(axis=0) == 1)).sum())
                     for i, p in enumerate(I)}
    # B-level coverage: intermediate matches not pre-empted by a deep match
    deep_any = deep_match.any(axis=0)
    interm_effective = {p: int((interm_match[i] & ~deep_any).sum()) for i, p in enumerate(I)}
    dead = {"deep": [p for p in D if deep_freq[p] == 0],
            "intermediate": [p for p in I if interm_freq[p] == 0]}
    n_live = len(D) + len(I) - len(dead["deep"]) - len(dead["intermediate"])

    print(f"\nPress for Accuracy (>3 words): N={n_utt:,}")
    print(f"Pattern counts from analysis_pipeline: deep={len(D)}, intermediate={len(I)}, "
          f"total={len(D) + len(I)}, live={n_live}")
    print(f"Dead deep ({len(dead['deep'])}): {dead['deep']}")
    print(f"Dead intermediate ({len(dead['intermediate'])}): {dead['intermediate']}")
    print("\nIntermediate frequency (matches / unique / not pre-empted by deep):")
    for p in sorted(I, key=lambda p: -interm_freq[p]):
        print(f"  {interm_freq[p]:>5} {interm_unique[p]:>5} {interm_effective[p]:>5}  {p}")

    dead_out = {
        "n_utterances": n_utt,
        "n_deep_patterns": len(D),
        "n_intermediate_patterns": len(I),
        "n_live_patterns": n_live,
        "dead_deep": dead["deep"],
        "dead_intermediate": dead["intermediate"],
        "deep_frequency": dict(sorted(deep_freq.items(), key=lambda kv: -kv[1])),
        "intermediate_frequency": {
            p: {"matches": interm_freq[p], "unique_matches": interm_unique[p],
                "matches_not_preempted_by_deep": interm_effective[p]}
            for p in sorted(I, key=lambda p: -interm_freq[p])},
    }
    out = os.path.join(output_dir, "additional_option5b_deadpatterns.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dead_out, f, indent=2)
    print(f"Saved: {out}")

    def go(deep_set, interm_set, **extra):
        r = run(deep_set, interm_set)[0]
        r["n_interm_patterns"] = len(interm_set)
        r.update(extra)
        return r

    def line(name, r):
        e = r["evidence_pct"]
        ci = r["binary_or_ci95"] or [None, None]
        print(f"  {name:<18}D={r['n_deep_patterns']:>2} I={r['n_interm_patterns']:>2}  "
              f"A/B/C={r['dist_pct']['A']}/{r['dist_pct']['B']}/{r['dist_pct']['C']}%  "
              f"ev {e['A']}->{e['B']}->{e['C']}  mono={'Y' if r['monotonic'] else 'N'}  "
              f"OR={r['binary_or']} [{ci[0]}, {ci[1]}]  p={r['binary_p']:.1e}"
              f"{'  DEGENERATE' if r['degenerate'] else ''}")

    # --- Baseline, must reproduce v1 exactly
    baseline = go(D, I)
    print("\nBaseline")
    line("all 36", baseline)
    if (baseline["binary_or"], baseline["binary_or_ci95"]) != BASELINE_OR:
        raise RuntimeError(f"Baseline OR {baseline['binary_or']} {baseline['binary_or_ci95']} "
                           f"!= v1 {BASELINE_OR}; reconcile before interpreting.")
    base_or = baseline["binary_or"]

    def delta(r):
        return round((r["binary_or"] / base_or - 1) * 100, 2)

    deep_ranked, D1, D2 = _split_half(D, deep_freq)
    interm_ranked, I1, I2 = _split_half(I, interm_freq)

    # --- F. Combined-set split-half (primary)
    F = {"P1": go(D1, I1), "P2": go(D2, I2),
         "P1_patterns": {"deep": D1, "intermediate": I1},
         "P2_patterns": {"deep": D2, "intermediate": I2}}
    for k in ("P1", "P2"):
        F[k]["delta_binary_or_pct"] = delta(F[k])
    print("\nF. Combined-set split-half (primary)")
    line("P1", F["P1"])
    line("P2", F["P2"])

    # --- E1. Intermediate split-half
    E1 = {"I1": go(D, I1), "I2": go(D, I2), "I1_patterns": I1, "I2_patterns": I2}
    for k in ("I1", "I2"):
        E1[k]["delta_binary_or_pct"] = delta(E1[k])
    print("\nE1. Intermediate split-half (deep = all 22)")
    line("I1", E1["I1"])
    line("I2", E1["I2"])

    # --- E2. Intermediate leave-one-out
    E2 = {}
    for p in I:
        r = go(D, [q for q in I if q != p], pattern_freq=interm_freq[p],
               pattern_unique_matches=interm_unique[p])
        r["delta_binary_or_pct"] = delta(r)
        E2[p] = r
    valid = {p: r for p, r in E2.items() if not r["degenerate"]}
    most = max(valid, key=lambda p: abs(valid[p]["delta_binary_or_pct"]))
    E2_summary = {
        "n_runs": len(E2),
        "n_degenerate_excluded": len(E2) - len(valid),
        "binary_or_min": min(r["binary_or"] for r in valid.values()),
        "binary_or_max": max(r["binary_or"] for r in valid.values()),
        "evidence_pct_B_min": min(r["evidence_pct"]["B"] for r in valid.values()),
        "evidence_pct_B_max": max(r["evidence_pct"]["B"] for r in valid.values()),
        "pct_monotonic": round(float(np.mean([r["monotonic"] for r in valid.values()])
                                     * 100), 1),
        "most_influential_pattern": most,
        "max_abs_delta_binary_or_pct": abs(valid[most]["delta_binary_or_pct"]),
        "binary_or_without_it": valid[most]["binary_or"],
    }
    print("\nE2. Intermediate leave-one-out")
    print(f"  OR range [{E2_summary['binary_or_min']}, {E2_summary['binary_or_max']}]  "
          f"B evidence range [{E2_summary['evidence_pct_B_min']}, "
          f"{E2_summary['evidence_pct_B_max']}]  monotonic {E2_summary['pct_monotonic']}%  "
          f"degenerate excluded {E2_summary['n_degenerate_excluded']}")
    print(f"  most influential: {most!r}  OR -> {E2_summary['binary_or_without_it']} "
          f"({valid[most]['delta_binary_or_pct']:+}%)")

    # --- E3. Drop top-k most frequent intermediate
    E3 = {}
    print("\nE3. Drop top-k most frequent intermediate")
    for k in DROP_TOPK:
        r = go(D, interm_ranked[k:], dropped_patterns=interm_ranked[:k])
        r["delta_binary_or_pct"] = delta(r)
        E3[str(k)] = r
        line(f"drop top-{k}", r)

    # --- E4. Random intermediate subsets
    rng = np.random.default_rng(ABLATION_SEED)
    E4_runs, E4_summary = {}, {}
    print(f"\nE4. Random intermediate subsets (B={RANDOM_DRAWS}, seed={ABLATION_SEED})")
    for m in INTERM_RANDOM_SIZES:
        runs = []
        for _ in range(RANDOM_DRAWS):
            draw = [str(p) for p in rng.choice(I, m, replace=False)]
            runs.append(go(D, draw, patterns=draw))
        valid_runs = [r for r in runs if not r["degenerate"]]
        E4_runs[str(m)] = runs
        E4_summary[str(m)] = s = _summarise_runs(valid_runs, base_or)
        s["n_degenerate_excluded"] = len(runs) - len(valid_runs)
        s["pct_binary_or_gt_1"] = round(float(np.mean(
            [r["binary_or"] > 1 for r in valid_runs]) * 100), 1)
        print(f"  m={m:<3} OR median {s['binary_or']['median']} "
              f"[{s['binary_or']['p2.5']}, {s['binary_or']['p97.5']}]  "
              f"range [{s['binary_or']['min']}, {s['binary_or']['max']}]  "
              f"within +/-30% {s['binary_or']['pct_within_30pct_of_baseline']}%  "
              f"p<.001 {s['pct_binary_p_lt_001']}%  monotonic {s['pct_monotonic']}%  "
              f"degenerate excluded {s['n_degenerate_excluded']}")

    # --- G. Tier-removal anchors. An empty level is expected by construction,
    # so degeneracy is judged on the tier that remains.
    G1 = go([], I)
    G1["degenerate"] = G1["evidence_n"]["B"] < 30
    G2 = go(D, [])
    G2["degenerate"] = G2["evidence_n"]["C"] < 30
    G = {"G1_no_deep": G1, "G2_no_intermediate": G2}
    for r in G.values():
        r["delta_binary_or_pct"] = delta(r)
    print("\nG. Tier-removal anchors")
    line("G1 no deep", G1)
    line("G2 no interm", G2)

    # --- Pre-registered verdict (spec v2 §5)
    halves = [F["P1"], F["P2"]]
    halves_ok = [not r["degenerate"] for r in halves]
    within = all(abs(r["delta_binary_or_pct"]) <= 30 for r in halves)
    signif = all(r["binary_p"] < 0.001 for r in halves)
    same_sign = all(r["binary_or"] > 1 for r in halves)
    e2_ok = E2_summary["max_abs_delta_binary_or_pct"] <= 15
    if not all(halves_ok):
        verdict = "undetermined (degenerate split-half)"
    elif within and signif and e2_ok:
        verdict = "robust"
    elif signif and same_sign:
        verdict = "partly robust"
    else:
        verdict = "fragile"
    criteria = {
        "F_halves_non_degenerate": all(halves_ok),
        "F_both_within_30pct": within,
        "F_both_p_lt_001": signif,
        "F_both_or_gt_1": same_sign,
        "E2_max_shift_le_15pct": e2_ok,
    }
    print(f"\nPre-registered verdict: {verdict.upper()}")
    for k, v in criteria.items():
        print(f"  {k:<26}{v}")

    results = {
        "spec": "specs/pattern_ablation_spec_v2.md",
        "n_utterances": n_utt,
        "seed": ABLATION_SEED,
        "verdict": verdict,
        "verdict_criteria": criteria,
        "patterns": dead_out,
        "baseline": baseline,
        "F_combined_split_half": F,
        "E1_interm_split_half": E1,
        "E2_interm_loo": E2,
        "E2_summary": E2_summary,
        "E3_interm_drop_topk": E3,
        "E4_summary": E4_summary,
        "E4_random": E4_runs,
        "G_tier_removal": G,
    }
    out = os.path.join(output_dir, "additional_option5b_ablation.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")

    loo_df = pd.DataFrame([{
        "left_out_pattern": p,
        "pattern_freq": r["pattern_freq"],
        "pattern_unique_matches": r["pattern_unique_matches"],
        "pct_intermediate": r["dist_pct"]["B"],
        "evidence_pct_A": r["evidence_pct"]["A"],
        "evidence_pct_B": r["evidence_pct"]["B"],
        "evidence_pct_C": r["evidence_pct"]["C"],
        "evidence_n_B": r["evidence_n"]["B"],
        "monotonic": r["monotonic"],
        "degenerate": r["degenerate"],
        "chi2": r["chi2"],
        "cramers_v": r["cramers_v"],
        "binary_or": r["binary_or"],
        "binary_or_ci_low": r["binary_or_ci95"][0],
        "binary_or_ci_high": r["binary_or_ci95"][1],
        "delta_binary_or_pct": r["delta_binary_or_pct"],
    } for p, r in E2.items()])
    loo_df = loo_df.reindex(loo_df["delta_binary_or_pct"].abs()
                            .sort_values(ascending=False).index)
    out_csv = os.path.join(output_dir, "additional_option5b_loo.csv")
    loo_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    return results


# ============================================================
# OPTION 6 — Matched transcript-level effect size (2 x 2 on TalkMoves)
# ============================================================

# Shared measurement config (specs/matched_effect_size_spec.md §3), read by
# every cell and by the NCTE reference — never re-declared per cell.
#
# Depth encoding: both pipelines already use {0, 1, 2}. The category mapping
# via L3_DEPTH_MAP (0 none / 1 surface / 2 deep), the pattern classifier via
# A/B/C -> 0/1/2 (as in ncte_replication.study_r2_sequential). The {0, .5, 1}
# hypothesis in the spec came from reading DV rates (0.105/0.078) as depths.
DEPTH_ENCODING = {"A": 0, "B": 1, "C": 2}
DEPTH_ENCODING_DESC = ("{0,1,2}: category mapping None/Context/KeepingTogether/Marking=0, "
                       "Restating/Revoicing/PressAccuracy=1, PressReasoning/GettingStudentsRelate=2; "
                       "pattern classifier A=0, B=1, C=2")
MIN_TEACHER_UTT = 20
MIN_STUDENT_UTT = 10
MIN_UTTERANCE_RULE = (f">={MIN_TEACHER_UTT} teacher and >={MIN_STUDENT_UTT} student utterances "
                      f"per transcript, any length (counted before the >3-word classifier filter)")
CLASSIFIER_MIN_WORDS = 3     # classifier scores teacher utterances with > 3 words
ELABORATE_MIN_WORDS = 10
BOOT_B = 10_000
BOOT_SEED = 42
STUDY1_GATE = {"n_transcripts": 536, "cohens_d": 0.468, "t_stat": 5.42,
               "high_mean": 0.193, "low_mean": 0.129}
NCTE_PUBLISHED = {"n_transcripts": 1654, "cohens_d": 0.433}
MATCHED_BAND = (0.35, 0.55)

MATCHED_CELLS = OrderedDict([
    ("A", ("category_mapping", "evidence_tag")),     # reproduce 0.468
    ("B", ("category_mapping", "elaborate_10w")),    # DV effect
    ("C", ("pattern_classifier", "evidence_tag")),   # IV effect
    ("D", ("pattern_classifier", "elaborate_10w")),  # matched to NCTE
])


def _turn_elaboration_rate(student: pd.DataFrame) -> pd.Series:
    """>=10-word rate over student *turns*: consecutive student rows sharing a
    `Turn` value are merged, matching NCTE where one row is one speaker turn.
    Rows with a missing `Turn` stay single-sentence turns."""
    st = student.reindex(columns=["transcript_id", "Turn", "word_count"])
    pos = pd.Series(st.index, index=st.index)
    new_turn = ((st["transcript_id"] != st["transcript_id"].shift())
                | (st["Turn"] != st["Turn"].shift())
                | (pos.diff() != 1))
    st["turn_key"] = new_turn.cumsum()
    turns = st.groupby(["transcript_id", "turn_key"])["word_count"].sum()
    return (turns >= ELABORATE_MIN_WORDS).groupby(level=0).mean()


def _talkmoves_transcript_table(combined: pd.DataFrame) -> pd.DataFrame:
    """One row per transcript: mean depth under each IV, rate under each DV."""
    teacher = combined[combined["t_move"].notna()]
    student = combined[combined["s_move"].notna()]
    scored = teacher[teacher["word_count"] > CLASSIFIER_MIN_WORDS].copy()
    scored["pc_depth"] = (scored["Sentence"].apply(classify_mentalizing_depth)
                          .map(DEPTH_ENCODING))

    by_t = teacher.groupby("transcript_id")
    by_s = student.groupby("transcript_id")
    tx = pd.DataFrame({
        # Study 1 counts teacher utterances with a mapped L3 level
        "n_teacher": by_t["l3_depth"].count(),
        "n_scored": scored.groupby("transcript_id").size(),
        "n_student": by_s.size(),
        "category_mapping": by_t["l3_depth"].mean(),
        "pattern_classifier": scored.groupby("transcript_id")["pc_depth"].mean(),
        "evidence_tag": by_s["s_move"].apply(lambda s: (s == "ProvidingEvidence").mean()),
        "elaborate_10w": by_s["word_count"].apply(lambda w: (w >= ELABORATE_MIN_WORDS).mean()),
        "elaborate_10w_turn": _turn_elaboration_rate(student),
    })
    tx["n_scored"] = tx["n_scored"].fillna(0).astype(int)
    return tx


def _ncte_transcript_table(ncte_dir: str) -> pd.DataFrame:
    """NCTE per-transcript table, as in ncte_replication.study_r2_sequential."""
    su = pd.read_csv(os.path.join(ncte_dir, "ncte_single_utterances.csv"))
    text_col = "cleaned_text" if "cleaned_text" in su.columns else "text"
    teacher = su[su["speaker"] == "teacher"]
    student = su[su["speaker"].isin(["student", "multiple students"])]
    scored = teacher[teacher["num_words"] > CLASSIFIER_MIN_WORDS].copy()
    scored["pc_depth"] = (scored[text_col].apply(classify_mentalizing_depth)
                          .map(DEPTH_ENCODING))
    tx = pd.DataFrame({
        "n_teacher": teacher.groupby("OBSID").size(),
        "n_scored": scored.groupby("OBSID").size(),
        "n_student": student.groupby("OBSID").size(),
        "pattern_classifier": scored.groupby("OBSID")["pc_depth"].mean(),
        "elaborate_10w": student.groupby("OBSID")["num_words"].apply(
            lambda w: (w >= ELABORATE_MIN_WORDS).mean()),
    })
    tx["n_scored"] = tx["n_scored"].fillna(0).astype(int)
    print(f"  NCTE: {len(su):,} utterances, {len(scored):,} scored teacher utterances, "
          f"{su['OBSID'].nunique()} transcripts")
    return tx


def _include(tx: pd.DataFrame, teacher_col: str = "n_teacher") -> pd.DataFrame:
    return tx[(tx[teacher_col] >= MIN_TEACHER_UTT) & (tx["n_student"] >= MIN_STUDENT_UTT)]


def _matched_effect(tx: pd.DataFrame, iv: str, dv: str, rule: str) -> dict:
    """Median split on mean depth; Cohen's d (pooled SD as in Study 1 / R2d),
    Hedges' g, Student t-test and a stratified percentile bootstrap CI."""
    sub = tx.dropna(subset=[iv, dv])
    med = sub[iv].median()
    hi = sub.loc[sub[iv] >= med, dv].to_numpy(float)
    lo = sub.loc[sub[iv] < med, dv].to_numpy(float)

    def d_of(h, l, axis=None):
        sd = np.sqrt((h.var(axis=axis, ddof=1) + l.var(axis=axis, ddof=1)) / 2)
        return (h.mean(axis=axis) - l.mean(axis=axis)) / sd

    d = float(d_of(hi, lo))
    j = 1 - 3 / (4 * (len(hi) + len(lo)) - 9)
    t, p = stats.ttest_ind(hi, lo)

    # Resample transcripts within each half, holding the split fixed
    rng = np.random.default_rng(BOOT_SEED)
    bh = hi[rng.integers(0, len(hi), (BOOT_B, len(hi)))]
    bl = lo[rng.integers(0, len(lo), (BOOT_B, len(lo)))]
    boot = d_of(bh, bl, axis=1)
    ci = np.percentile(boot, [2.5, 97.5])

    return {
        "iv": iv,
        "dv": dv,
        "n_transcripts": int(len(sub)),
        "min_utterance_rule": rule,
        "depth_encoding": DEPTH_ENCODING_DESC,
        "mean_depth": round(float(sub[iv].mean()), 4),
        "median_depth": round(float(med), 4),
        "high_mean": round(float(hi.mean()), 3),
        "low_mean": round(float(lo.mean()), 3),
        "high_sd": round(float(hi.std(ddof=1)), 3),
        "low_sd": round(float(lo.std(ddof=1)), 3),
        "high_n": int(len(hi)),
        "low_n": int(len(lo)),
        "cohens_d": round(d, 3),
        "hedges_g": round(d * j, 3),
        "ci95": [round(float(ci[0]), 3), round(float(ci[1]), 3)],
        "hedges_g_ci95": [round(float(ci[0] * j), 3), round(float(ci[1] * j), 3)],
        "t_stat": round(float(t), 3),
        "p": float(p),
    }


def _cell_line(name: str, r: dict) -> str:
    return (f"  {name:<3} {r['iv']:<19} x {r['dv']:<19} N={r['n_transcripts']:<5} "
            f"hi={r['high_mean']:.3f} lo={r['low_mean']:.3f}  d={r['cohens_d']:.3f} "
            f"[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]  g={r['hedges_g']:.3f}  "
            f"t={r['t_stat']:.2f} p={r['p']:.2e}")


def option6_matched_effect_size(combined: pd.DataFrame, ncte_dir: str,
                                output_dir: str) -> dict:
    """2 x 2 (IV: category mapping | pattern classifier) x (DV: evidence tag |
    >=10-word response) transcript-level median split on TalkMoves, to put the
    NCTE d = 0.433 next to a like-for-like TalkMoves value (cell D)."""
    print("\n" + "=" * 70)
    print("OPTION 6: Matched Transcript-Level Effect Size (2 x 2)")
    print("=" * 70)
    print(f"  Depth encoding: {DEPTH_ENCODING_DESC}")
    print(f"  Inclusion rule: {MIN_UTTERANCE_RULE}")

    tx_all = _talkmoves_transcript_table(combined)
    tx = _include(tx_all)
    print(f"  TalkMoves: {len(tx_all)} transcripts, {len(tx)} meet the inclusion rule")

    results = OrderedDict()
    results["config"] = {
        "depth_encoding": DEPTH_ENCODING_DESC,
        "min_utterance_rule": MIN_UTTERANCE_RULE,
        "classifier_scope": (f"all teacher utterances with >{CLASSIFIER_MIN_WORDS} words, "
                             f"talk-move labels ignored (as on NCTE)"),
        "elaborate_threshold_words": ELABORATE_MIN_WORDS,
        "bootstrap": {"B": BOOT_B, "seed": BOOT_SEED,
                      "method": "percentile; transcripts resampled within each half, split fixed"},
        "cohens_d_pooled_sd": "sqrt((var_high + var_low) / 2), as in Study 1 and NCTE R2d",
        "talkmoves_n_transcripts_loaded": int(len(tx_all)),
        "talkmoves_n_transcripts_included": int(len(tx)),
        "talkmoves_scored_teacher_utterances": int(tx_all["n_scored"].sum()),
    }

    # Validation gate: cell A must reproduce the published Study 1 numbers
    a = _matched_effect(tx, *MATCHED_CELLS["A"], MIN_UTTERANCE_RULE)
    observed = {"n_transcripts": a["n_transcripts"], "cohens_d": a["cohens_d"],
                "t_stat": round(a["t_stat"], 2), "high_mean": a["high_mean"],
                "low_mean": a["low_mean"]}
    passed = observed == STUDY1_GATE
    results["validation_gate"] = {"passed": passed, "expected": STUDY1_GATE,
                                  "observed": observed}
    print(f"\n  Validation gate (cell A vs Study 1): {'PASS' if passed else 'FAIL'}")
    print(f"    expected {STUDY1_GATE}\n    observed {observed}")
    if not passed:
        out = os.path.join(output_dir, "additional_option6_matched_d.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        raise SystemExit(f"Cell A does not reproduce Study 1 — reconcile before "
                         f"interpreting B/C/D. Gate written to {out}")

    print()
    cells = OrderedDict()
    for name, (iv, dv) in MATCHED_CELLS.items():
        cells[name] = a if name == "A" else _matched_effect(tx, iv, dv, MIN_UTTERANCE_RULE)
        print(_cell_line(name, cells[name]))
    results["cells"] = cells

    d = {k: cells[k]["cohens_d"] for k in cells}
    results["decomposition"] = {
        "A_minus_D": round(d["A"] - d["D"], 3),
        "dv_effect_A_minus_B": round(d["A"] - d["B"], 3),
        "iv_effect_A_minus_C": round(d["A"] - d["C"], 3),
        "interaction_A_minus_B_minus_C_plus_D": round(d["A"] - d["B"] - d["C"] + d["D"], 3),
        "identity": "A - D = (A - B) + (A - C) - (A - B - C + D)",
    }
    print(f"\n  A - D = {results['decomposition']['A_minus_D']:+.3f}  "
          f"(DV A-B {results['decomposition']['dv_effect_A_minus_B']:+.3f}, "
          f"IV A-C {results['decomposition']['iv_effect_A_minus_C']:+.3f}, "
          f"interaction {results['decomposition']['interaction_A_minus_B_minus_C_plus_D']:+.3f})")

    # Sensitivity: unit of the >=10-word DV and the teacher-count rule
    sens = OrderedDict()
    sens["D_turn_level_dv"] = _matched_effect(
        tx, "pattern_classifier", "elaborate_10w_turn", MIN_UTTERANCE_RULE)
    sens["D_turn_level_dv"]["note"] = ("TalkMoves rows are sentences, NCTE rows are turns; "
                                       "consecutive student rows with the same Turn merged")
    scored_rule = (f">={MIN_TEACHER_UTT} scored (>{CLASSIFIER_MIN_WORDS}-word) teacher and "
                   f">={MIN_STUDENT_UTT} student utterances (original NCTE R2c rule)")
    sens["D_ncte_original_rule"] = _matched_effect(
        _include(tx_all, "n_scored"), "pattern_classifier", "elaborate_10w", scored_rule)
    results["sensitivity"] = sens
    print("\n  Sensitivity:")
    print(_cell_line("D/t", sens["D_turn_level_dv"]))
    print(_cell_line("D/r", sens["D_ncte_original_rule"]))

    # NCTE reference, same effect-size function, both inclusion rules
    ncte_path = os.path.join(ncte_dir, "ncte_single_utterances.csv")
    if os.path.exists(ncte_path):
        print("\n  NCTE reference:")
        ntx = _ncte_transcript_table(ncte_dir)
        ncte = OrderedDict()
        ncte["original_rule"] = _matched_effect(
            _include(ntx, "n_scored"), "pattern_classifier", "elaborate_10w", scored_rule)
        ncte["harmonised_rule"] = _matched_effect(
            _include(ntx), "pattern_classifier", "elaborate_10w", MIN_UTTERANCE_RULE)
        ok = (ncte["original_rule"]["n_transcripts"] == NCTE_PUBLISHED["n_transcripts"]
              and ncte["original_rule"]["cohens_d"] == NCTE_PUBLISHED["cohens_d"])
        ncte["reproduces_published"] = {"passed": ok, "expected": NCTE_PUBLISHED}
        results["ncte_reference"] = ncte
        print(_cell_line("N/o", ncte["original_rule"]))
        print(_cell_line("N/h", ncte["harmonised_rule"]))
        print(f"    reproduces published 0.433 / N=1654: {'PASS' if ok else 'FAIL'}")
    else:
        print(f"\n  [NCTE reference skipped: {ncte_path} not found]")

    # Pre-registered reading (spec §5), fixed before seeing D
    dd = cells["D"]
    if dd["p"] >= 0.05:
        verdict = "ESCALATE"
        reading = ("Cell D not significant: the replication claim needs rewriting, "
                   "not renumbering. Escalate before editing.")
    elif MATCHED_BAND[0] <= dd["cohens_d"] <= MATCHED_BAND[1]:
        verdict = "MATCHED"
        reading = (f"D = {dd['cohens_d']:.3f} lies in {MATCHED_BAND}: the cross-corpus "
                   f"comparison holds under matched measurement. Report D vs 0.433; "
                   f"0.468 stays as the Study 1 result.")
    else:
        verdict = "CONFOUNDED"
        reading = (f"D = {dd['cohens_d']:.3f} lies outside {MATCHED_BAND}: the 0.468 vs 0.433 "
                   f"similarity was partly coincidental. Report D vs 0.433 and note in "
                   f"Limitations that the two operationalisations give different "
                   f"transcript-level effect sizes on the same corpus.")
    results["interpretation"] = {"band": list(MATCHED_BAND), "verdict": verdict,
                                 "reading": reading,
                                 "matched_pair": {"talkmoves_D": dd["cohens_d"],
                                                  "ncte": NCTE_PUBLISHED["cohens_d"]}}
    print(f"\n  Verdict: {verdict}\n    {reading}")

    out = os.path.join(output_dir, "additional_option6_matched_d.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    out_csv = os.path.join(output_dir, "additional_option6_transcripts.csv")
    tx.to_csv(out_csv)
    print(f"\nSaved: {out}\nSaved: {out_csv}")
    return results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="DToM additional experiments")
    parser.add_argument("--option", default="all",
                        choices=["all", "1", "2", "2-extract", "2-summarize", "3", "4", "5", "5b", "6"])
    parser.add_argument("--data-dir", default="data/TalkMoves/data")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--ncte-data-dir", default="data/NCTE",
                        help="NCTE CSV directory for the Option 6 reference values")
    parser.add_argument("--coding-path", default="output/additional_option2_coding.json",
                        help="JSON map id->reason category for Option 2 summary")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    need_corpus = args.option in ("all", "3", "4", "5", "5b", "6")
    combined = load_transcripts(args.data_dir) if need_corpus else None

    if args.option in ("all", "1"):
        option1_confusion_matrix(args.output_dir)
    if args.option in ("all", "2", "2-extract"):
        option2_extract_lifts(args.output_dir)
    if args.option in ("2-summarize",) or (args.option == "all" and os.path.exists(args.coding_path)):
        if os.path.exists(args.coding_path):
            option2_summarize(args.output_dir, args.coding_path)
        else:
            print(f"\n[Option 2 summary skipped: {args.coding_path} not found. "
                  f"Code the reasons first, then run --option 2-summarize]")
    if args.option in ("all", "3"):
        option3_within_category(combined, args.output_dir)
    if args.option in ("all", "4"):
        option4_grade_level(combined, args.data_dir, args.output_dir)
    if args.option in ("all", "5"):
        option5_pattern_ablation(combined, args.output_dir)
    if args.option in ("all", "5b"):
        option5b_two_tier_ablation(combined, args.output_dir)
    if args.option in ("all", "6"):
        option6_matched_effect_size(combined, args.ncte_data_dir, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
