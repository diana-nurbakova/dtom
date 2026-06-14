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

Options 1 & 2 reuse the Study-3 LLM output files in `output/`.
Options 3 & 4 re-run the rule-based pipeline on the TalkMoves corpus.

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
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="DToM additional experiments")
    parser.add_argument("--option", default="all",
                        choices=["all", "1", "2", "2-extract", "2-summarize", "3", "4"])
    parser.add_argument("--data-dir", default="data/TalkMoves/data")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--coding-path", default="output/additional_option2_coding.json",
                        help="JSON map id->reason category for Option 2 summary")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    need_corpus = args.option in ("all", "3", "4")
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

    print("\nDone.")


if __name__ == "__main__":
    main()
