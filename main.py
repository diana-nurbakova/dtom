#!/usr/bin/env python3
"""
DToM — Double Theory of Mind Empirical Analysis
================================================

Runs all studies:
  Study 1: L3 Depth Mapping across talk move categories
  Study 2: Within-category mentalizing depth (rule-based)
  Study 3: Convergent validity via LLM classification
  NCTE Replication: Cross-dataset validation on NCTE transcripts

Usage:
  uv run main.py                          # Run Studies 1 & 2 only
  uv run main.py --with-llm               # Run all three studies
  uv run main.py --study3-only            # Run Study 3 only
  uv run main.py --ncte                   # Run NCTE replication only
  uv run main.py --ncte --ncte-data-dir data/NCTE  # Custom NCTE data path
"""

import argparse
import sys

from dotenv import load_dotenv


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="DToM Empirical Analysis — all studies"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/TalkMoves/data",
        help="Path to TalkMoves/data directory",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output",
        help="Directory for output files",
    )
    parser.add_argument(
        "--with-llm", action="store_true",
        help="Also run Study 3 (LLM classifier, requires ANTHROPIC_API_KEY)",
    )
    parser.add_argument(
        "--study3-only", action="store_true",
        help="Run only Study 3 (LLM classifier)",
    )
    parser.add_argument(
        "--ncte", action="store_true",
        help="Run NCTE replication (Studies R1-R3)",
    )
    parser.add_argument(
        "--ncte-data-dir", type=str, default="data/NCTE",
        help="Path to directory containing NCTE CSV files",
    )
    parser.add_argument(
        "--ncte-output-dir", type=str, default="ncte_output",
        help="Directory for NCTE replication output files",
    )
    args = parser.parse_args()

    if not args.study3_only and not args.ncte:
        print("=" * 70)
        print("RUNNING STUDIES 1 & 2 (Analysis Pipeline)")
        print("=" * 70)
        sys.argv = [
            "dtom_analysis_pipeline",
            "--data-dir", args.data_dir,
            "--output-dir", args.output_dir,
        ]
        from dtom.analysis_pipeline import main as run_pipeline
        run_pipeline()

    if (args.with_llm or args.study3_only) and not args.ncte:
        print("\n" + "=" * 70)
        print("RUNNING STUDY 3 (LLM Classifier)")
        print("=" * 70)
        sys.argv = [
            "dtom_llm_classifier",
            "--data-dir", args.data_dir,
            "--output-dir", args.output_dir,
        ]
        from dtom.llm_classifier import main as run_llm
        run_llm()

    if args.ncte:
        print("\n" + "=" * 70)
        print("RUNNING NCTE REPLICATION (Studies R1-R3)")
        print("=" * 70)
        sys.argv = [
            "dtom_ncte_replication",
            "--data-dir", args.ncte_data_dir,
            "--output-dir", args.ncte_output_dir,
        ]
        from dtom.ncte_replication import main as run_ncte
        run_ncte()


if __name__ == "__main__":
    main()
