#!/usr/bin/env python3
"""
DToM — Double Theory of Mind Empirical Analysis
================================================

Runs all three studies:
  Study 1: L3 Depth Mapping across talk move categories
  Study 2: Within-category mentalizing depth (rule-based)
  Study 3: Convergent validity via LLM classification

Usage:
  uv run main.py                          # Run Studies 1 & 2 only
  uv run main.py --with-llm               # Run all three studies
  uv run main.py --study3-only            # Run Study 3 only
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
    args = parser.parse_args()

    if not args.study3_only:
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

    if args.with_llm or args.study3_only:
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


if __name__ == "__main__":
    main()
