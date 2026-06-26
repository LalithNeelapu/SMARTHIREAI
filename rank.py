#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ranker import run_ranking


def main():
    parser = argparse.ArgumentParser(
        description="Rank candidates for the Redrob Senior AI Engineer role."
    )
    parser.add_argument(
        "--candidates",
        default=str(
            ROOT / "data" / "[PUB] India_runs_data_and_ai_challenge"
            / "[PUB] India_runs_data_and_ai_challenge"
            / "India_runs_data_and_ai_challenge" / "candidates.jsonl"
        ),
        help="Path to candidates.jsonl",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "output" / "final_submission.csv"),
        help="Path to write submission CSV",
    )
    parser.add_argument(
        "--filtered-cache",
        default=str(ROOT / "output" / "filtered_candidates.jsonl"),
        help="Optional path to cache the filtered candidate pool",
    )
    parser.add_argument(
        "--model",
        choices=("fast", "quality"),
        default="fast",
        help="Embedding model: fast=MiniLM (~10 min), quality=BGE (~35 min)",
    )
    args = parser.parse_args()

    model_name = (
        "BAAI/bge-small-en-v1.5"
        if args.model == "quality"
        else "sentence-transformers/all-MiniLM-L6-v2"
    )

    run_ranking(
        candidates_path=args.candidates,
        output_path=args.out,
        filtered_cache=args.filtered_cache,
        model_name=model_name,
    )


if __name__ == "__main__":
    main()
