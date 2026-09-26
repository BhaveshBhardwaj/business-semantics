"""
Main entry point for Business Entity Resolution Challenge.
Generates matching_results.tsv and candidate_pairs.tsv.
"""

import os
import sys
import argparse

# Add current directory to path
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)

from src.pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="Business Entity Resolution Prediction Pipeline")
    parser.add_argument(
        "--test-dir",
        type=str,
        default=os.path.join(base_dir, "..", "..", "dataset", "test"),
        help="Path to folder containing test_source1.tsv, test_source2.tsv, test_source3.tsv"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(base_dir, "..", "..", "output"),
        help="Path to folder where matching_results.tsv and candidate_pairs.tsv will be written"
    )
    torch_default = os.path.join(base_dir, "models", "matching_torch.pt")
    xgb_default = os.path.join(base_dir, "models", "matching_xgb.json")
    lgb_default = os.path.join(base_dir, "models", "matching_lgbm.txt")
    if os.path.isfile(torch_default):
        default_model = torch_default
    elif os.path.isfile(xgb_default):
        default_model = xgb_default
    else:
        default_model = lgb_default

    parser.add_argument(
        "--model-path",
        type=str,
        default=default_model,
        help="Path to trained model file (supports PyTorch .pt, XGBoost .json, or LightGBM .txt)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold for entity matching (default: loaded from models/threshold.txt or 0.50)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Maximum candidates per Source 1 entity in blocking stage (default: 25)"
    )

    args = parser.parse_args()

    # Load threshold if not explicitly specified
    threshold = args.threshold
    if threshold is None:
        meta_path = os.path.join(base_dir, "models", "threshold.txt")
        if os.path.isfile(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                threshold = float(f.read().strip())
        else:
            threshold = 0.50

    run_pipeline(
        test_dir=os.path.abspath(args.test_dir),
        output_dir=os.path.abspath(args.output_dir),
        model_path=os.path.abspath(args.model_path),
        threshold=threshold,
        top_k=args.top_k
    )

if __name__ == "__main__":
    main()
