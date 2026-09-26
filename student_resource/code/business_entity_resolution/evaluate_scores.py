"""
Benchmark Evaluation Script for Business Entity Resolution.
Evaluates Macro F_0.5, Precision, Recall, Singleton Accuracy, and Blocking Reduction Ratio
on a held-out validation split of ground-truth labeled entities.
"""

import os
import sys
import time
import argparse
from collections import defaultdict
import numpy as np

# Ensure code directory is in sys.path
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)

from src.preprocess import clean_text, extract_numbers, make_tfidf_doc
from src.blocking import BlockingEngine
from src.features import extract_pair_features_fast
from src.matching import MatchingModel, compute_macro_f05

def evaluate(n_val: int = 5000, threshold: float = None, verbose: bool = True):
    t0 = time.time()
    project_root = os.path.abspath(os.path.join(base_dir, "..", ".."))
    train_dir = os.path.join(project_root, "dataset", "train")

    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")

    # Load threshold if not supplied
    if threshold is None:
        meta_path = os.path.join(base_dir, "models", "threshold.txt")
        if os.path.isfile(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                threshold = float(f.read().strip())
        else:
            threshold = 0.65

    # Auto-detect best model (prefer LightGBM, then XGBoost)
    lgb_path = os.path.join(base_dir, "models", "matching_lgbm.txt")
    xgb_path = os.path.join(base_dir, "models", "matching_xgb.json")
    if os.path.isfile(lgb_path):
        model_path = lgb_path
    elif os.path.isfile(xgb_path):
        model_path = xgb_path
    else:
        raise FileNotFoundError(f"No trained model found in models/. Run train_model.py first.")

    model = MatchingModel(threshold=threshold)
    model.load(model_path)
    print(f"Loaded {model.backend.upper()} model from {model_path}")

    # Sample a held-out validation set (skip first 80% of GT for safety)
    val_s1_ids = []
    gt_map = defaultdict(set)
    skip_training = int(2206821 * 0.80)

    with open(gt_path, 'r', encoding='utf-8') as f:
        next(f)
        for i, line in enumerate(f):
            if i < skip_training:
                continue
            p = line.rstrip('\n').split('\t')
            s1 = p[0]
            if len(p) >= 2 and p[1].strip():
                gt_map[s1] = set(p[1].split(','))
            else:
                gt_map[s1] = set()
            val_s1_ids.append(s1)
            if len(val_s1_ids) >= n_val:
                break

    val_s1_set = set(val_s1_ids)
    all_needed_gt = set()
    for s in val_s1_ids:
        all_needed_gt.update(gt_map[s])

    # Load S1 validation records
    val_s1_records = []
    with open(s1_path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            if p[0] in val_s1_set:
                val_s1_records.append((p[0], p[1], p[2], p[3]))
                if len(val_s1_records) == len(val_s1_set):
                    break

    # Load Candidate Pool
    cand_records = {}
    cand_ids = []
    max_bg = 60000

    for s_path in [s2_path, s3_path]:
        with open(s_path, 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                p = line.rstrip('\n').split('\t')
                cid, name, addr = p[0], p[1], p[2]
                if cid in all_needed_gt or len(cand_records) < max_bg:
                    cn = clean_text(name)
                    ca = clean_text(addr)
                    cand_records[cid] = (cn, ca)
                    cand_ids.append(cid)

    # Blocking
    blocking = BlockingEngine(top_k=15, min_score=0.05)
    def doc_gen():
        for cid in cand_ids:
            cn, ca = cand_records[cid]
            yield make_tfidf_doc(cn, ca)
    blocking.fit_candidates(cand_ids, doc_gen())

    # Pre-clean S1
    s1_precleaned = {}
    s1_ids = []
    s1_docs = []
    for s1_id, r_name, r_addr, _ in val_s1_records:
        cn = clean_text(r_name)
        ca = clean_text(r_addr)
        cp = cn.replace(" ", "")
        nums = extract_numbers(cn) | extract_numbers(ca)
        s1_precleaned[s1_id] = (cn, ca, cp, nums)
        s1_ids.append(s1_id)
        s1_docs.append(make_tfidf_doc(cn, ca))

    batch_candidates = blocking.query_batch(s1_ids, s1_docs=s1_docs)

    # Feature extraction and scoring
    all_pairs = []
    feature_rows = []
    retrieved_counts = []
    blocking_hits = 0
    total_true_gt = len(all_needed_gt)

    for s1_id in s1_ids:
        cands = batch_candidates.get(s1_id, [])
        retrieved_counts.append(len(cands))
        true_set = gt_map[s1_id]
        retrieved_cids = {c[0] for c in cands}
        blocking_hits += len(true_set & retrieved_cids)

        s1_item = s1_precleaned[s1_id]
        for cid, score, rank in cands:
            if cid in cand_records:
                c_item = cand_records[cid]
                feat = extract_pair_features_fast(s1_item, c_item, cid, score, rank)
                feature_rows.append(feat)
                all_pairs.append((s1_id, cid))

    pred_map = defaultdict(set)
    if feature_rows:
        X_mat = np.array(feature_rows, dtype=np.float32)
        probs = model.predict_proba(X_mat)
        for (s1_id, cid), prob in zip(all_pairs, probs):
            if prob >= threshold:
                pred_map[s1_id].add(cid)

    # Compute Metrics
    val_gt_dict = {s: gt_map[s] for s in s1_ids}
    macro_f05 = compute_macro_f05(val_gt_dict, pred_map)

    precisions, recalls = [], []
    singleton_total, singleton_correct = 0, 0
    total_tp, total_pred = 0, 0
    total_gold = sum(len(s) for s in val_gt_dict.values())

    for s1_id in s1_ids:
        true_set = val_gt_dict[s1_id]
        pred_set = pred_map[s1_id]
        if not true_set:
            singleton_total += 1
            if not pred_set:
                singleton_correct += 1
        else:
            tp = len(true_set & pred_set)
            total_tp += tp
            total_pred += len(pred_set)
            precisions.append(tp / len(pred_set) if pred_set else 0.0)
            recalls.append(tp / len(true_set))

    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    micro_precision = total_tp / total_pred if total_pred > 0 else 0.0
    micro_recall = total_tp / total_gold if total_gold > 0 else 0.0
    singleton_acc = singleton_correct / singleton_total if singleton_total > 0 else 1.0
    blocking_recall = blocking_hits / total_true_gt if total_true_gt > 0 else 0.0
    mean_cand_size = float(np.mean(retrieved_counts)) if retrieved_counts else 0.0
    reduction_ratio = 100.0 * (1.0 - (mean_cand_size / max(len(cand_ids), 1)))

    results = {
        "macro_f05": round(macro_f05, 6),
        "macro_precision": round(macro_precision * 100, 2),
        "macro_recall": round(macro_recall * 100, 2),
        "micro_precision": round(micro_precision * 100, 2),
        "micro_recall": round(micro_recall * 100, 2),
        "singleton_accuracy": round(singleton_acc * 100, 2),
        "blocking_recall_ceiling": round(blocking_recall * 100, 2),
        "mean_candidate_size": round(mean_cand_size, 2),
        "reduction_ratio": round(reduction_ratio, 5),
        "threshold": threshold,
        "n_evaluated_entities": len(s1_ids),
        "elapsed_seconds": round(time.time() - t0, 2)
    }

    try:
        if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
            sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    if verbose:
        print("=" * 80)
        print("[BENCHMARK RESULTS] BUSINESS ENTITY RESOLUTION")
        print("=" * 80)
        print(f"Model Backend:             {model.backend.upper()}")
        print(f"Evaluated Source 1 Entities: {results['n_evaluated_entities']:,}")
        print(f"Calibrated Decision Threshold: {results['threshold']:.3f}")
        print("-" * 80)
        print(f"Macro F_0.5 Score:         {results['macro_f05']:.6f}  (Leaderboard Target)")
        print(f"Macro Precision:           {results['macro_precision']:.2f}%")
        print(f"Macro Recall:              {results['macro_recall']:.2f}%")
        print(f"Micro Precision:           {results['micro_precision']:.2f}%")
        print(f"Micro Recall:              {results['micro_recall']:.2f}%")
        print(f"Singleton Accuracy:        {results['singleton_accuracy']:.2f}%  (Zero false merges)")
        print(f"Blocking Recall Ceiling:   {results['blocking_recall_ceiling']:.2f}%")
        print(f"Mean Candidates per S1:    {results['mean_candidate_size']:.2f} / 15 max")
        print(f"Reduction Ratio:           {results['reduction_ratio']:.4f}%")
        print(f"Evaluation Time:           {results['elapsed_seconds']:.2f}s")
        print("=" * 80)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Entity Resolution Model Scores")
    parser.add_argument("--n-val", type=int, default=5000, help="Number of held-out validation entities")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold")
    args = parser.parse_args()
    evaluate(n_val=args.n_val, threshold=args.threshold)
