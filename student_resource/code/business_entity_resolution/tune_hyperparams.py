"""
Standalone Bayesian Hyperparameter Optimization with Optuna for Entity Matching.
Optimizes XGBoost (GPU) parameters directly against validation Macro F_0.5.
Saves best configuration to models/best_xgb_params.json.
"""

import os
import sys
import time
import argparse
import json
import pickle
import random
import numpy as np
import optuna
from collections import defaultdict
from sklearn.metrics import roc_auc_score

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)

from src.features import FEATURE_NAMES, extract_pair_features_fast
from train_model import compute_full_metrics, pbar

def main():
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Tuning for Entity Resolution")
    parser.add_argument("--n-trials", type=int, default=10, help="Number of Optuna trials (default: 10)")
    parser.add_argument("--timeout", type=int, default=None, help="Stop study after the given number of seconds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=os.path.join(base_dir, "models", "best_xgb_params.json"),
                        help="Path to save best parameters")
    args = parser.parse_args()

    cache_file = os.path.join(base_dir, "models", "data_cache.pkl")
    if not os.path.exists(cache_file):
        print(f"[ERROR] Cache file not found: {cache_file}")
        print("Please run `python train_model.py` (or with --sample-frac 0.1) once to build data cache.")
        sys.exit(1)

    print("=" * 80)
    print("  BAYESIAN HYPERPARAMETER OPTIMIZATION (OPTUNA + GPU XGBOOST)")
    print(f"  Trials: {args.n_trials} | Objective: Maximize Validation Macro F_0.5")
    print(f"  Data Source: {cache_file}")
    print("=" * 80, flush=True)

    t0 = time.time()
    with open(cache_file, "rb") as f:
        cache_data = pickle.load(f)

    gt_map = cache_data['gt_map']
    val_s1_ids = cache_data['val_s1_ids']
    train_s1_ids = cache_data['train_s1_ids']
    s1_precleaned = cache_data['s1_precleaned']
    s1_docs = cache_data['s1_docs']
    cand_records = cache_data['cand_records']
    blocking = cache_data['blocking']

    if not hasattr(blocking, 'addr_anchor_index'):
        print("  Building auxiliary multi-channel indices (exact, compact, rare-token, address)...", flush=True)
        blocking.build_auxiliary_indices(cand_records)

    print(f"  Loaded cache in {time.time()-t0:.1f}s | Train entities: {len(train_s1_ids):,} | Val entities: {len(val_s1_ids):,}", flush=True)

    # Fast feature extraction helper
    def extract_set(entity_ids, is_train, desc):
        X, y, pair_info = [], [], []
        batch_size = 2500
        n_total = len(entity_ids)
        pb = pbar(total=n_total, desc=f"  {desc}", unit=" ent")

        for b_start in range(0, n_total, batch_size):
            b_end = min(b_start + batch_size, n_total)
            sub_ids = entity_ids[b_start:b_end]
            sub_docs = [s1_docs[s] for s in sub_ids if s in s1_docs]
            valid_ids = [s for s in sub_ids if s in s1_docs]

            sub_cands = blocking.query_batch(valid_ids, s1_docs=sub_docs, s1_precleaned=s1_precleaned, batch_size=batch_size)

            for s1_id in valid_ids:
                if s1_id not in s1_precleaned:
                    continue
                true_set = gt_map.get(s1_id, set())
                cands = list(sub_cands.get(s1_id, []))
                s1_item = s1_precleaned[s1_id]

                retrieved_ids = set()
                for cid, score, rank in cands:
                    if cid in cand_records:
                        retrieved_ids.add(cid)
                        label = 1 if cid in true_set else 0
                        feat = extract_pair_features_fast(s1_item, cand_records[cid], cid, score, rank)
                        X.append(feat)
                        y.append(label)
                        if not is_train:
                            pair_info.append((s1_id, cid))

                # Inject missed positives for training
                if is_train:
                    for true_id in true_set:
                        if true_id not in retrieved_ids and true_id in cand_records:
                            feat = extract_pair_features_fast(s1_item, cand_records[true_id], true_id, 0.0, 99)
                            X.append(feat)
                            y.append(1)

            pb.update(b_end - b_start)
        pb.close()
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), pair_info

    print("\nPreparing feature matrices for tuning...", flush=True)
    X_train, y_train, _ = extract_set(train_s1_ids, True, "Train Features")
    X_val, y_val, val_pair_info = extract_set(val_s1_ids, False, "Val Features")

    pos_train = int(np.sum(y_train == 1))
    neg_train = int(np.sum(y_train == 0))
    scale_pos = neg_train / pos_train if pos_train > 0 else 1.0
    print(f"  Train: {len(X_train):,} pairs ({pos_train:,} pos / {neg_train:,} neg)")
    print(f"  Val:   {len(X_val):,} pairs ({int(np.sum(y_val==1)):,} pos / {int(np.sum(y_val==0)):,} neg)", flush=True)

    import xgboost as xgb
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_NAMES)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_NAMES)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    best_overall_f05 = 0.0
    best_overall_th = 0.85

    def objective(trial):
        nonlocal best_overall_f05, best_overall_th

        params = {
            'device': 'cuda',
            'tree_method': 'hist',
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'auc'],
            'seed': args.seed,
            'max_depth': trial.suggest_int('max_depth', 6, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.15, log=True),
            'subsample': trial.suggest_float('subsample', 0.65, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.60, 0.95),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 50),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 5.0, log=True),
            'scale_pos_weight': scale_pos * trial.suggest_float('scale_pos_mult', 0.6, 2.0),
        }

        bst = xgb.train(
            params, dtrain,
            num_boost_round=800,
            evals=[(dval, 'val')],
            early_stopping_rounds=30,
            verbose_eval=False,
        )

        preds = bst.predict(dval, iteration_range=(0, bst.best_iteration + 1))

        # Sweep thresholds
        trial_best_f05 = 0.0
        trial_best_th = 0.85
        for th in np.arange(0.70, 0.96, 0.02):
            th = round(float(th), 2)
            f05, prec, rec, sing = compute_full_metrics(val_s1_ids, gt_map, val_pair_info, preds, th)
            if f05 > trial_best_f05:
                trial_best_f05 = f05
                trial_best_th = th

        trial.set_user_attr("best_threshold", trial_best_th)
        if trial_best_f05 > best_overall_f05:
            best_overall_f05 = trial_best_f05
            best_overall_th = trial_best_th

        return trial_best_f05

    print(f"\nStarting {args.n_trials} Bayesian Optimization trials...", flush=True)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed))
    pb_tune = pbar(total=args.n_trials, desc="  Tuning Progress", unit=" trial")

    def on_step(study, trial):
        pb_tune.update(1)
        pb_tune.set_postfix({"Best F0.5": f"{study.best_value:.5f}"})

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, callbacks=[on_step])
    pb_tune.close()

    print("\n" + "=" * 80)
    print("  OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"  Best Validation Macro F_0.5: {study.best_value:.6f}")
    print(f"  Best Decision Threshold:      {study.best_trial.user_attrs.get('best_threshold', 0.88):.3f}")
    print("\n  Winning Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"    - {k:<20s}: {v}")

    # Save to JSON
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_dict = dict(study.best_params)
    output_dict['best_macro_f05'] = study.best_value
    output_dict['best_threshold'] = study.best_trial.user_attrs.get('best_threshold', 0.88)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=2)

    print(f"\n  Saved best parameters to: {args.output}")
    print("  These will be automatically loaded on next run of `python train_model.py`!")
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
