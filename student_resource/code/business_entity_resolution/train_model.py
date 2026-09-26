"""
Full-Data Business Entity Resolution Training Pipeline.
==========================================================
Trains a LightGBM + XGBoost ensemble on ALL 2.2M ground-truth entities for
maximum Macro F_0.5 on the Amazon ML Challenge 2026.

Key Design Decisions:
  - LightGBM (GBDT) is the primary model: best-in-class for 28-feature tabular
    binary classification. XGBoost GPU Hist is optionally trained as a secondary
    model for ensemble/fallback.
  - Trains on the FULL dataset (2.2M S1 entities, ~7.6M positive pairs) with
    a configurable stratified 90/10 train/val split.
  - Aggressive blocking (top_k=15, min_score=0.05) to maximize recall ceiling.
  - Hard negative mining: all blocked non-matches + all missed GT positives.
  - Fine-grained threshold sweep (0.001 steps) optimized for F_0.5.
  - Per-round verbose metrics: train/val loss, AUC, F_0.5, Precision, Recall,
    Singleton Accuracy — all printed to terminal after each boosting milestone.
  - Memory-efficient streaming architecture safe for 16 GB RAM machines.

Usage:
  python train_model.py                          # Full train (all 2.2M entities)
  python train_model.py --sample-frac 0.10       # Quick 10% sample run
  python train_model.py --use-xgb-gpu            # Also train XGBoost on GPU
  python train_model.py --help                   # All options
"""

import os
import sys
import time
import argparse
import random
import gc
import json
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import pickle

# Terminal encoding safety
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)

import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from src.preprocess import clean_text, extract_numbers, make_tfidf_doc
from src.blocking import BlockingEngine
from src.features import extract_pair_features_fast, FEATURE_NAMES
from src.matching import compute_macro_f05


def pbar(iterable=None, total=None, desc="", unit="it"):
    """Create a robust tqdm progress bar (Windows-safe)."""
    return tqdm(
        iterable=iterable, total=total, desc=desc, unit=unit,
        ncols=100, ascii=True, file=sys.stdout, mininterval=0.3,
        dynamic_ncols=False
    )


def compute_full_metrics(val_s1_ids, gt_map, val_pair_info, val_probs, threshold):
    """Compute Macro F_0.5, Precision, Recall, Singleton Accuracy at a threshold."""
    pred_dict = defaultdict(set)
    for (s1_id, cid), prob in zip(val_pair_info, val_probs):
        if prob >= threshold:
            pred_dict[s1_id].add(cid)

    scores, precisions, recalls = [], [], []
    singleton_total, singleton_correct = 0, 0

    for s1_id in val_s1_ids:
        true_set = gt_map.get(s1_id, set())
        pred_set = pred_dict.get(s1_id, set())

        if not true_set:
            singleton_total += 1
            if not pred_set:
                scores.append(1.0)
                singleton_correct += 1
            else:
                scores.append(0.0)
        else:
            if not pred_set:
                scores.append(0.0)
                precisions.append(0.0)
                recalls.append(0.0)
            else:
                tp = len(true_set & pred_set)
                if tp == 0:
                    scores.append(0.0)
                    precisions.append(0.0)
                    recalls.append(0.0)
                else:
                    prec = tp / len(pred_set)
                    rec = tp / len(true_set)
                    f05 = (1.25 * prec * rec) / (0.25 * prec + rec)
                    scores.append(f05)
                    precisions.append(prec)
                    recalls.append(rec)

    macro_f05 = float(np.mean(scores)) if scores else 0.0
    macro_prec = float(np.mean(precisions)) if precisions else 0.0
    macro_rec = float(np.mean(recalls)) if recalls else 0.0
    sing_acc = singleton_correct / singleton_total if singleton_total > 0 else 1.0

    return macro_f05, macro_prec, macro_rec, sing_acc


def main():
    parser = argparse.ArgumentParser(
        description="Full-Data LightGBM Training for Business Entity Resolution"
    )
    parser.add_argument("--sample-frac", type=float, default=1.0,
                        help="Fraction of S1 entities to use (1.0=all, 0.1=10%% sample)")
    parser.add_argument("--sample-entities", type=int, default=0,
                        help="Exact number of entities to sample (0 = use --sample-frac, e.g. 35000 for high-accuracy rapid training)")
    parser.add_argument("--val-frac", type=float, default=0.05,
                        help="Fraction of entities for validation (default: 5%%)")
    parser.add_argument("--top-k", type=int, default=40,
                        help="Blocking top-K candidates per query (default: 40)")
    parser.add_argument("--min-score", type=float, default=0.02,
                        help="Minimum blocking cosine score (default: 0.02)")
    parser.add_argument("--max-cands", type=int, default=500000,
                        help="Max background candidate records to load (default: 500K, GT always loaded)")
    parser.add_argument("--num-rounds", type=int, default=2000,
                        help="Boosting rounds (default: 2000)")
    parser.add_argument("--early-stop", type=int, default=50,
                        help="Early stopping patience (default: 50)")
    parser.add_argument("--cpu-only", action="store_true", default=False,
                        help="Use LightGBM on CPU instead of XGBoost GPU (default: GPU)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=50000,
                        help="Batch size for feature extraction streaming (default: 50k)")
    parser.add_argument("--clear-cache", action="store_true", default=False,
                        help="Clear the feature chunks cache")
    parser.add_argument("--tune", action="store_true", default=False,
                        help="Run Optuna Bayesian hyperparameter optimization before training")
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Number of Optuna tuning trials if --tune is set (default: 30)")
    parser.add_argument("--ignore-tuned", action="store_true", default=False,
                        help="Ignore saved tuned parameters in models/best_xgb_params.json")
    args = parser.parse_args()

    print("=" * 90)
    print("  FULL-DATA BUSINESS ENTITY RESOLUTION TRAINING PIPELINE")
    backend = "LightGBM CPU" if args.cpu_only else "XGBoost GPU (RTX 4050)"
    print(f"  Target: Macro F_0.5 >= 0.999 | {backend} | 28-Feature Pairwise Classifier")
    print(f"  Max candidates: {args.max_cands:,} (GT always included) | RAM-safe mode")
    print("=" * 90)
    t_global = time.time()

    project_root = os.path.abspath(os.path.join(base_dir, "..", ".."))
    train_dir = os.path.join(project_root, "dataset", "train")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")

    # Clean feature chunk cache if requested
    if args.clear_cache:
        chunk_dir = os.path.join(base_dir, "models", "feature_chunks")
        if os.path.exists(chunk_dir):
            for root, dirs, files in os.walk(chunk_dir, topdown=False):
                for f in files:
                    try:
                        os.remove(os.path.join(root, f))
                    except Exception:
                        pass
                for d in dirs:
                    try:
                        os.rmdir(os.path.join(root, d))
                    except Exception:
                        pass
            print("  [!] Cleared feature chunks cache.", flush=True)

    legacy_cache = os.path.join(base_dir, "models", "data_cache.pkl")
    if os.path.exists(legacy_cache):
        try:
            os.remove(legacy_cache)
        except Exception:
            pass

    # =========================================================================
    # STAGE 1: Load Ground Truth
    # =========================================================================
    print("\n[1/7] Loading Ground Truth (2.2M entities)...", flush=True)
    t0 = time.time()
    gt_map = {}
    all_gt_ids = []

    with open(gt_path, 'r', encoding='utf-8') as f:
        next(f)
        pb = pbar(f, desc="  Ground Truth", unit=" ent")
        for line in pb:
            p = line.rstrip('\n').split('\t')
            s1 = p[0]
            if len(p) >= 2 and p[1].strip():
                gt_map[s1] = set(p[1].split(','))
            else:
                gt_map[s1] = set()
            all_gt_ids.append(s1)
        pb.close()

    total_entities = len(all_gt_ids)
    total_singletons = sum(1 for s in all_gt_ids if not gt_map[s])
    total_matched = total_entities - total_singletons
    total_pairs = sum(len(v) for v in gt_map.values())

    print(f"  -> Total S1 entities:     {total_entities:,}")
    print(f"  -> Singletons:            {total_singletons:,} ({total_singletons/total_entities*100:.1f}%)")
    print(f"  -> Matched entities:      {total_matched:,} ({total_matched/total_entities*100:.1f}%)")
    print(f"  -> Total positive pairs:  {total_pairs:,}")
    print(f"  -> Loaded in {time.time()-t0:.1f}s", flush=True)

    # Stratified shuffle split
    random.seed(args.seed)
    shuffled = list(all_gt_ids)
    random.shuffle(shuffled)

    if args.sample_entities > 0:
        n_use = min(args.sample_entities, total_entities)
        shuffled = shuffled[:n_use]
        print(f"  -> Sampling exact count: {n_use:,} entities ({n_use/total_entities*100:.2f}%)")
    elif args.sample_frac < 1.0:
        n_use = int(total_entities * args.sample_frac)
        shuffled = shuffled[:n_use]
        print(f"  -> Sampling {args.sample_frac*100:.1f}%: {n_use:,} entities")

    n_val = int(len(shuffled) * args.val_frac)
    val_s1_ids = shuffled[:n_val]
    train_s1_ids = shuffled[n_val:]

    val_s1_set = set(val_s1_ids)
    train_s1_set = set(train_s1_ids)
    all_active = val_s1_set | train_s1_set

    # Count singletons in each split
    train_singletons = sum(1 for s in train_s1_ids if not gt_map[s])
    val_singletons = sum(1 for s in val_s1_ids if not gt_map[s])

    print(f"\n  Split: {len(train_s1_ids):,} train ({train_singletons:,} singletons) | "
          f"{len(val_s1_ids):,} val ({val_singletons:,} singletons)", flush=True)

    # Collect all needed candidate IDs
    needed_cand_ids = set()
    for s in all_active:
        needed_cand_ids.update(gt_map.get(s, set()))
    print(f"  -> Required GT candidate records: {len(needed_cand_ids):,}", flush=True)

    # =========================================================================
    # STAGE 2: Load & Pre-clean Source 1 Records
    # =========================================================================
    print(f"\n[2/7] Loading Source 1 records ({len(all_active):,} entities)...", flush=True)
    t0 = time.time()
    s1_precleaned = {}  # s1_id -> (cn, ca, cp, nums)
    s1_docs = {}        # s1_id -> tfidf_doc_string

    with open(s1_path, 'r', encoding='utf-8') as f:
        next(f)
        pb = pbar(f, desc="  Source 1", unit=" rec")
        for line in pb:
            p = line.rstrip('\n').split('\t')
            s1_id = p[0]
            if s1_id in all_active:
                cn = clean_text(p[1])
                ca = clean_text(p[2])
                cp = cn.replace(" ", "")
                nums = extract_numbers(cn) | extract_numbers(ca)
                s1_precleaned[s1_id] = (cn, ca, cp, nums)
                s1_docs[s1_id] = make_tfidf_doc(cn, ca)
                if len(s1_precleaned) >= len(all_active):
                    pb.update(1)
                    break
        pb.close()

    print(f"  -> Pre-cleaned {len(s1_precleaned):,} S1 records in {time.time()-t0:.1f}s", flush=True)

    # =========================================================================
    # STAGE 3: Load Candidate Pool (Source 2 + Source 3)
    # =========================================================================
    print(f"\n[3/7] Loading Candidate Pool (Source 2 + Source 3)...", flush=True)
    t0 = time.time()
    cand_records = {}  # cid -> (cn, ca)
    cand_ids = []
    exact_name_index = defaultdict(list)
    gt_coverage = 0

    for s_path in [s2_path, s3_path]:
        src_name = os.path.basename(s_path)
        with open(s_path, 'r', encoding='utf-8') as f:
            next(f)
            pb = pbar(f, desc=f"  {src_name}", unit=" rec")
            for line in pb:
                p = line.rstrip('\n').split('\t')
                cid = p[0]
                is_needed = cid in needed_cand_ids
                can_bg = (args.max_cands == 0 or len(cand_records) < args.max_cands)

                if is_needed or can_bg:
                    cn = clean_text(p[1])
                    ca = clean_text(p[2])
                    cand_records[cid] = (cn, ca)
                    cand_ids.append(cid)
                    if is_needed:
                        gt_coverage += 1
                    if cn:
                        exact_name_index[cn].append(cid)
            pb.close()

    n_cands = len(cand_records)
    print(f"  -> Candidate pool: {n_cands:,} records ({gt_coverage:,}/{len(needed_cand_ids):,} GT covered)")
    print(f"  -> Loaded in {time.time()-t0:.1f}s", flush=True)

    # =========================================================================
    # STAGE 4: Build Blocking Index
    # =========================================================================
    print(f"\n[4/7] Building TF-IDF Blocking Index (top_k={args.top_k}, min_score={args.min_score})...", flush=True)
    t0 = time.time()
    blocking = BlockingEngine(top_k=args.top_k, min_score=args.min_score)

    def cand_doc_gen():
        pb_inner = pbar(total=len(cand_ids), desc="  Vectorizing Candidates", unit=" doc")
        for cid in cand_ids:
            cn, ca = cand_records[cid]
            pb_inner.update(1)
            yield make_tfidf_doc(cn, ca)
        pb_inner.close()

    blocking.fit_candidates(cand_ids, cand_doc_gen())
    print("  Building auxiliary multi-channel indices (exact, compact, rare-token, address)...", flush=True)
    blocking.build_auxiliary_indices(cand_records)
    print(f"  -> Blocking engine ready in {time.time()-t0:.1f}s (RAM-safe in-memory mode, no crash-prone pickle dumping)", flush=True)

    # =========================================================================
    # STAGE 5: Extract Feature Matrices (Streaming)
    # =========================================================================
    print(f"\n[5/7] Extracting 28-Feature Pairwise Matrices (with Hard Negative Mining)...", flush=True)
    t0 = time.time()

    def build_feature_matrix(entity_ids, is_training, desc_label):
        """Build feature matrix with blocking + exact name match retrieval + hard negatives."""
        chunk_X_list = []
        chunk_y_list = []
        pair_info = []
        recall_hits, recall_total = 0, 0
        batch_size_query = 2500

        n_total = len(entity_ids)
        pb_outer = pbar(total=n_total, desc=f"  {desc_label}", unit=" ent")
        
        cache_dir = os.path.join(base_dir, "models", "feature_chunks", desc_label.replace(" ", "_"))
        os.makedirs(cache_dir, exist_ok=True)

        for b_start in range(0, n_total, batch_size_query):
            b_end = min(b_start + batch_size_query, n_total)
            chunk_file = os.path.join(cache_dir, f"chunk_{b_start}_{b_end}.pkl")
            
            if os.path.exists(chunk_file):
                try:
                    with open(chunk_file, 'rb') as f:
                        c_data = pickle.load(f)
                    chunk_X_list.append(np.asarray(c_data['X'], dtype=np.float32))
                    chunk_y_list.append(np.asarray(c_data['y'], dtype=np.int32))
                    if not is_training:
                        pair_info.extend(c_data['pair_info'])
                    recall_hits += c_data['recall_hits']
                    recall_total += c_data['recall_total']
                    pb_outer.update(b_end - b_start)
                    continue
                except Exception as e:
                    print(f"\n  [WARNING] Failed to load chunk cache {chunk_file}: {e}")

            cX, cy, c_pair_info = [], [], []
            c_hits, c_total = 0, 0

            sub_ids = entity_ids[b_start:b_end]
            sub_docs = [s1_docs[s] for s in sub_ids if s in s1_docs]
            valid_ids = [s for s in sub_ids if s in s1_docs]

            # Blocking query with multi-channel auxiliary index retrieval
            sub_cands = blocking.query_batch(valid_ids, s1_docs=sub_docs, s1_precleaned=s1_precleaned, batch_size=batch_size_query)

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
                        c_item = cand_records[cid]
                        feat = extract_pair_features_fast(s1_item, c_item, cid, score, rank)
                        cX.append(feat)
                        cy.append(label)
                        if not is_training:
                            c_pair_info.append((s1_id, cid))

                # Hard negative mining: inject missed GT positives (training only)
                if is_training:
                    for true_id in true_set:
                        if true_id not in retrieved_ids and true_id in cand_records:
                            c_item = cand_records[true_id]
                            feat = extract_pair_features_fast(s1_item, c_item, true_id, 0.0, 99)
                            cX.append(feat)
                            cy.append(1)

                if true_set:
                    c_total += len(true_set)
                    c_hits += len(true_set & retrieved_ids)
            
            arr_X = np.asarray(cX, dtype=np.float32) if cX else np.empty((0, 28), dtype=np.float32)
            arr_y = np.asarray(cy, dtype=np.int32) if cy else np.empty(0, dtype=np.int32)
            
            # Save chunk to disk as compact numpy arrays
            with open(chunk_file, 'wb') as f:
                pickle.dump({
                    'X': arr_X,
                    'y': arr_y,
                    'pair_info': c_pair_info,
                    'recall_hits': c_hits,
                    'recall_total': c_total
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            chunk_X_list.append(arr_X)
            chunk_y_list.append(arr_y)
            if not is_training:
                pair_info.extend(c_pair_info)
            recall_hits += c_hits
            recall_total += c_total

            pb_outer.update(b_end - b_start)

        pb_outer.close()
        X = np.vstack(chunk_X_list) if chunk_X_list else np.empty((0, 28), dtype=np.float32)
        y = np.concatenate(chunk_y_list) if chunk_y_list else np.empty(0, dtype=np.int32)
        del chunk_X_list, chunk_y_list
        gc.collect()
        recall = recall_hits / recall_total if recall_total > 0 else 0.0
        return X, y, pair_info, recall

    X_train, y_train, _, train_recall = build_feature_matrix(train_s1_ids, True, "Train Features")
    X_val, y_val, val_pair_info, val_recall = build_feature_matrix(val_s1_ids, False, "Val Features")

    pos_train = int(np.sum(y_train == 1))
    neg_train = int(np.sum(y_train == 0))
    pos_val = int(np.sum(y_val == 1))
    neg_val = int(np.sum(y_val == 0))

    print(f"\n  Train Matrix: {len(X_train):,} pairs ({pos_train:,} pos / {neg_train:,} neg) | Blocking Recall: {train_recall*100:.2f}%")
    print(f"  Val Matrix:   {len(X_val):,} pairs ({pos_val:,} pos / {neg_val:,} neg) | Blocking Recall: {val_recall*100:.2f}%")
    print(f"  Features: {X_train.shape[1]} dimensions | Extracted in {time.time()-t0:.1f}s", flush=True)

    # Free memory before training
    del s1_precleaned, s1_docs, cand_records, cand_ids, exact_name_index, blocking
    gc.collect()

    # =========================================================================
    # STAGE 6: Train Model (XGBoost GPU default, LightGBM CPU fallback)
    # =========================================================================
    scale_pos = neg_train / pos_train if pos_train > 0 else 1.0
    use_gpu = not args.cpu_only

    if use_gpu:
        print(f"\n{'='*90}")
        print(f"[6/7] Training XGBoost on GPU (device=cuda, {args.num_rounds} rounds, early_stop={args.early_stop})")
        print(f"{'='*90}", flush=True)
        t0 = time.time()

        try:
            import xgboost as xgb
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_NAMES)
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_NAMES)

            xgb_params = {
                'device': 'cuda',
                'tree_method': 'hist',
                'objective': 'binary:logistic',
                'eval_metric': ['logloss', 'auc'],
                'learning_rate': 0.05,
                'max_depth': 8,
                'min_child_weight': 20,
                'subsample': 0.80,
                'colsample_bytree': 0.80,
                'scale_pos_weight': scale_pos,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'seed': args.seed,
                'nthread': -1,
            }

            tuned_params_file = os.path.join(base_dir, "models", "best_xgb_params.json")
            if os.path.exists(tuned_params_file) and not args.ignore_tuned:
                try:
                    with open(tuned_params_file, 'r', encoding='utf-8') as f:
                        tuned_cfg = json.load(f)
                    print(f"  [+] Loaded tuned hyperparameters from {tuned_params_file}")
                    for k, v in tuned_cfg.items():
                        if k in ['max_depth', 'min_child_weight']:
                            xgb_params[k] = int(v)
                        elif k in ['learning_rate', 'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda', 'gamma']:
                            xgb_params[k] = float(v)
                        elif k == 'scale_pos_mult':
                            xgb_params['scale_pos_weight'] = scale_pos * float(v)
                except Exception as e:
                    print(f"  [!] Failed to load tuned params from {tuned_params_file}: {e}")

            if args.tune:
                print(f"\n{'='*90}")
                print(f"[TUNING] Running Optuna Bayesian Hyperparameter Optimization ({args.n_trials} trials)...")
                print(f"{'='*90}", flush=True)
                import optuna
                optuna.logging.set_verbosity(optuna.logging.WARNING)

                def objective(trial):
                    trial_params = dict(xgb_params)
                    trial_params['max_depth'] = trial.suggest_int('max_depth', 6, 12)
                    trial_params['learning_rate'] = trial.suggest_float('learning_rate', 0.02, 0.15, log=True)
                    trial_params['subsample'] = trial.suggest_float('subsample', 0.65, 0.95)
                    trial_params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.60, 0.95)
                    trial_params['min_child_weight'] = trial.suggest_int('min_child_weight', 5, 50)
                    trial_params['reg_lambda'] = trial.suggest_float('reg_lambda', 0.1, 10.0, log=True)
                    trial_params['reg_alpha'] = trial.suggest_float('reg_alpha', 1e-3, 5.0, log=True)
                    trial_params['scale_pos_weight'] = scale_pos * trial.suggest_float('scale_pos_mult', 0.6, 2.0)

                    m = xgb.train(
                        trial_params, dtrain,
                        num_boost_round=min(800, args.num_rounds),
                        evals=[(dval, 'val')],
                        early_stopping_rounds=30,
                        verbose_eval=False,
                    )
                    preds = m.predict(dval, iteration_range=(0, m.best_iteration + 1))
                    best_trial_f05 = 0.0
                    for th in [0.75, 0.80, 0.85, 0.88, 0.90, 0.92]:
                        f05, _, _, _ = compute_full_metrics(val_s1_ids, gt_map, val_pair_info, preds, th)
                        if f05 > best_trial_f05:
                            best_trial_f05 = f05
                    del m
                    gc.collect()
                    return best_trial_f05

                study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed))
                pb_tune = pbar(total=args.n_trials, desc="  Optuna Tuning", unit=" trial")
                def callback(s, t):
                    pb_tune.update(1)
                study.optimize(objective, n_trials=args.n_trials, callbacks=[callback])
                pb_tune.close()

                print(f"\n  [+] Best Optuna Macro F_0.5: {study.best_value:.6f}")
                print(f"  [+] Best Parameters: {study.best_params}", flush=True)

                os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)
                with open(tuned_params_file, "w", encoding="utf-8") as f:
                    json.dump(study.best_params, f, indent=2)

                for k, v in study.best_params.items():
                    if k in ['max_depth', 'min_child_weight']:
                        xgb_params[k] = int(v)
                    elif k in ['learning_rate', 'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda']:
                        xgb_params[k] = float(v)
                    elif k == 'scale_pos_mult':
                        xgb_params['scale_pos_weight'] = scale_pos * float(v)

            print(f"  XGBoost params: depth={xgb_params['max_depth']}, lr={xgb_params['learning_rate']}, "
                  f"subsample={xgb_params['subsample']}, scale_pos={scale_pos:.2f}", flush=True)

            xgb_model = xgb.train(
                xgb_params, dtrain,
                num_boost_round=args.num_rounds,
                evals=[(dtrain, 'train'), (dval, 'val')],
                early_stopping_rounds=args.early_stop,
                verbose_eval=25,
            )

            best_iteration = xgb_model.best_iteration
            trained_model = xgb_model
            model_backend = "xgboost"
            print(f"\n  -> XGBoost GPU training completed in {time.time()-t0:.1f}s")
            print(f"  -> Best iteration: {best_iteration}", flush=True)

            # Feature importance
            importance_dict = xgb_model.get_score(importance_type='gain')
            print(f"\n  Top 10 Features by Gain:")
            sorted_feats = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
            for i, (feat, gain) in enumerate(sorted_feats):
                print(f"    {i+1:>2}. {feat:<30s} gain={gain:,.0f}")
            print(flush=True)

        except Exception as e:
            print(f"\n  [WARNING] XGBoost GPU failed: {e}")
            print(f"  Falling back to LightGBM CPU...", flush=True)
            use_gpu = False

    if not use_gpu:
        print(f"\n{'='*90}")
        print(f"[6/7] Training LightGBM GBDT on CPU ({args.num_rounds} rounds, early_stop={args.early_stop})")
        print(f"{'='*90}", flush=True)
        t0 = time.time()

        lgb_params = {
            'objective': 'binary',
            'metric': ['binary_logloss', 'auc'],
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 127,
            'max_depth': 8,
            'min_child_samples': 50,
            'feature_fraction': 0.80,
            'bagging_fraction': 0.80,
            'bagging_freq': 1,
            'scale_pos_weight': scale_pos,
            'lambda_l1': 0.1,
            'lambda_l2': 1.0,
            'verbose': -1,
            'n_jobs': -1,
            'seed': args.seed,
            'force_col_wise': True,
        }

        train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES, free_raw_data=False)
        val_data = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_NAMES, reference=train_data, free_raw_data=False)

        class MetricsCallback:
            def __init__(self, print_every=50):
                self.print_every = print_every
                self.best_val_loss = float('inf')
                self.best_round = 0
                self.header_printed = False

            def __call__(self, env):
                iteration = env.iteration + 1
                if iteration == 1 or iteration % self.print_every == 0 or iteration == env.end_iteration:
                    train_loss = env.evaluation_result_list[0][2]
                    val_loss = env.evaluation_result_list[2][2]
                    train_auc = env.evaluation_result_list[1][2]
                    val_auc = env.evaluation_result_list[3][2]

                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.best_round = iteration
                        marker = " *"
                    else:
                        marker = ""

                    if not self.header_printed:
                        print(f"\n  {'Round':>6} | {'Train Loss':>11} | {'Val Loss':>11} | {'Train AUC':>10} | {'Val AUC':>10} | {'Best':>5}")
                        print(f"  {'-'*6}-+-{'-'*11}-+-{'-'*11}-+-{'-'*10}-+-{'-'*10}-+-{'-'*5}")
                        self.header_printed = True

                    print(f"  {iteration:>6} | {train_loss:>11.6f} | {val_loss:>11.6f} | {train_auc:>10.6f} | {val_auc:>10.6f} |{marker}", flush=True)

        metrics_cb = MetricsCallback(print_every=50)
        callbacks = [
            lgb.early_stopping(stopping_rounds=args.early_stop, verbose=False),
            metrics_cb,
        ]

        lgb_model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=args.num_rounds,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=callbacks,
        )

        best_iteration = lgb_model.best_iteration
        trained_model = lgb_model
        model_backend = "lightgbm"
        print(f"\n  -> LightGBM training completed in {time.time()-t0:.1f}s")
        print(f"  -> Best iteration: {best_iteration}", flush=True)

        importance = lgb_model.feature_importance(importance_type='gain')
        sorted_idx = np.argsort(importance)[::-1]
        print(f"\n  Top 10 Features by Gain:")
        for i, idx in enumerate(sorted_idx[:10]):
            print(f"    {i+1:>2}. {FEATURE_NAMES[idx]:<30s} gain={importance[idx]:,.0f}")
        print(flush=True)

    # =========================================================================
    # STAGE 7: Threshold Optimization for Macro F_0.5
    # =========================================================================
    print(f"{'='*90}")
    print(f"[7/7] Threshold Optimization for Macro F_0.5")
    print(f"{'='*90}", flush=True)
    t0 = time.time()

    if model_backend == "xgboost":
        import xgboost as xgb
        dval_pred = xgb.DMatrix(X_val, feature_names=FEATURE_NAMES)
        val_probs = trained_model.predict(dval_pred, iteration_range=(0, best_iteration + 1))
    else:
        val_probs = trained_model.predict(X_val, num_iteration=best_iteration)

    val_auc = roc_auc_score(y_val, val_probs)
    print(f"  -> Validation ROC AUC: {val_auc:.6f}")
    print(f"  -> Val entities: {len(val_s1_ids):,} ({val_singletons:,} singletons)", flush=True)

    # Coarse sweep first
    print(f"\n  Phase 1: Coarse Sweep (0.30 to 0.95, step=0.05)")
    print(f"  {'Threshold':>10} | {'Macro F0.5':>11} | {'Precision':>10} | {'Recall':>10} | {'Sing. Acc':>10}")
    print(f"  {'-'*10}-+-{'-'*11}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    best_th_coarse = 0.50
    best_f05_coarse = 0.0

    for th in np.arange(0.30, 0.96, 0.05):
        th = round(float(th), 2)
        f05, prec, rec, sing = compute_full_metrics(val_s1_ids, gt_map, val_pair_info, val_probs, th)
        marker = " *" if f05 > best_f05_coarse else ""
        if f05 > best_f05_coarse:
            best_f05_coarse = f05
            best_th_coarse = th
        print(f"  {th:>10.2f} | {f05:>11.6f} | {prec*100:>9.2f}% | {rec*100:>9.2f}% | {sing*100:>9.2f}%{marker}", flush=True)

    # Fine sweep around best coarse threshold
    fine_lo = max(0.10, best_th_coarse - 0.10)
    fine_hi = min(0.99, best_th_coarse + 0.10)
    print(f"\n  Phase 2: Fine Sweep ({fine_lo:.2f} to {fine_hi:.2f}, step=0.005)")
    print(f"  {'Threshold':>10} | {'Macro F0.5':>11} | {'Precision':>10} | {'Recall':>10} | {'Sing. Acc':>10}")
    print(f"  {'-'*10}-+-{'-'*11}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    best_th = best_th_coarse
    best_f05 = best_f05_coarse
    best_prec = 0.0
    best_rec = 0.0
    best_sing = 0.0

    for th in np.arange(fine_lo, fine_hi + 0.001, 0.005):
        th = round(float(th), 3)
        f05, prec, rec, sing = compute_full_metrics(val_s1_ids, gt_map, val_pair_info, val_probs, th)
        marker = " *" if f05 > best_f05 else ""
        if f05 > best_f05:
            best_f05 = f05
            best_th = th
            best_prec = prec
            best_rec = rec
            best_sing = sing
        print(f"  {th:>10.3f} | {f05:>11.6f} | {prec*100:>9.2f}% | {rec*100:>9.2f}% | {sing*100:>9.2f}%{marker}", flush=True)

    # Ultra-fine sweep
    ultra_lo = max(0.05, best_th - 0.02)
    ultra_hi = min(0.99, best_th + 0.02)
    print(f"\n  Phase 3: Ultra-Fine Sweep ({ultra_lo:.3f} to {ultra_hi:.3f}, step=0.001)")

    for th in np.arange(ultra_lo, ultra_hi + 0.0001, 0.001):
        th = round(float(th), 3)
        f05, prec, rec, sing = compute_full_metrics(val_s1_ids, gt_map, val_pair_info, val_probs, th)
        if f05 > best_f05:
            best_f05 = f05
            best_th = th
            best_prec = prec
            best_rec = rec
            best_sing = sing

    print(f"\n  {'='*70}")
    print(f"  OPTIMAL DECISION BOUNDARY")
    print(f"  {'='*70}")
    print(f"  Backend:            {model_backend.upper()}")
    print(f"  Threshold:          {best_th:.3f}")
    print(f"  Macro F_0.5:        {best_f05:.6f}")
    print(f"  Macro Precision:    {best_prec*100:.2f}%")
    print(f"  Macro Recall:       {best_rec*100:.2f}%")
    print(f"  Singleton Accuracy: {best_sing*100:.2f}%")
    print(f"  ROC AUC:            {val_auc:.6f}")
    print(f"  {'='*70}", flush=True)

    # =========================================================================
    # Save Model & Artifacts
    # =========================================================================
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    if model_backend == "xgboost":
        model_path = os.path.join(models_dir, "matching_xgb.json")
        trained_model.save_model(model_path)
    else:
        model_path = os.path.join(models_dir, "matching_lgbm.txt")
        trained_model.save_model(model_path, num_iteration=best_iteration)

    model_size = os.path.getsize(model_path) / (1024**2)
    print(f"\n  Saved {model_backend.upper()} model: {model_path} ({model_size:.1f} MB)")

    th_path = os.path.join(models_dir, "threshold.txt")
    with open(th_path, 'w', encoding='utf-8') as f:
        f.write(f"{best_th:.3f}\n")
    print(f"  Saved threshold: {th_path} (value={best_th:.3f})")

    # Save training metrics for documentation
    metrics_path = os.path.join(models_dir, "training_metrics.txt")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write(f"Training Configuration\n")
        f.write(f"=====================\n")
        f.write(f"Model: {model_backend.upper()}\n")
        f.write(f"Train entities: {len(train_s1_ids):,}\n")
        f.write(f"Val entities: {len(val_s1_ids):,}\n")
        f.write(f"Train pairs: {len(X_train):,} ({pos_train:,} pos / {neg_train:,} neg)\n")
        f.write(f"Val pairs: {len(X_val):,} ({pos_val:,} pos / {neg_val:,} neg)\n")
        f.write(f"Blocking: top_k={args.top_k}, min_score={args.min_score}\n")
        f.write(f"Max candidates: {args.max_cands:,}\n")
        f.write(f"Blocking recall (train): {train_recall*100:.2f}%\n")
        f.write(f"Blocking recall (val): {val_recall*100:.2f}%\n")
        f.write(f"Boosting rounds: {best_iteration}\n\n")
        f.write(f"Validation Results\n")
        f.write(f"==================\n")
        f.write(f"Threshold: {best_th:.3f}\n")
        f.write(f"Macro F_0.5: {best_f05:.6f}\n")
        f.write(f"Macro Precision: {best_prec*100:.2f}%\n")
        f.write(f"Macro Recall: {best_rec*100:.2f}%\n")
        f.write(f"Singleton Accuracy: {best_sing*100:.2f}%\n")
        f.write(f"ROC AUC: {val_auc:.6f}\n")
    print(f"  Saved training metrics: {metrics_path}")

    total_time = time.time() - t_global
    print(f"\n{'='*90}")
    print(f"  TRAINING COMPLETE | Total time: {total_time/60:.1f} minutes ({total_time:.0f}s)")
    print(f"  Backend: {model_backend.upper()} | Model: {model_path}")
    print(f"  Threshold: {best_th:.3f} | Macro F_0.5: {best_f05:.6f}")
    print(f"{'='*90}", flush=True)


if __name__ == "__main__":
    main()
