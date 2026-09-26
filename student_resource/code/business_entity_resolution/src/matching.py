"""
Machine Learning Matching Model Module.
Trains and evaluates LightGBM gradient boosted decision trees for pairwise
match classification, and optimizes the decision threshold for Macro F_0.5.

Supports:
  - LightGBM GBDT (primary, CPU — fastest for 28-feature tabular)
  - XGBoost (GPU or CPU fallback)
"""

import os
import sys
import numpy as np
import lightgbm as lgb
from .features import FEATURE_NAMES


def compute_macro_f05(gt_dict: dict, pred_dict: dict) -> float:
    """
    Compute competition Macro F_0.5 score across all Source 1 entities.
    Formula: F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    Singletons: 1.0 if correctly empty, 0.0 if any false merge.
    """
    scores = []
    for s1, true_set in gt_dict.items():
        pred_set = pred_dict.get(s1, set())
        if not true_set:
            # Singleton entity
            scores.append(1.0 if not pred_set else 0.0)
        else:
            if not pred_set:
                scores.append(0.0)
            else:
                tp = len(true_set & pred_set)
                if tp == 0:
                    scores.append(0.0)
                else:
                    prec = tp / len(pred_set)
                    rec = tp / len(true_set)
                    f05 = (1.25 * prec * rec) / (0.25 * prec + rec)
                    scores.append(f05)
    return float(np.mean(scores)) if scores else 0.0


class MatchingModel:
    """Unified matching model supporting LightGBM and XGBoost backends."""

    def __init__(self, threshold: float = 0.50, backend: str = "auto"):
        self.threshold = threshold
        self.backend = backend
        self.model = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              use_gpu: bool = False):
        """
        Train classifier with early stopping.
        Backend selection:
          - use_gpu=True → XGBoost GPU Hist
          - use_gpu=False → LightGBM GBDT (default, best for tabular)
        """
        if use_gpu:
            import xgboost as xgb
            self.backend = "xgboost"
            pos_count = int(np.sum(y_train == 1))
            neg_count = int(np.sum(y_train == 0))
            scale_pos = neg_count / pos_count if pos_count > 0 else 1.0

            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_NAMES)
            evals = [(dtrain, 'train')]
            if X_val is not None and y_val is not None:
                dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_NAMES)
                evals.append((dval, 'val'))

            params = {
                'device': 'cuda',
                'tree_method': 'hist',
                'objective': 'binary:logistic',
                'eval_metric': ['logloss', 'auc'],
                'learning_rate': 0.05,
                'max_depth': 8,
                'subsample': 0.80,
                'colsample_bytree': 0.80,
                'scale_pos_weight': scale_pos,
                'random_state': 42
            }
            self.model = xgb.train(
                params, dtrain,
                num_boost_round=1500,
                evals=evals,
                early_stopping_rounds=50,
                verbose_eval=100
            )
        else:
            self.backend = "lightgbm"
            pos_count = int(np.sum(y_train == 1))
            neg_count = int(np.sum(y_train == 0))
            scale_pos = neg_count / pos_count if pos_count > 0 else 1.0

            params = {
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
                'seed': 42,
                'force_col_wise': True,
            }

            train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
            valid_sets = [train_data]
            valid_names = ['train']

            if X_val is not None and y_val is not None:
                val_data = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_NAMES, reference=train_data)
                valid_sets.append(val_data)
                valid_names.append('valid')

            callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)] if X_val is not None else []
            self.model = lgb.train(
                params, train_data,
                num_boost_round=2000,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks
            )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict match probability for feature matrix X."""
        if self.model is None:
            raise ValueError("Model is not trained or loaded.")

        if self.backend == "xgboost":
            import xgboost as xgb
            dmat = xgb.DMatrix(X, feature_names=FEATURE_NAMES)
            return self.model.predict(dmat)
        else:
            return self.model.predict(X)

    def save(self, filepath: str):
        """Save model to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if self.backend == "xgboost":
            self.model.save_model(filepath)
        else:
            self.model.save_model(filepath)

    def load(self, filepath: str):
        """Load model from file (auto-detects XGBoost .json or LightGBM .txt)."""
        if filepath.endswith('.json') or filepath.endswith('.ubj') or 'xgb' in os.path.basename(filepath).lower():
            import xgboost as xgb
            self.model = xgb.Booster()
            self.model.load_model(filepath)
            self.backend = "xgboost"
        else:
            try:
                self.model = lgb.Booster(model_file=filepath)
                self.backend = "lightgbm"
            except Exception:
                import xgboost as xgb
                self.model = xgb.Booster()
                self.model.load_model(filepath)
                self.backend = "xgboost"
