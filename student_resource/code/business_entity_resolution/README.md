# Business Entity Resolution — Amazon ML Challenge 2026

## Overview
High-precision entity resolution system that matches business entities across
multiple data sources (Source 1, Source 2, Source 3) using TF-IDF blocking +
LightGBM pairwise classification, optimized for **Macro F₀.₅ ≥ 0.999**.

## Architecture
```
Source 1 Entities
       │
       ▼
┌──────────────────┐
│  Text Preprocessing  │   normalize, lowercase, strip punctuation
└──────────┬───────┘
           ▼
┌──────────────────┐
│  TF-IDF Blocking     │   HashingVectorizer + sparse top-K retrieval
│  (top_k=15, min=0.05)│   + exact company name index fallback
└──────────┬───────┘
           ▼
┌──────────────────┐
│  28-Feature Extraction│   name sims, address sims, number overlap,
│                        │   cross-field matches, blocking scores
└──────────┬───────┘
           ▼
┌──────────────────┐
│  LightGBM GBDT        │   127 leaves, depth 8, scale_pos_weight,
│  + XGBoost GPU (opt.)  │   early stopping, 2000 rounds
└──────────┬───────┘
           ▼
┌──────────────────┐
│  Threshold Sweep      │   3-phase: coarse→fine→ultra-fine
│  Optimized for F₀.₅   │   precision-weighted decision boundary
└──────────┬───────┘
           ▼
   Final Match Output
```

## Approach History

### Iteration 1: TF-IDF + XGBoost (CPU)
- **Model**: XGBoost with HashingVectorizer TF-IDF features
- **Training**: ~75K entity sample (3.4% of data)
- **Result**: Macro F₀.₅ ≈ 0.88
- **Issue**: Underfitting from small sample; TF-IDF features alone insufficient

### Iteration 2: PyTorch Deep MLP (GPU)
- **Model**: `EntityMatchNet` — 4-layer residual MLP with BatchNorm + Dropout
- **Training**: ~75K entities on RTX 4050 GPU
- **Result**: Macro F₀.₅ ≈ 0.92
- **Issue**: GPU underutilized (1.2GB/6GB VRAM); MLP wrong architecture for
  28-feature tabular data; small sample still limits generalization

### Iteration 3 (Current): Full-Data LightGBM
- **Model**: LightGBM GBDT (127 leaves, depth 8) — optimal for tabular
- **Training**: ALL 2.2M entities, 7.6M+ positive pairs
- **Blocking**: Expanded to top_k=15, min_score=0.05 for near-100% recall
- **Features**: 28 pairwise signals + exact name index + hard negative mining
- **Threshold**: 3-phase sweep (coarse→fine→ultra-fine) optimized for F₀.₅
- **Target**: Macro F₀.₅ ≥ 0.999

## Training

```bash
# Full training (all 2.2M entities, ~2-4 hours)
python train_model.py

# Quick validation (10% sample, ~15 min)
python train_model.py --sample-frac 0.10

# With XGBoost GPU ensemble
python train_model.py --use-xgb-gpu
```

## Inference

```bash
# Generate submission
python main.py --test-dir ../../dataset/test --output-dir output
```

## Evaluation

```bash
# Benchmark on held-out validation entities
python evaluate_scores.py --n-val 5000
```

## Project Structure

```
business_entity_resolution/
├── main.py               # CLI entry point for inference
├── train_model.py        # Full-data LightGBM training pipeline
├── evaluate_scores.py    # Benchmark evaluation script
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── models/
│   ├── matching_lgbm.txt     # Trained LightGBM model
│   ├── matching_xgb.json     # (Optional) XGBoost GPU model
│   ├── threshold.txt         # Optimal decision threshold
│   └── training_metrics.txt  # Training run metrics
└── src/
    ├── __init__.py
    ├── preprocess.py     # Text normalization & cleaning
    ├── blocking.py       # TF-IDF blocking engine
    ├── features.py       # 28-feature pairwise extraction
    ├── matching.py       # Model loading & prediction
    └── pipeline.py       # End-to-end inference orchestrator
```

## Competition Metric

**Macro F₀.₅** averaged over all Source 1 entities:
- Per-entity: `F₀.₅ = 1.25 × Precision × Recall / (0.25 × Precision + Recall)`
- Singletons: Score = 1.0 if correctly empty, 0.0 on any false merge
- **False merges are catastrophic** — singleton misclassification = instant 0.0

## Requirements
```
numpy
scipy
scikit-learn
lightgbm
sparse_dot_topn
tqdm
xgboost  # optional, for GPU ensemble
```
