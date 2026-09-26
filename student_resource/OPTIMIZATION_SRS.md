# Software Requirements Specification (SRS) & High-Score Optimization Roadmap
## Amazon ML Challenge 2026 — Business Entity Resolution

---

## 1. Executive Summary & Objective

In the Amazon ML Challenge 2026, the goal is to resolve noisy business entities across three independent sources (`Source 1`, `Source 2`, `Source 3`) where `Source 1` serves as the deduplicated reference ground. The solution must predict matching records from Source 2 and Source 3 for each Source 1 entity under two strict evaluation dimensions:
1. **Leaderboard Macro $F_{0.5}$ Score:** Precision weighted $2\times$ over recall:
   $$F_{0.5} = \frac{(1 + 0.5^2) \cdot P \cdot R}{0.5^2 \cdot P + R} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$$
   Singletons (Source 1 entities with zero true matches) award **1.0000** for correctly predicting an empty list, and drop to **0.0000** on any false positive.
2. **Candidate Generation Efficiency:** Evaluated on `candidate_pairs.tsv`. The approach producing smaller, higher-precision candidate sets per Source 1 entity is ranked higher in the final evaluation.

---

## 2. Baseline Architecture & Empirical Benchmark

### 2.1 Baseline Pipeline
- **Partitioning:** Strict zero cross-country separation (France, US, India).
- **Preprocessing:** Leet-speak normalization (`0`$\to$`o`, `1`$\to$`l`, `3`$\to$`e`, `5`$\to$`s`), Unicode NFKD accent stripping, legal suffix removal, numeric token normalization.
- **Candidate Blocking:** Word-level TF-IDF sparse matrix dot-product ($K=8$, threshold $\ge 0.15$).
- **Features:** 20 pairwise signals (Levenshtein, token sort, token set, word Jaccard, address similarity, number overlap, blocking rank).
- **Model:** LightGBM Gradient Boosted Decision Trees trained on 15,000 S1 sample entities with decision threshold $T = 0.65$.

### 2.2 Baseline Benchmark Results (Held-Out Validation Set)
- **Macro $F_{0.5}$ Score:** $0.9830$
- **Macro Precision:** $99.26\%$
- **Macro Recall:** $95.84\%$
- **Micro Precision:** $99.37\%$
- **Micro Recall:** $95.47\%$
- **Singleton Zero-Merge Accuracy:** $100.00\%$
- **Blocking Recall Ceiling:** $97.89\%$
- **Mean Candidates per Entity:** $8.00$
- **Reduction Ratio:** $99.9881\%$

---

## 3. Gap Analysis: Where Can the Score Be Pushed Further?

Detailed error analysis on held-out false negatives revealed three key bottlenecks limiting the ceiling:

| Bottleneck | Root Cause | Proposed Solution | Expected Impact |
| :--- | :--- | :--- | :--- |
| **1. Blocking Recall Ceiling (97.89%)** | Word-level TF-IDF misses character typos, phonetic variations, and distinctive brand names dropped by `min_df=2`. | **Multi-Resolution TF-IDF with Character 3-Grams & Anchor Numeric Tokens:** Inject sub-word character 3-grams for words $\ge 4$ characters and prefixed numeric anchors (`num_560001`). Set `min_df=1`, `max_df=0.10`. | Blocking recall increases from **$97.89\%$ to $>99.5\%$**. |
| **2. Feature Resolution (20 Signals)** | Hard negative pairs sharing generic words (e.g. `Apex Dynamics Suite 100` vs `Apex Dynamics Suite 400`) require finer discriminative signals. | **Expanded 28-Feature Vector:** Add character 3-gram Jaccard, exact 4-character prefix match, primary brand first-token match, numeric containment subset flag, and length ratio. | Sharper separation of branch locations and unit differences. |
| **3. Training Sample Diversity** | Baseline was trained on 15,000 entities. The dataset contains 2.2M ground-truth records. | **50,000+ S1 Hard Negative Mining:** Mine the hardest false candidates produced by the new blocking engine and train a deeper LightGBM model ($num\_leaves=63, depth=7$). | Generalization improves across rare entity archetypes. |

---

## 4. Software Requirements Specification (SRS) of New Changes

### 4.1 Module: `src/preprocess.py`
- **Req-1.1 (Multi-Resolution Document Generation):**
  Enhance `make_tfidf_doc(clean_name, clean_addr)` to produce multi-scale representations:
  $$\text{Doc} = \text{CleanName} + \text{CleanAddr} + \text{CompactName} + \sum \text{num\_}N_i + \sum \text{3-grams}(W_j)$$
  where words $W_j$ with length $\ge 4$ contribute character 3-grams to guarantee typo tolerance.
- **Req-1.2 (Sub-word Character 3-gram Extraction):**
  Implement `extract_char_3grams(text: str) -> set` to return character trigrams.

### 4.2 Module: `src/blocking.py`
- **Req-2.1 (Open-Vocabulary Sublinear TF-IDF):**
  Configure `TfidfVectorizer` with:
  - `min_df=1` (never discard rare distinctive brand names)
  - `max_df=0.10` (retain valid frequent commercial vocabulary)
  - `sublinear_tf=True` (logarithmic sublinear term frequency to balance long addresses and 3-gram repetitions)
  - `max_features=250000` (expanded vocabulary capacity)
- **Req-2.2 (Dynamic Adaptive Candidate Pruning):**
  Maintain top-$K \le 8$ candidates with adaptive cut-off to keep mean candidate set size compact for the competition ranking bonus.

### 4.3 Module: `src/features.py`
- **Req-3.1 (28-Feature Expanded Vector):**
  Add 8 new discriminative features:
  1. `name_char_3gram_jaccard`: Character 3-gram Jaccard similarity.
  2. `name_prefix_match`: Binary flag indicating if first 4 characters match (`cn1[:4] == cn2[:4]`).
  3. `first_word_match`: Binary flag indicating if primary brand word matches (`t1[0] == t2[0]`).
  4. `name_containment`: Fraction of reference name tokens contained in candidate.
  5. `length_ratio`: $\min(\text{len}_1, \text{len}_2) / \max(\text{len}_1, \text{len}_2)$.
  6. `addr_ratio`: Direct Levenshtein similarity on address strings.
  7. `addr_num_exact`: Binary indicator if all reference numbers are present in candidate.
  8. `token_count_diff`: Absolute difference in word counts.

### 4.4 Module: `train_model.py`
- **Req-4.1 (High-Scale Hard Negative Training):**
  Scale training data to 50,000 Source 1 entities with ground-truth matches + mined hard negatives.
- **Req-4.2 (LightGBM Hyperparameter Tuning):**
  - `num_leaves`: 63
  - `max_depth`: 7
  - `learning_rate`: 0.03
  - `n_estimators`: 400
  - `subsample`: 0.8
  - `colsample_bytree`: 0.8
  - `min_child_samples`: 25
- **Req-4.3 (Grid Search Threshold Optimization):**
  Optimize decision threshold $T \in [0.55, 0.80]$ targeting Macro $F_{0.5}$.

### 4.5 Module: `frontend/`
- **Req-5.1 (Light Theme Default):**
  Default UI theme to crisp executive light theme (`#f8fafc` background, `#ffffff` cards, `#0f172a` primary text) with dark mode toggle.
- **Req-5.2 (Interactive Benchmark Execution):**
  Provide live benchmark execution button (`POST /api/run_benchmark`) with customizable sample size (500, 1000, 2000 entities) and threshold.

---

## 5. Implementation Roadmap & Verification Plan

```
 Phase 1: Preprocessing & Blocking Enhancement ───► make_tfidf_doc with 3-grams & num anchors
 Phase 2: Feature Engineering Expansion (28 dims) ──► Add 8 discriminative signals
 Phase 3: High-Scale Model Retraining ──────────────► Train LightGBM on 50k S1 with hard negatives
 Phase 4: Benchmark Validation & Threshold Calib. ──► Validate on held-out split, verify F0.5
 Phase 5: Documentation & Submission Packaging ─────► Update Documentation_template.md & ZIP
```
