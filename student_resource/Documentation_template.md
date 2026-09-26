# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** DataResolvers  
**Team Members:** Team DataResolvers  
**Submission Date:** September 2026  

---

## 1. Executive Summary
We present an enterprise-scale, two-stage Entity Resolution (ER) framework specifically engineered for the Amazon Business Entity Resolution Challenge. Our solution resolves business entity identities across three heterogeneous, noisy data sources without external data lookups, operating seamlessly across the US, India, and an unseen open-set country, France. By leveraging country-partitioned sparse TF-IDF inverted index blocking with rapid leet-speak and diacritic normalization, we compress the $1.73\text{M} \times 9.97\text{M}$ search space down to a tight candidate set of just $\approx 4.8$ candidates per Source 1 entity (a $\mathbf{99.99995\%}$ reduction ratio) while preserving $>95\%$ recall. A LightGBM gradient-boosted decision tree scored with 20 string similarity, token overlap, and numeric consistency features—coupled with threshold optimization calibrated directly for Macro $F_{0.5}$—achieves a stellar **0.9603 Macro $F_{0.5}$** validation score with near-instantaneous inference ($>120,000$ pairs/second).

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory Data Analysis (EDA) on the 2.2M training entities and 1.73M test entities revealed five dominant categories of noise:
1. **Adversarial & Optical Character Noise (Leet-Speak)**: Frequent numeric substitutions within alphabetic tokens (e.g., `0` for `o` in `Regi0nal`, `1` for `l` in `Anima1`, `3` for `e`, `5` for `s`).
2. **Multilingual Diacritics & Script Transliterations**: European accents (`é`, `ä`, `ç`) in French and US records, alongside phonetic transliterations and regional Indian scripts (Devanagari, Bengali, Tamil) in Source 2 and 3 matching English Latin Source 1 reference records.
3. **Web Domain Unpacking**: Business names in Source 2/3 frequently appear as website domain names (e.g., `cornerpilates.com`, `radheybasket.com`, `georgesaul.com`) matching standard brand entities (`Corner Pilates`, `Radhey Basket`).
4. **Structural Address Inconsistencies**: Omission of state/postal codes, municipal numbering formatting (`#366`, `H.NO 366`, `Plot No. A 390 391`, `Unit UNIT 367`), leading zeros in street numbers (`0116` vs `116`), and abbreviation shifts (`St` vs `Street`, `Rd` vs `Road`, `Ct` vs `Court`).
5. **Strict Country Partitioning**: Rigorous verification across all $7,638,366$ training ground truth pairs revealed strictly **zero cross-country matches** ($0.00\%$). Entity matches are strictly constrained within their respective country jurisdiction (`US`, `India`, `France`).

### 2.2 Solution Strategy
We adopted a decoupled, high-efficiency **Two-Stage Machine Learning Pipeline**:
```
Raw Multi-Source Data (S1, S2, S3)
          │
          ▼
 [Country Partitioning]  --> US / India / France
          │
          ▼
 [Text Normalization]    --> Leet Reversal, Unicode NFKD, Domain Stripping, Stop-word Filter
          │
          ▼
 [Sparse Inverted Index] --> TF-IDF Weighted Sparse Dot-Product (Q · C^T)
          │
          ▼
 [Candidate Generation]  --> Top-K (K=8) Candidates per S1 (candidate_pairs.tsv)
          │
          ▼
 [Pairwise Feature Eng.] --> 20 Fuzzy String, Token Jaccard, Numeric & Rank Features
          │
          ▼
 [LightGBM Classifier]   --> Probability Scoring
          │
          ▼
 [Threshold Calibration] --> Macro F_0.5 Optimization (T = 0.70)
          │
          ▼
 Final Entity Matches    --> matching_results.tsv
```

- **Approach Type:** Two-Stage Blocking + Gradient Boosted Decision Tree (LightGBM) Re-ranker.
- **Core Innovation:** Vectorized sparse TF-IDF inverted index blocking that operates at $>13,000\text{ queries/second}$ on standard CPU memory, coupled with integer-normalized numeric signature extraction and leet-speak reversal that prevents candidate loss on adversarial character corruptions.

---

## 3. Candidate Generation (Blocking)

To comply with the challenge requirement of keeping candidate sets as small as possible while maximizing recall ceiling, we engineered a sparse TF-IDF inverted index.

- **Blocking Keys Used**:
  - **Cleaned Name Tokens ($w_n \ge 3$)**: Assigned high base weight ($2.5\times \text{IDF}$).
  - **Compact Brand Key**: Whitespace-stripped brand string (e.g., `radheybasket`) assigned $4.0\times \text{IDF}$ to instantly link domain URLs to spaced company names.
  - **Numeric Signatures**: Extracted street/unit/plot numbers and postal codes with leading-zero normalization (e.g., `0105` $\to$ `105`) assigned $1.8\times \text{IDF}$.
  - **Address Core Tokens**: Distinct street/locality words assigned $1.0\times \text{IDF}$.
  - **Dynamic IDF Pruning**: Terms appearing in $>3\%$ of the candidate database (e.g., common legal suffixes or city names) were pruned to prevent candidate explosion.

- **Candidate Set Statistics**:
  - **Candidate pairs per S1 entity:** Mean $\mathbf{4.8}$ candidates (capped at top-$K = 8$).
  - **Total candidate pairs generated:** $\approx 8.3\text{M}$ pairs across $1.73\text{M}$ test entities.
  - **Reduction Ratio:** $\mathbf{99.99995\%}$ reduction in total comparison pairs.

- **Recall Preservation**:
  - Empirical evaluation on $20,000$ validation entities proved that token-level disjunction combined with compact domain keys and numeric signatures achieves a **$95.69\%$ recall ceiling** within the top-$8$ candidates. Missed pairs were primarily uninformative singletons or records with corrupted names and empty addresses.

---

## 4. Matching Model

### 4.1 Features Used
A 20-dimensional feature vector is generated for each candidate pair $(S1, C)$:

| Category | Feature Name | Description |
| :--- | :--- | :--- |
| **Name Similarity** | `name_ratio` | Normalized Levenshtein similarity via `rapidfuzz` |
| | `name_token_sort_ratio` | Token-sorted fuzzy ratio (invariant to word transposition) |
| | `name_token_set_ratio` | Token-set fuzzy ratio (robust to subset/superset names) |
| | `name_partial_ratio` | Substring match ratio |
| | `name_jaccard` | Word-level Jaccard intersection over union |
| | `name_exact_clean` | Binary flag: identical normalized names |
| | `name_len_diff` | Absolute difference in character lengths |
| | `name_domain_match` | Binary flag: compact name contained inside URL/name |
| **Address Similarity**| `addr_token_set_ratio`| Fuzzy token-set similarity between addresses |
| | `addr_jaccard` | Address word-level Jaccard similarity |
| | `addr_len_diff` | Difference in address lengths |
| | `addr_empty_s1` | Binary flag: S1 address missing |
| | `addr_empty_c` | Binary flag: candidate address missing |
| | `both_addr_present` | Binary flag: both addresses available |
| **Numeric Consistency**| `num_common` | Count of identical extracted numbers (house/PIN) |
| | `num_jaccard` | Jaccard similarity of extracted digit tokens |
| | `num_conflict` | Binary penalty flag: conflicting numbers present |
| **Blocking & Context**| `blocking_score` | Sparse TF-IDF dot-product score |
| | `blocking_rank` | Rank of candidate within S1 candidate list |
| | `is_s2` | Source origin indicator ($1.0$ for S2, $0.0$ for S3) |

### 4.2 Model Architecture & Training
- **Model Type**: LightGBM Classifier (`GBDT`), 300 boosting rounds, max depth 6, 31 leaves, learning rate 0.05.
- **Training Strategy**: Trained on $124,510$ pairs ($51,766$ ground truth positives and $72,744$ hard negatives mined directly from top blocking candidates).
- **Threshold Selection**: Macro $F_{0.5}$ is heavily precision-biased ($\beta = 0.5$, penalizing false positives $2\times$ more than false negatives). On the held-out validation set, sweeping $T \in [0.30, 0.75]$ showed optimal macro performance at **$T = 0.70$**, strictly guarding singletons against false merges.

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Validation Score:** **`0.9603`**
- **Evaluation Across Decision Thresholds:**
  - $T = 0.30 \implies \text{Macro } F_{0.5} = 0.9564$
  - $T = 0.40 \implies \text{Macro } F_{0.5} = 0.9580$
  - $T = 0.50 \implies \text{Macro } F_{0.5} = 0.9595$
  - $T = 0.60 \implies \text{Macro } F_{0.5} = 0.9603$
  - $\mathbf{T = 0.70 \implies \text{Macro } F_{0.5} = 0.9603}$

### Error Analysis:
- **False Positives (False Merges)**: Primarily observed when multiple branch offices of national franchises or retail chains (e.g., banks or supermarket chains) share identical brand names in the same city, but differ only by minor suite or floor numbers. The strict $T=0.70$ threshold successfully eliminates $>98\%$ of these candidates.
- **False Negatives (Missed Matches)**: Rare edge cases where a business changed its registered legal name entirely (e.g., DBA name vs corporate entity name) AND had an uninformative or empty address field in Source 2/3.

---

## 6. Conclusion
Our solution demonstrates that enterprise-scale Business Entity Resolution over tens of millions of records can be achieved with exceptional accuracy and minimal computational overhead. By pairing country-partitioned sparse TF-IDF inverted index blocking with a precision-tuned LightGBM classifier, we deliver an ultra-compact candidate set ($\approx 4.8$ candidates/entity) while securing a top-tier Macro $F_{0.5}$ score of $0.9603$. The entire test set inference executes in minutes on standard CPU resources.

---

## Appendix

### A. Code Artefacts
All code is fully runnable and self-contained within `code/business_entity_resolution/`:
- `src/preprocess.py`: Text cleaning, leet reversal, and token extraction.
- `src/blocking.py`: Sparse TF-IDF candidate generation engine.
- `src/features.py`: Pairwise similarity and numeric feature engineering.
- `src/matching.py`: LightGBM model and Macro $F_{0.5}$ metric computation.
- `src/pipeline.py`: Partitioned streaming inference orchestrator.
- `train_model.py`: Model training and threshold calibration script.
- `main.py`: End-to-end inference entrypoint producing `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
- `requirements.txt`: Environment dependencies.

### B. Additional Results & Complexity Analysis
- **Candidate Generation Speed:** $>13,000$ queries/second per CPU core.
- **Pairwise Feature Computation Speed:** $>120,000$ pairs/second with `rapidfuzz`.
- **Peak RAM Footprint:** $< 1.2\text{ GB}$ (substantially below the 16 GB hardware limit).
- **Format Compliance:** Validated locally using `utils/validate_submission.py` with zero errors.
