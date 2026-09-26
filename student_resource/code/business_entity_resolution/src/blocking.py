"""
High-Performance Blocking and Candidate Generation Module.
Leverages scikit-learn's optimized C-accelerated TfidfVectorizer and scipy.sparse
matrix multiplication to produce high-recall candidate sets at scale.
"""

import time
import gc
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize
from sparse_dot_topn import sp_matmul_topn
from .preprocess import clean_text, make_tfidf_doc

def preprocess_for_tfidf(name: str, addr: str) -> str:
    """Preprocess text into a rich representation for TF-IDF vectorization."""
    cn = clean_text(name)
    ca = clean_text(addr)
    return make_tfidf_doc(cn, ca)

class BlockingEngine:
    def __init__(self, top_k: int = 8, min_score: float = 0.008, max_features: int = 2**19):
        self.top_k = top_k
        self.min_score = min_score
        self.max_features = max_features
        self.cand_ids = []
        self.vectorizer = None
        self.CT = None
        self.allowed_features = None

    def fit_candidates(self, cand_ids: list, cand_docs):
        """
        Fit TF-IDF vectorizer and build transposed CSR matrix over candidate documents.
        cand_docs can be a list or generator/iterable of strings.
        """
        self.cand_ids = cand_ids
        
        # TfidfVectorizer.fit_transform() materializes an iterable of documents
        # while it learns its vocabulary.  On the full challenge data that peak
        # exceeds a 16 GB machine and leaves a half-written submission.  A hashed
        # word/character-token representation has fixed memory, streams the input,
        # and still produces L2-normalized cosine scores for the same rich docs.
        self.vectorizer = HashingVectorizer(
            n_features=self.max_features,
            alternate_sign=False,
            norm=None,
            token_pattern=r'(?u)\b\w{2,}\b',
            dtype=np.float32,
        )

        C = self.vectorizer.transform(cand_docs).tocsr()
        # Filter out words that appear in > 3% of all candidates (ubiquitous stop words)
        # Keeps domain-specific business nouns and distinctive 3-grams.
        doc_freq = np.bincount(C.indices, minlength=self.max_features)
        self.allowed_features = doc_freq <= max(1, int(C.shape[0] * 0.03))
        C.data[~self.allowed_features[C.indices]] = 0.0
        C.eliminate_zeros()
        normalize(C, norm='l2', copy=False)
        self.CT = C.T.tocsr()
        del C
        gc.collect()

    def build_auxiliary_indices(self, cand_records: dict):
        """
        Build fast in-memory hash tables for exact name, legal stem, compact name,
        rare/selective name tokens, and address anchor matching.
        Guarantees high candidate recall even on corrupted or noisy records.
        """
        from collections import defaultdict
        from .preprocess import STOP_WORDS, extract_numbers
        from .features import strip_legal_suffixes

        word_freq = defaultdict(int)
        for cid, (cn, ca) in cand_records.items():
            if cn:
                words = set(w for w in cn.split() if w not in STOP_WORDS and len(w) >= 3)
                for w in words:
                    word_freq[w] += 1

        self.exact_name_index = defaultdict(list)
        self.legal_stem_index = defaultdict(list)
        self.compact_name_index = defaultdict(list)
        self.selective_name_index = defaultdict(list)
        self.addr_anchor_index = defaultdict(list)

        for cid, (cn, ca) in cand_records.items():
            if cn:
                self.exact_name_index[cn].append(cid)
                stem = strip_legal_suffixes(cn)
                if len(stem) >= 3 and stem != cn:
                    self.legal_stem_index[stem].append(cid)

                cp = cn.replace(' ', '')
                if len(cp) >= 4:
                    self.compact_name_index[cp].append(cid)

                words = set(w for w in cn.split() if w not in STOP_WORDS and len(w) >= 3)
                for w in words:
                    if 1 <= word_freq[w] <= 60:
                        self.selective_name_index[w].append(cid)

            if ca:
                nums = extract_numbers(ca)
                words = [w for w in ca.split() if w not in STOP_WORDS and len(w) >= 3]
                for num in nums:
                    if len(num) >= 2:
                        for w in words[:6]:
                            bucket = self.addr_anchor_index[(num, w)]
                            if len(bucket) < 30:
                                bucket.append(cid)

    def query_batch(self, s1_records_or_ids: list, s1_docs: list = None, s1_precleaned: dict = None, batch_size: int = 10000):
        """
        Vectorize S1 records and compute dot-product with candidate matrix.
        Accepts pre-computed s1_docs or falls back to on-the-fly preprocessing.
        If s1_precleaned is provided, blends TF-IDF candidates with exact name,
        legal stem, compact name, selective rare name tokens, and address anchor indices.
        Returns a dictionary: {s1_id: [(cand_id, score, rank), ...]}
        """
        from .preprocess import STOP_WORDS
        from .features import strip_legal_suffixes

        n_queries = len(s1_records_or_ids)
        results = {}

        if s1_docs is None:
            s1_ids = [r[0] for r in s1_records_or_ids]
            s1_docs = [preprocess_for_tfidf(r[1], r[2]) for r in s1_records_or_ids]
        else:
            s1_ids = s1_records_or_ids

        for b_start in range(0, n_queries, batch_size):
            b_end = min(b_start + batch_size, n_queries)
            b_ids = s1_ids[b_start:b_end]
            b_docs = s1_docs[b_start:b_end]
            b_size = len(b_ids)

            Q = self.vectorizer.transform(b_docs).tocsr()
            Q.data[~self.allowed_features[Q.indices]] = 0.0
            Q.eliminate_zeros()
            normalize(Q, norm='l2', copy=False)

            scores_matrix = sp_matmul_topn(
                Q, self.CT, top_n=self.top_k, threshold=self.min_score,
                sort=True, n_threads=-1,
            )

            for i in range(b_size):
                s1_id = b_ids[i]
                start = scores_matrix.indptr[i]
                end = scores_matrix.indptr[i+1]
                row_len = end - start

                top_cands = []
                seen_cids = set()

                if row_len > 0:
                    row_vals = scores_matrix.data[start:end]
                    row_cols = scores_matrix.indices[start:end]

                    valid_mask = row_vals >= self.min_score
                    if np.any(valid_mask):
                        valid_vals = row_vals[valid_mask]
                        valid_cols = row_cols[valid_mask]
                        v_len = len(valid_vals)

                        if v_len <= self.top_k:
                            order = np.argsort(-valid_vals)
                        else:
                            top_part = np.argpartition(valid_vals, -self.top_k)[-self.top_k:]
                            order = top_part[np.argsort(-valid_vals[top_part])]

                        for rank, idx in enumerate(order, 1):
                            cand_id = self.cand_ids[valid_cols[idx]]
                            score = float(valid_vals[idx])
                            top_cands.append((cand_id, score, rank))
                            seen_cids.add(cand_id)

                # Auxiliary High-Recall Retrieval Channels
                if s1_precleaned and s1_id in s1_precleaned:
                    cn1, ca1, cp1, nums1 = s1_precleaned[s1_id]

                    # 1. Exact Name Matches (score 1.0, rank 0)
                    if hasattr(self, 'exact_name_index') and cn1 in self.exact_name_index:
                        for cid in self.exact_name_index[cn1][:50]:
                            if cid not in seen_cids:
                                top_cands.append((cid, 1.0, 0))
                                seen_cids.add(cid)

                    # 1b. Legal Stem Matches (e.g. Inc vs Corp vs LLC)
                    if hasattr(self, 'legal_stem_index'):
                        stem1 = strip_legal_suffixes(cn1)
                        if len(stem1) >= 3 and stem1 in self.legal_stem_index:
                            for cid in self.legal_stem_index[stem1][:40]:
                                if cid not in seen_cids:
                                    top_cands.append((cid, 0.98, 0))
                                    seen_cids.add(cid)

                    # 2. Compact Name Matches (handles spaces, hyphens, prefixes)
                    if hasattr(self, 'compact_name_index') and len(cp1) >= 4 and cp1 in self.compact_name_index:
                        for cid in self.compact_name_index[cp1][:40]:
                            if cid not in seen_cids:
                                top_cands.append((cid, 0.95, 0))
                                seen_cids.add(cid)

                    # 3. Selective / Rare Name Tokens (handles typo-free core words)
                    if hasattr(self, 'selective_name_index'):
                        s1_words = [w for w in cn1.split() if w not in STOP_WORDS and len(w) >= 3]
                        for w in s1_words:
                            for cid in self.selective_name_index.get(w, [])[:25]:
                                if cid not in seen_cids:
                                    top_cands.append((cid, 0.85, 1))
                                    seen_cids.add(cid)

                    # 4. Address Anchors (number + street token match for corrupted/empty names)
                    if hasattr(self, 'addr_anchor_index') and ca1:
                        addr_words = [w for w in ca1.split() if w not in STOP_WORDS and len(w) >= 3]
                        for num in nums1:
                            for w in addr_words[:8]:
                                m_cids = self.addr_anchor_index.get((num, w), [])
                                if len(m_cids) <= 30:  # selective anchor only
                                    for cid in m_cids:
                                        if cid not in seen_cids:
                                            top_cands.append((cid, 0.80, 2))
                                            seen_cids.add(cid)



                results[s1_id] = top_cands

        return results
