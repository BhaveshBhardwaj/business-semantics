"""
End-to-End Pipeline Orchestration Module.
Processes test data partitioned by country (France, US, India), performs
scalable candidate generation (blocking), extracts features, scores candidate pairs
with the trained ML model, and writes compliant matching_results.tsv and candidate_pairs.tsv.
"""

import os
import sys
import gc
import time
import shutil
import numpy as np
from tqdm import tqdm
from .preprocess import clean_text, extract_numbers, make_tfidf_doc
from .blocking import BlockingEngine, preprocess_for_tfidf
from .features import extract_pair_features_fast
from .matching import MatchingModel

def partition_test_files_by_country(test_dir: str, tmp_dir: str):
    """
    Split test_source1, test_source2, test_source3 by country into tmp_dir in a single streaming pass.
    Avoids repeatedly scanning multi-million-row files.
    """
    os.makedirs(tmp_dir, exist_ok=True)

    # Country is an open-set string field.  Derive partitions from Source 1
    # instead of assuming the three labels known when this challenge brief was
    # written.  Numeric partition names are safe even if a future country label
    # contains a path separator.
    countries = set()
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    with open(s1_path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) >= 4:
                countries.add(p[3].strip())
    countries = sorted(countries)
    country_index = {country: i for i, country in enumerate(countries)}

    print("Pre-partitioning test dataset by country into temporary storage...", flush=True)
    t0 = time.time()

    for src_name in ['test_source1', 'test_source2', 'test_source3']:
        src_path = os.path.join(test_dir, f"{src_name}.tsv")
        if not os.path.isfile(src_path):
            continue

        writers = {
            c: open(os.path.join(tmp_dir, f"{src_name}_{country_index[c]:03d}.tsv"), 'w', encoding='utf-8')
            for c in countries
        }

        with open(src_path, 'r', encoding='utf-8') as f:
            next(f) # header
            for line in f:
                p = line.rstrip('\n').split('\t')
                if len(p) >= 4:
                    country = p[3].strip()
                    if country in writers:
                        writers[country].write(line)

        for w in writers.values():
            w.close()

    print(f"Partitioning completed in {time.time() - t0:.1f}s across {len(countries)} country labels.", flush=True)
    return [(country, country_index[country]) for country in countries]

def run_pipeline(test_dir: str, output_dir: str, model_path: str, threshold: float = 0.65, top_k: int = 25):
    """
    Run end-to-end inference over all test source files.
    Generates:
        output_dir/matching_results.tsv
        output_dir/candidate_pairs.tsv
    """
    os.makedirs(output_dir, exist_ok=True)
    matching_tsv = os.path.join(output_dir, "matching_results.tsv")
    candidate_tsv = os.path.join(output_dir, "candidate_pairs.tsv")

    tmp_dir = os.path.join(output_dir, "_tmp_partitioned")

    print("=" * 80)
    print("Running Business Entity Resolution Pipeline")
    print(f"Test Directory:   {test_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Model Path:       {model_path}")
    print(f"Threshold:        {threshold:.2f} | Top-K: {top_k}")
    print("=" * 80)

    # 1. Pre-partition test files by country for linear streaming
    country_partitions = partition_test_files_by_country(test_dir, tmp_dir)

    # 2. Load trained model
    matching_model = MatchingModel(threshold=threshold)
    matching_model.load(model_path)
    print(f"Loaded {matching_model.backend.upper()} matching model successfully.", flush=True)

    # Initialize output files with headers
    with open(matching_tsv, 'w', encoding='utf-8') as f_match:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")

    with open(candidate_tsv, 'w', encoding='utf-8') as f_cand:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

    total_s1_processed = 0
    t_start = time.time()

    for country, country_idx in country_partitions:
        print(f"\n>>> Processing Country Partition: {country.upper()} <<<", flush=True)
        t_country = time.time()

        s1_part = os.path.join(tmp_dir, f"test_source1_{country_idx:03d}.tsv")
        s2_part = os.path.join(tmp_dir, f"test_source2_{country_idx:03d}.tsv")
        s3_part = os.path.join(tmp_dir, f"test_source3_{country_idx:03d}.tsv")

        # 1. Load candidate database for this country (Source 2 and Source 3)
        print(f"[{country}] Loading candidate records from Source 2 and Source 3...", flush=True)
        cand_records = {} # cid -> (clean_name, clean_addr)
        cand_ids = []

        for src_fn in [s2_part, s3_part]:
            if not os.path.isfile(src_fn):
                continue
            with open(src_fn, 'r', encoding='utf-8') as f:
                for line in f:
                    p = line.rstrip('\n').split('\t')
                    if len(p) >= 4:
                        cid, raw_name, raw_addr = p[0], p[1], p[2]
                        cn = clean_text(raw_name)
                        ca = clean_text(raw_addr)
                        cand_records[cid] = (cn, ca)
                        cand_ids.append(cid)

        n_cands = len(cand_records)
        print(f"[{country}] Loaded {n_cands:,} candidate records in {time.time() - t_country:.1f}s.", flush=True)

        if n_cands == 0:
            print(f"[{country}] Warning: No candidates found for {country}. Writing empty rows.")
            if os.path.isfile(s1_part):
                with open(s1_part, 'r', encoding='utf-8') as f_s1, \
                     open(matching_tsv, 'a', encoding='utf-8') as f_match, \
                     open(candidate_tsv, 'a', encoding='utf-8') as f_cand:
                    for line in f_s1:
                        p = line.rstrip('\n').split('\t')
                        if len(p) >= 4:
                            f_match.write(f"{p[0]}\t\n")
                            f_cand.write(f"{p[0]}\t\n")
                            total_s1_processed += 1
            continue

        # 2. Build Blocking Index for this country using generator
        print(f"[{country}] Fitting TF-IDF Blocking Index...", flush=True)
        t_fit = time.time()
        blocking = BlockingEngine(top_k=top_k, min_score=0.008)
        
        def cand_doc_generator():
            for cid in cand_ids:
                cn, ca = cand_records[cid]
                yield make_tfidf_doc(cn, ca)

        blocking.fit_candidates(cand_ids, cand_doc_generator())
        blocking.build_auxiliary_indices(cand_records)
        print(f"[{country}] Blocking Index fitted in {time.time() - t_fit:.1f}s.", flush=True)

        # 3. Stream Source 1 records in batches
        print(f"[{country}] Streaming Source 1 queries and generating matches...", flush=True)
        # A 10k sparse matrix product can spike memory on the large India
        # partition.  This bounded batch size is slower only marginally but
        # prevents an interrupted run from leaving a truncated submission.
        batch_size = 5000
        current_raw_batch = []
        country_s1_count = 0

        with open(s1_part, 'r', encoding='utf-8') as f_s1, \
             open(matching_tsv, 'a', encoding='utf-8') as f_match, \
             open(candidate_tsv, 'a', encoding='utf-8') as f_cand:

            pbar = tqdm(f_s1, desc=f"[{country.upper()}] Matching", unit=" ent", file=sys.stdout, ascii=True, mininterval=0.5)
            for line in pbar:
                p = line.rstrip('\n').split('\t')
                if len(p) >= 4:
                    current_raw_batch.append((p[0], p[1], p[2], p[3]))
                    country_s1_count += 1

                    if len(current_raw_batch) >= batch_size:
                        _process_batch_fast(current_raw_batch, blocking, cand_records, matching_model,
                                           threshold, f_match, f_cand)
                        total_s1_processed += len(current_raw_batch)
                        current_raw_batch = []

            # Process final batch
            if current_raw_batch:
                _process_batch_fast(current_raw_batch, blocking, cand_records, matching_model,
                                   threshold, f_match, f_cand)
                total_s1_processed += len(current_raw_batch)
                current_raw_batch = []

            pbar.close()
            f_match.flush()
            f_cand.flush()

        print(f"[{country}] Completed {country_s1_count:,} Source 1 entities in {time.time() - t_country:.1f}s.")

        # Clean memory for next country
        del cand_records
        del cand_ids
        del blocking
        gc.collect()

    # Clean up temporary partitioned files
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n" + "=" * 80)
    print(f"Inference Completed in {time.time() - t_start:.1f}s!")
    print(f"Total Source 1 Entities Processed: {total_s1_processed:,}")
    print(f"Output files generated:")
    print(f"  - {matching_tsv}")
    print(f"  - {candidate_tsv}")
    print("=" * 80)

def _process_batch_fast(s1_batch, blocking, cand_records, matching_model, threshold, f_match, f_cand):
    """Process batch with pre-cleaned representations for ultra-fast feature extraction."""
    # 1. Pre-clean S1 batch records once
    s1_precleaned = {}
    s1_ids = []
    s1_docs = []
    for s1_id, raw_name, raw_addr, _ in s1_batch:
        cn = clean_text(raw_name)
        ca = clean_text(raw_addr)
        cp = cn.replace(" ", "")
        nums = extract_numbers(cn) | extract_numbers(ca)
        s1_precleaned[s1_id] = (cn, ca, cp, nums)
        s1_ids.append(s1_id)
        s1_docs.append(make_tfidf_doc(cn, ca))

    # 2. Blocking query with multi-channel auxiliary index retrieval
    batch_candidates = blocking.query_batch(s1_ids, s1_docs=s1_docs, s1_precleaned=s1_precleaned)

    # 3. Extract features for all pairs using pre-cleaned representations
    all_pairs = [] # (s1_id, cid)
    feature_rows = []

    for s1_id in s1_ids:
        cands = batch_candidates.get(s1_id, [])
        s1_item = s1_precleaned[s1_id]
        for cid, score, rank in cands:
            if cid in cand_records:
                c_item = cand_records[cid]
                feat = extract_pair_features_fast(s1_item, c_item, cid, score, rank)
                feature_rows.append(feat)
                all_pairs.append((s1_id, cid))

    # 4. Model Scoring
    match_dict = {s1_id: [] for s1_id in s1_ids}
    cand_dict = {s1_id: [] for s1_id in s1_ids}

    for s1_id in s1_ids:
        cands = batch_candidates.get(s1_id, [])
        cand_dict[s1_id] = [cid for cid, _, _ in cands if cid in cand_records]

    if feature_rows:
        X_mat = np.array(feature_rows, dtype=np.float32)
        probs = matching_model.predict_proba(X_mat)

        for (s1_id, cid), prob, feat in zip(all_pairs, probs, feature_rows):
            if prob >= threshold:
                # Reject explicit geographical postal code conflicts unless exact name match
                if feat[5] == 0 and feat[31] == -1.0 and feat[1] < 90:
                    continue
                match_dict[s1_id].append(cid)

    # 5. Write to files
    for s1_id in s1_ids:
        c_list = cand_dict.get(s1_id, [])
        m_list = match_dict.get(s1_id, [])

        unique_c = list(dict.fromkeys(c_list))
        unique_m = list(dict.fromkeys(m_list))

        # Guarantee every matched ID is present in candidate set
        for mid in unique_m:
            if mid not in unique_c:
                unique_c.append(mid)

        cand_str = ",".join(unique_c)
        match_str = ",".join(unique_m)

        f_cand.write(f"{s1_id}\t{cand_str}\n")
        f_match.write(f"{s1_id}\t{match_str}\n")
