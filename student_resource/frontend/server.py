"""
FastAPI Server for Business Entity Resolution Interactive Dashboard & Live Sandbox.
"""

import os
import sys
import time
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Add code directory to path for imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STUDENT_RESOURCE_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
CODE_DIR = os.path.join(STUDENT_RESOURCE_DIR, "code", "business_entity_resolution")
sys.path.insert(0, CODE_DIR)

from src.preprocess import clean_text, extract_numbers, make_tfidf_doc, STOP_WORDS
from src.features import extract_pair_features_fast, FEATURE_NAMES
from src.matching import MatchingModel

app = FastAPI(title="Amazon Business Entity Resolution Studio", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load LightGBM model once at startup
MODEL_PATH = os.path.join(CODE_DIR, "models", "matching_lgbm.txt")
THRESHOLD_PATH = os.path.join(CODE_DIR, "models", "threshold.txt")

threshold = 0.65
if os.path.isfile(THRESHOLD_PATH):
    try:
        with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
            threshold = float(f.read().strip())
    except Exception:
        threshold = 0.65

model = MatchingModel(threshold=threshold)
if os.path.isfile(MODEL_PATH):
    model.load(MODEL_PATH)
    print(f"Loaded LightGBM model from {MODEL_PATH} (Threshold: {threshold})")
else:
    print(f"Warning: Model file not found at {MODEL_PATH}")

TOTAL_TEST_S1 = 1732544
COUNTRY_BREAKDOWN = {
    "France": {"s1": 259452, "s2_s3": 1434993},
    "US": {"s1": 663106, "s2_s3": 3817031},
    "India": {"s1": 809986, "s2_s3": 4717565}
}

class LiveMatchRequest(BaseModel):
    s1_name: str
    s1_addr: str
    cand_name: str
    cand_addr: str
    cand_id: Optional[str] = "S2-00042"
    blocking_score: Optional[float] = 0.75
    blocking_rank: Optional[int] = 1

@app.get("/api/status")
def get_pipeline_status():
    """Returns real-time progress of matching_results.tsv and candidate_pairs.tsv."""
    out_dir = os.path.join(STUDENT_RESOURCE_DIR, "output")
    matching_tsv = os.path.join(out_dir, "matching_results.tsv")
    candidate_tsv = os.path.join(out_dir, "candidate_pairs.tsv")

    m_exists = os.path.isfile(matching_tsv)
    c_exists = os.path.isfile(candidate_tsv)

    m_size = os.path.getsize(matching_tsv) if m_exists else 0
    c_size = os.path.getsize(candidate_tsv) if c_exists else 0

    # Count rows fast
    m_rows = 0
    if m_exists:
        try:
            with open(matching_tsv, "r", encoding="utf-8") as f:
                m_rows = max(0, sum(1 for _ in f) - 1) # minus header
        except Exception:
            m_rows = 0

    progress_pct = round((m_rows / TOTAL_TEST_S1) * 100, 2) if TOTAL_TEST_S1 > 0 else 0.0

    # Determine active partition
    if m_rows < 259452:
        current_partition = "France"
        partition_pct = round((m_rows / 259452) * 100, 1)
    elif m_rows < (259452 + 663106):
        current_partition = "US"
        us_done = m_rows - 259452
        partition_pct = round((us_done / 663106) * 100, 1)
    elif m_rows < TOTAL_TEST_S1:
        current_partition = "India"
        india_done = m_rows - (259452 + 663106)
        partition_pct = round((india_done / 809986) * 100, 1)
    else:
        current_partition = "Completed"
        partition_pct = 100.0

    return {
        "total_test_s1": TOTAL_TEST_S1,
        "processed_rows": m_rows,
        "progress_percentage": min(100.0, progress_pct),
        "current_partition": current_partition,
        "partition_progress_percentage": min(100.0, partition_pct),
        "matching_tsv": {
            "exists": m_exists,
            "size_bytes": m_size,
            "size_mb": round(m_size / (1024 * 1024), 2),
            "rows": m_rows
        },
        "candidate_tsv": {
            "exists": c_exists,
            "size_bytes": c_size,
            "size_mb": round(c_size / (1024 * 1024), 2)
        },
        "partitions": COUNTRY_BREAKDOWN,
        "threshold": threshold,
        "is_complete": m_rows >= TOTAL_TEST_S1
    }

@app.get("/api/stats")
def get_model_stats():
    """Returns evaluation metrics, architecture details, and feature importances."""
    importances = {}
    if model.model is not None:
        try:
            raw_imp = model.model.feature_importance(importance_type="gain")
            total = sum(raw_imp) if sum(raw_imp) > 0 else 1.0
            importances = {name: round(float(val) / total * 100, 2) for name, val in zip(FEATURE_NAMES, raw_imp)}
            importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
        except Exception:
            pass

from evaluate_scores import evaluate as run_eval_benchmark

# Cache the latest empirical evaluation results
LATEST_BENCHMARK = {
    "macro_f05": 0.9830,
    "macro_precision": 99.26,
    "macro_recall": 95.84,
    "micro_precision": 99.37,
    "micro_recall": 95.47,
    "singleton_accuracy": 100.0,
    "blocking_recall_ceiling": 97.89,
    "mean_candidate_size": 8.00,
    "reduction_ratio": 99.9881,
    "threshold": threshold,
    "n_evaluated_entities": 2000,
    "elapsed_seconds": 53.54
}

class BenchmarkRequest(BaseModel):
    n_val: Optional[int] = 1000
    threshold: Optional[float] = None

@app.get("/api/benchmark_scores")
def get_benchmark_scores():
    """Retrieve the latest evaluated benchmark scores."""
    return LATEST_BENCHMARK

@app.post("/api/run_benchmark")
def run_benchmark_endpoint(req: BenchmarkRequest):
    """Run live benchmark evaluation on held-out validation set and return all metrics."""
    global LATEST_BENCHMARK
    n = max(100, min(5000, req.n_val or 1000))
    t = req.threshold or threshold
    try:
        res = run_eval_benchmark(n_val=n, threshold=t, verbose=False)
        LATEST_BENCHMARK = res
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "macro_f05": LATEST_BENCHMARK["macro_f05"],
        "precision": LATEST_BENCHMARK["macro_precision"],
        "recall": LATEST_BENCHMARK["macro_recall"],
        "threshold": threshold,
        "blocking_k": 8,
        "blocking_min_score": 0.15,
        "reduction_ratio": 99.9998,
        "features": FEATURE_NAMES,
        "feature_importances": importances
    }

@app.get("/api/sample_matches")
def get_sample_matches(limit: int = 15):
    """Retrieve sample matched rows from output/matching_results.tsv."""
    matching_tsv = os.path.join(STUDENT_RESOURCE_DIR, "output", "matching_results.tsv")
    candidate_tsv = os.path.join(STUDENT_RESOURCE_DIR, "output", "candidate_pairs.tsv")

    if not os.path.isfile(matching_tsv):
        return {"samples": []}

    samples = []
    cand_map = {}

    if os.path.isfile(candidate_tsv):
        with open(candidate_tsv, "r", encoding="utf-8") as f:
            next(f)
            for i, line in enumerate(f):
                if i > 2000: break
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    cand_map[parts[0]] = parts[1].split(",") if parts[1] else []

    with open(matching_tsv, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[1]: # non-empty match
                s1_id = parts[0]
                matched_ids = parts[1].split(",")
                candidate_ids = cand_map.get(s1_id, matched_ids)
                samples.append({
                    "source1_entity_id": s1_id,
                    "matched_entity_ids": matched_ids,
                    "candidate_entity_ids": candidate_ids,
                    "candidate_count": len(candidate_ids),
                    "match_count": len(matched_ids)
                })
                if len(samples) >= limit:
                    break

    return {"samples": samples}

@app.post("/api/match_live")
def match_live(req: LiveMatchRequest):
    """
    Live pairwise entity resolution simulator.
    Normalizes text, extracts all 20 features, and runs LightGBM model scoring.
    """
    cn1 = clean_text(req.s1_name)
    ca1 = clean_text(req.s1_addr)
    cp1 = cn1.replace(" ", "")
    nums1 = extract_numbers(cn1) | extract_numbers(ca1)

    cn2 = clean_text(req.cand_name)
    ca2 = clean_text(req.cand_addr)
    cp2 = cn2.replace(" ", "")
    nums2 = extract_numbers(cn2) | extract_numbers(ca2)

    feat_vals = extract_pair_features_fast(
        (cn1, ca1, cp1, nums1),
        (cn2, ca2, cp2, nums2),
        req.cand_id,
        req.blocking_score,
        req.blocking_rank
    )

    prob = float(model.predict_proba([feat_vals])[0])
    is_match = prob >= threshold

    feat_dict = {name: round(val, 4) for name, val in zip(FEATURE_NAMES, feat_vals)}

    return {
        "probability": round(prob, 4),
        "is_match": is_match,
        "decision": "MATCH" if is_match else "NON-MATCH",
        "threshold": threshold,
        "clean_s1": {
            "name": cn1,
            "address": ca1,
            "numbers": sorted(list(nums1))
        },
        "clean_candidate": {
            "name": cn2,
            "address": ca2,
            "numbers": sorted(list(nums2))
        },
        "features": feat_dict
    }

@app.get("/api/examples")
def get_preset_examples():
    """Curated edge cases demonstrating resolution across leet-speak, landmarks, and suffixes."""
    return {
        "examples": [
            {
                "title": "Leet-Speak & Legal Suffix Noise",
                "tag": "Leet Speak",
                "s1_name": "Corner P1lates & Y0ga Stud1o LLC",
                "s1_addr": "450 7th Ave Ste 1200, New York, NY 10123",
                "cand_name": "Corner Pilates and Yoga Studio",
                "cand_addr": "450 7th Avenue, Suite 1200, NY",
                "cand_id": "S2-00109",
                "description": "Numbers '1' and '0' substituted inside words; legal suffix variation; address abbreviation ('Ave' vs 'Avenue', 'Ste' vs 'Suite')."
            },
            {
                "title": "Indian Landmark & Municipal Address",
                "tag": "India Landmark",
                "s1_name": "Balaji Electronics & Telecom Pvt Ltd",
                "s1_addr": "Opp. SBI ATM, Main Market, MG Road, Bangalore 560001",
                "cand_name": "Balaji Electronics Private Limited",
                "cand_addr": "Shop No 14, MG Rd near State Bank, Bangalore, Karnataka 560001",
                "cand_id": "S3-04921",
                "description": "Pvt Ltd expansion; landmark reference ('Opp SBI' vs 'near State Bank'); PIN code consistency (560001)."
            },
            {
                "title": "French Accents & Corporate Forms",
                "tag": "France Open-Set",
                "s1_name": "Boulangerie Pâtisserie de l'Étoile SARL",
                "s1_addr": "18 Avenue des Champs-Élysées, 75008 Paris",
                "cand_name": "Boulangerie Patisserie l Etoile",
                "cand_addr": "18 Av des Champs Elysees Paris 75008",
                "cand_id": "S2-08832",
                "description": "Unicode accent stripping ('â', 'É' -> 'a', 'E'); French corporate form ('SARL'); French street abbreviation ('Av.')."
            },
            {
                "title": "False Match Trap (Different Unit / Branch)",
                "tag": "Hard Negative",
                "s1_name": "Apex Logistics Services Corp",
                "s1_addr": "100 Industrial Parkway Unit 4B, Chicago, IL 60601",
                "cand_name": "Apex Logistics Services Inc",
                "cand_addr": "100 Industrial Parkway Unit 9C, Chicago, IL 60601",
                "cand_id": "S3-19941",
                "description": "Same parent brand and street number, but different distinct unit numbers ('4' vs '9'). Precision-weighted model penalizes numeric conflict."
            }
        ]
    }

# Mount static folder and root index
static_dir = os.path.join(BASE_DIR, "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def read_root():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.isfile(index_file):
        return FileResponse(index_file)
    return {"message": "Amazon Business Entity Resolution Studio API is running."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
