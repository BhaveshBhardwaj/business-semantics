import os
import sys
import pickle
import time
from collections import defaultdict
from src.preprocess import clean_text

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
gt_path = os.path.join(project_root, "dataset", "train", "train_ground_truth.tsv")
s1_path = os.path.join(project_root, "dataset", "train", "train_source1.tsv")
s2_path = os.path.join(project_root, "dataset", "train", "train_source2.tsv")
s3_path = os.path.join(project_root, "dataset", "train", "train_source3.tsv")

print("Loading GT...")
gt_map = {}
with open(gt_path, 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\n').split('\t')
        s1 = p[0]
        cands = set(p[1].split(',')) if len(p) >= 2 and p[1].strip() else set()
        gt_map[s1] = cands

s1_names = {}
s1_addrs = {}
with open(s1_path, 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\n').split('\t')
        s1_names[p[0]] = clean_text(p[1])
        s1_addrs[p[0]] = clean_text(p[2])

cand_names = {}
cand_addrs = {}
for path in [s2_path, s3_path]:
    with open(path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split('\t')
            cand_names[p[0]] = clean_text(p[1])
            cand_addrs[p[0]] = clean_text(p[2])

missed_w3 = 0
missed_w2 = 0
missed_c3 = 0
total_pos = 0

for s1, cands in gt_map.items():
    if not cands: continue
    s1_nw = set(w for w in s1_names.get(s1, '').split() if len(w) >= 3)
    s1_nw2 = set(w for w in s1_names.get(s1, '').split() if len(w) >= 2)
    s1_c3 = set(s1_names.get(s1, '').replace(" ","")[i:i+3] for i in range(len(s1_names.get(s1, '').replace(" ",""))-2))
    
    for c in cands:
        total_pos += 1
        c_nw = set(w for w in cand_names.get(c, '').split() if len(w) >= 3)
        c_nw2 = set(w for w in cand_names.get(c, '').split() if len(w) >= 2)
        c_c3 = set(cand_names.get(c, '').replace(" ","")[i:i+3] for i in range(len(cand_names.get(c, '').replace(" ",""))-2))
        
        if not (s1_nw & c_nw):
            missed_w3 += 1
        if not (s1_nw2 & c_nw2):
            missed_w2 += 1
        if not (s1_c3 & c_c3):
            missed_c3 += 1

print(f"Total pos: {total_pos}")
print(f"Positives sharing NO name words >= 3 chars: {missed_w3} ({missed_w3/total_pos*100:.2f}%)")
print(f"Positives sharing NO name words >= 2 chars: {missed_w2} ({missed_w2/total_pos*100:.2f}%)")
print(f"Positives sharing NO name char 3-grams: {missed_c3} ({missed_c3/total_pos*100:.2f}%)")
