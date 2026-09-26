import re
import numpy as np
import rapidfuzz.fuzz as fuzz
from .preprocess import extract_numbers

RE_POSTAL = re.compile(r'\b\d{5,6}\b')

LEGAL_TERMS = {
    'inc', 'corp', 'corporation', 'llp', 'ltd', 'limited', 'pvt', 'private',
    'co', 'company', 'services', 'enterprises', 'associates', 'group', 'holdings',
    'sa', 'sarl', 'sasu', 'sci', 'sas', 'eurl', 'cie', 'gmbh', 'llc', 'plc'
}

def strip_legal_suffixes(name: str) -> str:
    words = [w for w in name.split() if w not in LEGAL_TERMS]
    return " ".join(words)

def extract_postal_code(text: str) -> set:
    if not text:
        return set()
    return set(RE_POSTAL.findall(text))

def is_latin_text(s: str) -> bool:
    try:
        s.encode('latin-1')
        return True
    except UnicodeEncodeError:
        return False

def _char_3grams(s: str) -> set:
    if len(s) < 3:
        return {s} if s else set()
    return {s[i:i+3] for i in range(len(s) - 2)}

FEATURE_NAMES = [
    'name_ratio',
    'name_token_sort_ratio',
    'name_token_set_ratio',
    'name_partial_ratio',
    'name_jaccard',
    'name_exact_clean',
    'name_len_diff',
    'name_domain_match',
    'addr_token_set_ratio',
    'addr_jaccard',
    'addr_len_diff',
    'addr_empty_s1',
    'addr_empty_c',
    'both_addr_present',
    'num_common',
    'num_jaccard',
    'num_conflict',
    'blocking_score',
    'blocking_rank',
    'is_s2',
    'name_char_3gram_jaccard',
    'name_prefix_match',
    'first_word_match',
    'name_containment',
    'length_ratio',
    'addr_ratio',
    'addr_num_exact',
    'token_count_diff',
    'name_legal_stem_exact',
    'name_legal_stem_ratio',
    'addr_char_3gram_jaccard',
    'postal_code_match',
    'first_two_words_match',
    'exact_name_and_addr',
    'is_latin_disjoint',
    'high_name_no_addr'
]

def extract_pair_features_fast(s1_item, c_item, cid: str, blocking_score: float, blocking_rank: int):
    """
    Ultra-fast pairwise feature extraction from pre-cleaned representations.
    s1_item: (cn1, ca1, cp1, nums1)
    c_item:  (cn2, ca2) or (cn2, ca2, cp2, nums2)
    """
    cn1, ca1, cp1, nums1 = s1_item
    if len(c_item) == 2:
        cn2, ca2 = c_item
        cp2 = cn2.replace(" ", "")
        nums2 = extract_numbers(cn2) | extract_numbers(ca2)
    else:
        cn2, ca2, cp2, nums2 = c_item

    words1 = cn1.split()
    words2 = cn2.split()

    # 1. Name fuzzy similarities
    name_ratio = fuzz.ratio(cn1, cn2)
    name_token_sort_ratio = fuzz.token_sort_ratio(cn1, cn2)
    name_token_set_ratio = fuzz.token_set_ratio(cn1, cn2)
    name_partial_ratio = fuzz.partial_ratio(cn1, cn2)

    # Word-level Jaccard & Containment
    t1 = set(words1)
    t2 = set(words2)
    union_n = len(t1 | t2)
    name_jaccard = len(t1 & t2) / union_n if union_n > 0 else 0.0
    name_containment = len(t1 & t2) / len(t1) if t1 else 0.0

    name_exact_clean = 1.0 if cn1 == cn2 and len(cn1) > 0 else 0.0
    name_len_diff = float(abs(len(cn1) - len(cn2)))

    # Domain / compact match
    if len(cp1) >= 4 and len(cp2) >= 4 and (cp1 in cp2 or cp2 in cp1):
        name_domain_match = 1.0
    else:
        name_domain_match = 0.0

    # 2. Address similarities
    addr_empty_s1 = 1.0 if not ca1 else 0.0
    addr_empty_c = 1.0 if not ca2 else 0.0
    both_addr_present = 1.0 if (not addr_empty_s1 and not addr_empty_c) else 0.0

    if both_addr_present:
        addr_token_set_ratio = float(fuzz.token_set_ratio(ca1, ca2))
        addr_ratio = float(fuzz.ratio(ca1, ca2))
        at1 = set(ca1.split())
        at2 = set(ca2.split())
        union_a = len(at1 | at2)
        addr_jaccard = len(at1 & at2) / union_a if union_a > 0 else 0.0
        addr_len_diff = float(abs(len(ca1) - len(ca2)))

        ga1 = _char_3grams(ca1)
        ga2 = _char_3grams(ca2)
        u_ga = len(ga1 | ga2)
        addr_char_3gram_jaccard = len(ga1 & ga2) / u_ga if u_ga > 0 else 0.0

        post1 = extract_postal_code(ca1)
        post2 = extract_postal_code(ca2)
        if post1 and post2:
            postal_code_match = 1.0 if (post1 & post2) else -1.0
        else:
            postal_code_match = 0.0
    else:
        addr_token_set_ratio = 0.0
        addr_ratio = 0.0
        addr_jaccard = 0.0
        addr_len_diff = 0.0
        addr_char_3gram_jaccard = 0.0
        postal_code_match = 0.0

    # 3. Numeric consistency
    common_nums = nums1 & nums2
    num_common = float(len(common_nums))
    union_nums = len(nums1 | nums2)
    num_jaccard = num_common / union_nums if union_nums > 0 else 0.0
    num_conflict = 1.0 if (nums1 and nums2 and not common_nums) else 0.0
    addr_num_exact = 1.0 if (nums1 and nums1.issubset(nums2)) else 0.0

    # 4. Source & Blocking features
    is_s2 = 1.0 if cid.startswith('S2-') else 0.0

    # 5. Advanced discriminative signals
    g1 = _char_3grams(cn1)
    g2 = _char_3grams(cn2)
    u_g = len(g1 | g2)
    name_char_3gram_jaccard = len(g1 & g2) / u_g if u_g > 0 else 0.0

    name_prefix_match = 1.0 if len(cn1) >= 4 and len(cn2) >= 4 and cn1[:4] == cn2[:4] else 0.0
    first_word_match = 1.0 if words1 and words2 and words1[0] == words2[0] else 0.0

    max_l = max(len(cn1), len(cn2))
    length_ratio = min(len(cn1), len(cn2)) / max_l if max_l > 0 else 0.0
    token_count_diff = float(abs(len(words1) - len(words2)))

    # 6. Legal entity stem & exact compound matches
    stem1 = strip_legal_suffixes(cn1)
    stem2 = strip_legal_suffixes(cn2)
    name_legal_stem_exact = 1.0 if stem1 and stem2 and stem1 == stem2 else 0.0
    name_legal_stem_ratio = float(fuzz.token_sort_ratio(stem1, stem2)) if stem1 and stem2 else float(name_token_sort_ratio)

    first_two_words_match = 1.0 if (len(words1) >= 2 and len(words2) >= 2 and words1[:2] == words2[:2]) else 0.0
    exact_name_and_addr = 1.0 if (cn1 == cn2 and ca1 == ca2 and len(cn1) > 0) else 0.0

    # 7. Disjoint name / missing address discriminators
    is_latin_disjoint = 1.0 if (is_latin_text(cn1) and is_latin_text(cn2) and name_token_set_ratio < 45 and not name_domain_match) else 0.0
    high_name_no_addr = 1.0 if (addr_empty_c == 1.0 and (name_legal_stem_exact == 1.0 or name_token_sort_ratio >= 90)) else 0.0

    return [
        float(name_ratio),
        float(name_token_sort_ratio),
        float(name_token_set_ratio),
        float(name_partial_ratio),
        name_jaccard,
        name_exact_clean,
        name_len_diff,
        name_domain_match,
        addr_token_set_ratio,
        addr_jaccard,
        addr_len_diff,
        addr_empty_s1,
        addr_empty_c,
        both_addr_present,
        num_common,
        num_jaccard,
        num_conflict,
        float(blocking_score),
        float(blocking_rank),
        is_s2,
        name_char_3gram_jaccard,
        name_prefix_match,
        first_word_match,
        name_containment,
        length_ratio,
        addr_ratio,
        addr_num_exact,
        token_count_diff,
        name_legal_stem_exact,
        name_legal_stem_ratio,
        addr_char_3gram_jaccard,
        postal_code_match,
        first_two_words_match,
        exact_name_and_addr,
        is_latin_disjoint,
        high_name_no_addr
    ]

# Backward compatibility alias
def extract_pair_features(s1_tuple, c_tuple, blocking_score: float, blocking_rank: int):
    from .preprocess import clean_text, extract_numbers
    import re
    s1_name, s1_addr = s1_tuple
    c_name, c_addr, cid = c_tuple
    cn1 = clean_text(s1_name)
    cn2 = clean_text(c_name)
    ca1 = clean_text(s1_addr)
    ca2 = clean_text(c_addr)
    cp1 = re.sub(r'\s+', '', cn1)
    cp2 = re.sub(r'\s+', '', cn2)
    nums1 = extract_numbers(s1_name) | extract_numbers(s1_addr)
    nums2 = extract_numbers(c_name) | extract_numbers(c_addr)
    return extract_pair_features_fast(
        (cn1, ca1, cp1, nums1),
        (cn2, ca2, cp2, nums2),
        cid,
        blocking_score,
        blocking_rank
    )
