"""
Data Preprocessing and Text Normalization Module.
Handles noise patterns: character substitution (leet speak), accents/diacritics,
legal suffix variations, domain name extraction, and token extraction.
"""

import re
import unicodedata

# Comprehensive multilingual legal and address stop words
STOP_WORDS = {
    # English legal suffixes
    'inc', 'corp', 'corporation', 'llp', 'ltd', 'limited', 'pvt', 'private',
    'co', 'company', 'services', 'enterprises', 'associates', 'group', 'holdings',
    'foundation', 'trust', 'society', 'club', 'international', 'center', 'centre',
    # French legal suffixes
    'sarl', 'sasu', 'sci', 'sa', 'sas', 'eurl', 'fils', 'freres', 'cie', 'societe',
    # Common stop words
    'and', 'the', 'of', 'in', 'at', 'on', 'for', 'by', 'et', 'du', 'de', 'des', 'la', 'le',
    # Common address words
    'road', 'rd', 'street', 'st', 'ave', 'avenue', 'drive', 'dr', 'court', 'ct',
    'lane', 'ln', 'boulevard', 'blvd', 'way', 'place', 'pl', 'highway', 'hwy',
    'floor', 'flr', 'block', 'blk', 'near', 'opp', 'opposite', 'behind', 'beside',
    'plot', 'unit', 'suite', 'ste', 'room', 'rm', 'office', 'off', 'hno', 'house',
    'building', 'bldg', 'nagar', 'colony', 'marg', 'gali', 'rasta', 'chowk',
    'rue', 'allee', 'passage', 'quai', 'cours'
}

# Pre-compiled regular expressions for high-throughput text processing
RE_LEET_0 = re.compile(r'(?<=[a-zA-Z])0|0(?=[a-zA-Z])')
RE_LEET_1 = re.compile(r'(?<=[a-zA-Z])1|1(?=[a-zA-Z])')
RE_LEET_3 = re.compile(r'(?<=[a-zA-Z])3|3(?=[a-zA-Z])')
RE_LEET_5 = re.compile(r'(?<=[a-zA-Z])5|5(?=[a-zA-Z])')
RE_LEET_AT = re.compile(r'(?<=[a-zA-Z])@|@(?=[a-zA-Z])')
RE_DOMAIN = re.compile(r'\.(com|org|net|in|co|gov|edu|fr|io|biz|info)\b')
RE_PUNCT = re.compile(r'[^a-z0-9\s]')
RE_WHITESPACE = re.compile(r'\s+')
RE_DIGITS = re.compile(r'\b\d{1,8}\b')
RE_WORD_3 = re.compile(r'[a-z0-9]{3,}')

def clean_text(text: str) -> str:
    """Normalize text: handle leet speak, diacritics, web domains, and punctuation."""
    if not text:
        return ""
    # 1. Leet-speak substitution inside or adjacent to alphabetic characters
    text = RE_LEET_0.sub('o', text)
    text = RE_LEET_1.sub('l', text)
    text = RE_LEET_3.sub('e', text)
    text = RE_LEET_5.sub('s', text)
    text = RE_LEET_AT.sub('a', text)

    # 2. Unicode normalization: strip accents / diacritics (e.g. é -> e, ä -> a)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii').lower()

    # 3. Clean web domain suffixes (e.g., .com, .org, .net, .in, .fr)
    text = RE_DOMAIN.sub(' ', text)

    # 4. Strip punctuation and symbols, normalize whitespace
    text = RE_PUNCT.sub(' ', text)
    return RE_WHITESPACE.sub(' ', text).strip()

def extract_numbers(text: str) -> set:
    """Extract numeric tokens (e.g., street numbers, PIN codes, office numbers)."""
    if not text:
        return set()
    matches = RE_DIGITS.findall(text)
    return {str(int(m)) for m in matches if m.isdigit()}

def make_tfidf_doc(clean_name: str, clean_addr: str) -> str:
    """Create multi-resolution document for TF-IDF vectorization with char 3-grams and numeric anchors."""
    compact = clean_name.replace(" ", "")
    # Do not index generic legal/address words.  Besides adding little matching
    # signal they make sparse cosine products nearly dense (e.g. every Indian
    # address sharing "road"/"nagar"), which can exhaust memory at inference.
    name_terms = [w for w in clean_name.split() if w not in STOP_WORDS]
    addr_terms = [w for w in clean_addr.split() if w not in STOP_WORDS]
    parts = [" ".join(name_terms), " ".join(addr_terms)]
    if len(compact) >= 5:
        parts.extend([compact, compact])
    
    # Add distinctive numeric anchor tokens
    nums = extract_numbers(clean_name) | extract_numbers(clean_addr)
    for n in nums:
        parts.append(f"num_{n}")

    # Add character 3-grams for words >= 4 chars to guarantee typo tolerance in blocking
    words = clean_name.split()
    for w in words:
        if len(w) >= 4:
            for i in range(len(w) - 2):
                parts.append(w[i:i+3])

    return " ".join(parts)

def extract_tokens(name: str, addr: str):
    """
    Extract informative tokens for blocking and indexing.
    Returns:
        name_words: list of clean name tokens
        addr_words: list of clean address tokens
        numbers: set of normalized numeric tokens
        name_compact: concatenated name without spaces (for domain matching)
    """
    clean_name = clean_text(name)
    clean_addr = clean_text(addr)

    name_words = [w for w in RE_WORD_3.findall(clean_name) if w not in STOP_WORDS]
    addr_words = [w for w in RE_WORD_3.findall(clean_addr) if w not in STOP_WORDS]

    numbers = extract_numbers(clean_name) | extract_numbers(clean_addr)
    name_compact = clean_name.replace(" ", "")

    return name_words, addr_words, numbers, name_compact
