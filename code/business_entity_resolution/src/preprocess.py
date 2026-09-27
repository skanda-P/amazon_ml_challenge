"""
preprocess.py
-------------
Text normalisation utilities for business names and addresses.
All functions are stateless and safe to call in parallel.
"""

import re
import unicodedata

# ── Legal suffix expansions ─────────────────────────────────────────────────
LEGAL_SUFFIX_MAP = {
    r"\bcorp\b": "corporation",
    r"\bco\b": "company",
    r"\bpvt\b": "private",
    r"\bltd\b": "limited",
    r"\binc\b": "incorporated",
    r"\bllc\b": "limited liability company",
    r"\bllp\b": "limited liability partnership",
}

# ── Street / address abbreviation expansions ────────────────────────────────
ADDRESS_ABBR_MAP = {
    r"\bst\b": "street",
    r"\brd\b": "road",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bct\b": "court",
    r"\bpl\b": "place",
    r"\bsq\b": "square",
    r"\bfwy\b": "freeway",
    r"\bhwy\b": "highway",
    r"\bpkwy\b": "parkway",
    r"\bexpy\b": "expressway",
    r"\bnagar\b": "nagar",
    r"\bcolony\b": "colony",
}


def unicode_normalise(text: str) -> str:
    """Decompose unicode, strip diacritics, re-encode as ASCII."""
    if not isinstance(text, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalise_name(name: str) -> str:
    """
    Normalise a business name for comparison.
    Steps: unicode → lowercase → expand legal suffixes → strip punctuation →
           collapse whitespace.
    """
    if not isinstance(name, str):
        return ""
    text = unicode_normalise(name).lower()
    # replace & with 'and'
    text = text.replace("&", "and")
    # expand legal suffixes
    for pattern, replacement in LEGAL_SUFFIX_MAP.items():
        text = re.sub(pattern, replacement, text)
    # remove punctuation except spaces
    text = re.sub(r"[^\w\s]", " ", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalise_address(address: str) -> str:
    """
    Normalise a business address for comparison.
    Steps: unicode → lowercase → expand abbreviations → strip punctuation →
           collapse whitespace.
    """
    if not isinstance(address, str):
        return ""
    text = unicode_normalise(address).lower()
    for pattern, replacement in ADDRESS_ABBR_MAP.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenise(text: str) -> list[str]:
    """Split normalised text into tokens, dropping single-char tokens."""
    return [t for t in text.split() if len(t) > 1]
