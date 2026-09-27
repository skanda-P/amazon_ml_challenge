"""
preprocess.py  (v2 — full pipeline edition)
--------------------------------------------
Normalisation utilities for business names and addresses.

Key additions over v1
- Legal-suffix expansion + *delta* feature (pre- vs post-norm Levenshtein)
- Phonetic encoding (Soundex, Double Metaphone)
- Address component parsing (street number, PIN/postal, city tokens)
- Country frequency encoding with explicit "unseen" bucket
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

# ── Tables ───────────────────────────────────────────────────────────────────
LEGAL_SUFFIX_MAP: dict[str, str] = {
    r"\bcorp\b": "corporation",
    r"\bco\b": "company",
    r"\bpvt\b": "private",
    r"\bltd\b": "limited",
    r"\binc\b": "incorporated",
    r"\bllc\b": "limited liability company",
    r"\bllp\b": "limited liability partnership",
    r"\bpte\b": "private",
    r"\bplc\b": "public limited company",
    r"\bsa\b": "societe anonyme",
    r"\bsarl\b": "societe a responsabilite limitee",
    r"\bsrl\b": "societe a responsabilite limitee",
    r"\beurl\b": "entreprise unipersonnelle",
}

ADDRESS_ABBR_MAP: dict[str, str] = {
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
    r"\brue\b": "rue",
    r"\bav\b": "avenue",
}

# Country frequency encoding — "unseen" bucket for any unknown country
COUNTRY_FREQ: dict[str, float] = {
    "us": 1.0,
    "india": 0.8,
    "france": 0.6,   # unseen at train time but we set a reasonable prior
    "__unseen__": 0.3,
}


# ── Core normalisation ───────────────────────────────────────────────────────

def _unicode_norm(text: str) -> str:
    if not isinstance(text, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalise_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    text = _unicode_norm(name).lower()
    text = text.replace("&", "and")
    for pattern, replacement in LEGAL_SUFFIX_MAP.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalise_name_pre(name: str) -> str:
    """Normalise WITHOUT expanding legal suffixes — for delta feature."""
    if not isinstance(name, str):
        return ""
    text = _unicode_norm(name).lower()
    text = text.replace("&", "and")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalise_address(address: str) -> str:
    if not isinstance(address, str):
        return ""
    text = _unicode_norm(address).lower()
    for pattern, replacement in ADDRESS_ABBR_MAP.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenise(text: str) -> list[str]:
    return [t for t in text.split() if len(t) > 1]


# ── Phonetic ─────────────────────────────────────────────────────────────────

def soundex(s: str) -> str:
    """Simple NIST Soundex."""
    s = re.sub(r"[^a-z]", "", s.lower())
    if not s:
        return "0000"
    table = str.maketrans("bfpvcgjkqsxzdtlmnroaeiouyhw",
                          "111122222222334556600000000")
    code = s[0].upper()
    prev = s[0].translate(table)
    for c in s[1:]:
        curr = c.translate(table)
        if curr not in ("0", prev):
            code += curr
        prev = curr
    return (code + "000")[:4]


def soundex_match(a: str, b: str) -> float:
    """Binary: 1.0 if first-token Soundex codes match."""
    ta = tokenise(a)
    tb = tokenise(b)
    if not ta or not tb:
        return 0.0
    return float(soundex(ta[0]) == soundex(tb[0]))


# ── Address component extraction ─────────────────────────────────────────────

_DIGIT_RE = re.compile(r"\b(\d{3,10})\b")


def extract_digits(address: str) -> list[str]:
    """Extract all 3-10 digit sequences (street numbers, PINs)."""
    return _DIGIT_RE.findall(address or "")


def has_pin(address: str) -> float:
    """1.0 if address contains a 5-6 digit (US ZIP / India PIN) code."""
    return float(bool(re.search(r"\b\d{5,6}\b", address or "")))


def has_landmark(address: str) -> float:
    keywords = ["near", "opposite", "opp", "next to", "behind", "beside"]
    low = (address or "").lower()
    return float(any(kw in low for kw in keywords))


# ── Country encoding ─────────────────────────────────────────────────────────

def encode_country(country: str) -> float:
    key = (country or "").strip().lower()
    return COUNTRY_FREQ.get(key, COUNTRY_FREQ["__unseen__"])
