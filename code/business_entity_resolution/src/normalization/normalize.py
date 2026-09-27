"""
Stage B -- Multi-view normalization.

For every business name and address we keep MULTIPLE representations
rather than collapsing to a single canonical string, because different
representations are useful for different blocking channels / features,
and over-aggressive canonicalization can destroy discriminating
signal. Raw fields are always retained alongside normalized ones.

Nothing here is country-specific in a hard-coded, exclusionary way:
the legal-suffix and abbreviation maps are *aids*, not gates -- an
unseen country's business names still get lower-cased, tokenized,
n-grammed, etc. even if none of the suffix maps fire.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Legal-suffix / common-abbreviation maps. These are *hints* learned from
# common English-language business-register conventions (not looked up from
# any external database at runtime) -- both raw and expanded forms are kept.
# --------------------------------------------------------------------------
LEGAL_SUFFIX_MAP = {
    "corp": "corporation", "co": "company", "ltd": "limited", "pvt": "private",
    "inc": "incorporated", "llc": "limited liability company", "llp": "limited liability partnership",
    "gmbh": "gesellschaft mit beschrankter haftung", "sarl": "societe a responsabilite limitee",
    "plc": "public limited company", "assn": "association", "intl": "international",
    "mfg": "manufacturing", "svcs": "services", "svc": "service", "grp": "group",
    "bros": "brothers", "dept": "department", "ent": "enterprises", "enterp": "enterprises",
}

ADDR_ABBR_MAP = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "ln": "lane", "dr": "drive", "ct": "court", "cir": "circle",
    "hwy": "highway", "apt": "apartment", "fl": "floor", "flr": "floor", "bldg": "building",
    "sq": "square", "ngr": "nagar", "colny": "colony", "twp": "township", "stn": "station",
    "opp": "opposite", "nr": "near", "no": "number", "sec": "sector", "blk": "block",
}

LANDMARK_PATTERNS = [
    r"\bnear\b.*", r"\bopp(?:osite)?\b.*", r"\bbehind\b.*", r"\bnext to\b.*",
    r"\bbeside\b.*", r"\bin front of\b.*",
]

WS_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
NUM_RE = re.compile(r"\d+")
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _unicode_normalize(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def _collapse_ws(s: str) -> str:
    return WS_RE.sub(" ", s).strip()


def _strip_punct(s: str) -> str:
    s = s.replace("&", " and ")
    return PUNCT_RE.sub(" ", s)


def char_ngrams(s: str, n: int) -> set:
    s = s.replace(" ", "")
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


@dataclass
class NameViews:
    raw: str
    unicode_norm: str
    lower: str
    no_punct: str
    ws_norm: str
    no_suffix: str
    tokens: tuple
    alnum: str
    ngrams3: frozenset
    ngrams4: frozenset
    ngrams5: frozenset


def normalize_name(name) -> NameViews:
    if not isinstance(name, str) or not name.strip():
        return NameViews("", "", "", "", "", "", tuple(), "", frozenset(), frozenset(), frozenset())
    raw = name
    uni = _unicode_normalize(raw)
    low = uni.lower()
    nop = _strip_punct(low)
    ws = _collapse_ws(nop)
    tokens_raw = ws.split(" ") if ws else []
    expanded_tokens = [LEGAL_SUFFIX_MAP.get(t, t) for t in tokens_raw]
    # "no_suffix" view: drop trailing legal-suffix tokens entirely (both the
    # abbreviation and any already-expanded form), to expose the core name.
    suffix_tokens = set(LEGAL_SUFFIX_MAP.keys()) | set(LEGAL_SUFFIX_MAP.values())
    core_tokens = [t for t in expanded_tokens if t not in suffix_tokens]
    no_suffix = " ".join(core_tokens) if core_tokens else ws
    alnum = "".join(TOKEN_RE.findall(ws))
    return NameViews(
        raw=raw, unicode_norm=uni, lower=low, no_punct=nop, ws_norm=ws,
        no_suffix=no_suffix, tokens=tuple(expanded_tokens), alnum=alnum,
        ngrams3=frozenset(char_ngrams(ws, 3)),
        ngrams4=frozenset(char_ngrams(ws, 4)),
        ngrams5=frozenset(char_ngrams(ws, 5)),
    )


@dataclass
class AddressViews:
    raw: str
    normalized: str
    tokens: tuple
    ngrams3: frozenset
    numeric_tokens: tuple
    house_number: str
    postal_code: str
    street_tokens: tuple
    locality_guess: str
    landmark_stripped: str


def normalize_address(addr) -> AddressViews:
    if not isinstance(addr, str) or not addr.strip():
        return AddressViews("", "", tuple(), frozenset(), tuple(), "", "", tuple(), "", "")
    raw = addr
    uni = _unicode_normalize(raw)
    low = uni.lower()
    low = low.replace(",", " , ")
    # strip landmark phrases into a separate "clean" view, but keep raw too
    landmark_stripped = low
    for pat in LANDMARK_PATTERNS:
        landmark_stripped = re.sub(pat, " ", landmark_stripped)
    nop = _strip_punct(landmark_stripped)
    ws = _collapse_ws(nop)
    raw_tokens = ws.split(" ") if ws else []
    tokens = tuple(ADDR_ABBR_MAP.get(t, t) for t in raw_tokens)

    numeric_tokens = tuple(NUM_RE.findall(ws))
    # postal code heuristic: longest numeric token of length >= 4 (covers US
    # 5-digit ZIP and Indian 6-digit PIN); if none, empty.
    postal_candidates = [t for t in numeric_tokens if len(t) >= 4]
    postal_code = max(postal_candidates, key=len) if postal_candidates else ""
    # house number heuristic: first numeric token in the string that is NOT
    # the postal code (usually appears earliest in the address).
    house_number = ""
    for t in numeric_tokens:
        if t != postal_code:
            house_number = t
            break

    # street tokens: alphabetic tokens preceding the first comma or the
    # first 6 tokens, excluding pure numerics -- a light heuristic since we
    # have no gazetteer / geocoding available (and are not allowed to use one).
    street_tokens = tuple(t for t in tokens[:8] if not t.isdigit())

    # locality guess: token(s) after the last comma-separated segment before
    # country/postal, else empty. Very light heuristic, used only as a soft
    # feature, never as a hard filter.
    segments = [s.strip() for s in _collapse_ws(nop).split(" , ") if s.strip()]
    locality_guess = segments[-2] if len(segments) >= 2 else (segments[0] if segments else "")

    normalized = ws
    return AddressViews(
        raw=raw, normalized=normalized, tokens=tuple(t for t in tokens if t not in (",",)),
        ngrams3=frozenset(char_ngrams(normalized, 3)), numeric_tokens=numeric_tokens,
        house_number=house_number, postal_code=postal_code, street_tokens=street_tokens,
        locality_guess=locality_guess, landmark_stripped=landmark_stripped,
    )


def prepare_dataframe(df):
    """Vectorized-ish application of normalize_name / normalize_address /
    normalize_country over a records dataframe, returning a new dataframe
    of flattened views ready for blocking + feature engineering.
    Cached per-record so downstream stages never recompute normalization.
    """
    import pandas as pd

    names = df["business_name"].fillna("")
    addrs = df["business_address"].fillna("")
    countries = df["country"].fillna("")

    nviews = names.map(normalize_name)
    aviews = addrs.map(normalize_address)
    cviews = countries.map(normalize_country)

    out = pd.DataFrame({
        "entity_id": df["entity_id"].values,
        "source": df["source"].values,
        "country_norm": cviews.values,
        "name_raw": names.values,
        "name_ws": [v.ws_norm for v in nviews],
        "name_nosuffix": [v.no_suffix for v in nviews],
        "name_alnum": [v.alnum for v in nviews],
        "name_tokens": [frozenset(v.tokens) for v in nviews],
        "name_ngrams3": [v.ngrams3 for v in nviews],
        "addr_raw": addrs.values,
        "addr_ws": [v.normalized for v in aviews],
        "addr_tokens": [frozenset(v.tokens) for v in aviews],
        "addr_ngrams3": [v.ngrams3 for v in aviews],
        "house_number": [v.house_number for v in aviews],
        "postal_code": [v.postal_code for v in aviews],
        "street_tokens": [frozenset(v.street_tokens) for v in aviews],
        "locality_guess": [v.locality_guess for v in aviews],
        "numeric_tokens": [frozenset(v.numeric_tokens) for v in aviews],
    })
    return out


def normalize_country(country) -> str:
    """Country is treated as an open-set string label -- lower/strip only,
    never mapped against a fixed allow-list, so unseen labels (e.g. France
    in the test set) pass through unchanged in shape."""
    if not isinstance(country, str):
        return ""
    return _collapse_ws(_unicode_normalize(country).lower())
