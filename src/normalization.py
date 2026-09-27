# src/normalization.py
import re
import unicodedata
from typing import Dict, List, Any
import pandas as pd

# Legal suffixes mapping
LEGAL_SUFFIXES_MAP = {
    r"\bcorp(oration)?\b": "corporation",
    r"\binc(orporated)?\b": "incorporated",
    r"\bltd\b": "limited",
    r"\blimited\b": "limited",
    r"\bpvt\b": "private",
    r"\bprivate\b": "private",
    r"\bllc\b": "llc",
    r"\bllp\b": "llp",
    r"\bco(mpany)?\b": "company",
    r"\bgmbh\b": "gmbh",
    r"\bsa\b": "sa",
    r"\bsarl\b": "sarl",
    r"\bsas\b": "sas",
    r"\bplc\b": "plc",
}

ADDRESS_ABBREVIATIONS = {
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave?\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bct\b": "court",
    r"\bhwy\b": "highway",
    r"\bapt\b": "apartment",
    r"\bste\b": "suite",
    r"\bfl\b": "floor",
    r"\bbldg\b": "building",
    r"\bno\b": "number",
    r"\bnr\b": "near",
    r"\bopp\b": "opposite",
}

def unicode_to_ascii(text: str) -> str:
    """Converts accents and unicode characters to ASCII representations."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))

def normalize_country(country_str: str) -> str:
    """
    Normalizes country label without restricting to closed categories.
    Preserves open-set country representations (US, India, France, etc.).
    """
    if not country_str:
        return "unknown"
    clean = unicode_to_ascii(country_str).lower().strip()
    clean = re.sub(r"[^\w\s]", "", clean)
    clean = re.sub(r"\s+", " ", clean)
    
    country_map = {
        "usa": "united states",
        "us": "united states",
        "united states of america": "united states",
        "in": "india",
        "ind": "india",
        "fr": "france",
        "fra": "france",
    }
    return country_map.get(clean, clean)

def normalize_business_name(name_raw: str) -> Dict[str, Any]:
    """
    Generates rich multi-field representations for a business name.
    """
    if not name_raw or not isinstance(name_raw, str):
        name_raw = ""

    ascii_name = unicode_to_ascii(name_raw).lower().strip()
    name_amp = re.sub(r"\b&\b|\bamp\b", " and ", ascii_name)
    clean_punct = re.sub(r"[^\w\s]", " ", name_amp)
    normalized = re.sub(r"\s+", " ", clean_punct).strip()
    alphanumeric = re.sub(r"\W+", "", normalized)
    
    tokens = [t for t in normalized.split() if len(t) > 0]
    
    non_legal_tokens = []
    for t in tokens:
        is_suffix = False
        for pat in LEGAL_SUFFIXES_MAP:
            if re.match(pat, t):
                is_suffix = True
                break
        if not is_suffix:
            non_legal_tokens.append(t)
            
    name_without_legal_suffix = " ".join(non_legal_tokens)
    sorted_tokens = " ".join(sorted(tokens))
    
    ngrams = []
    padded = f"^{normalized}$"
    for i in range(len(padded) - 2):
        ngrams.append(padded[i:i+3])

    return {
        "name_raw": name_raw,
        "name_normalized": normalized,
        "name_alphanumeric": alphanumeric,
        "name_tokens": tokens,
        "name_sorted_tokens": sorted_tokens,
        "name_without_legal_suffix": name_without_legal_suffix if name_without_legal_suffix else normalized,
        "name_ngrams": ngrams,
    }

def normalize_address(address_raw: str) -> Dict[str, Any]:
    """
    Generates rich multi-field representations and extracted components for an address.
    """
    if not address_raw or not isinstance(address_raw, str):
        address_raw = ""

    ascii_addr = unicode_to_ascii(address_raw).lower().strip()
    expanded = ascii_addr
    for pat, repl in ADDRESS_ABBREVIATIONS.items():
        expanded = re.sub(pat, repl, expanded)

    clean_punct = re.sub(r"[^\w\s]", " ", expanded)
    normalized = re.sub(r"\s+", " ", clean_punct).strip()

    numeric_tokens = re.findall(r"\b\d+\b", normalized)
    postal_codes = [n for n in numeric_tokens if len(n) in (5, 6)]
    postal_code = postal_codes[0] if postal_codes else ""

    alphanumeric = re.sub(r"\W+", "", normalized)
    tokens = [t for t in normalized.split() if len(t) > 0]

    return {
        "address_raw": address_raw,
        "address_normalized": normalized,
        "address_alphanumeric": alphanumeric,
        "address_tokens": tokens,
        "numeric_tokens": numeric_tokens,
        "postal_code": postal_code,
    }

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies normalization across a DataFrame and appends clean columns.
    """
    df = df.copy()

    df["country_norm"] = df["country"].apply(normalize_country)

    name_dicts = df["business_name"].apply(normalize_business_name)
    df["name_norm"] = [d["name_normalized"] for d in name_dicts]
    df["name_alpha"] = [d["name_alphanumeric"] for d in name_dicts]
    df["name_sorted"] = [d["name_sorted_tokens"] for d in name_dicts]
    df["name_no_legal"] = [d["name_without_legal_suffix"] for d in name_dicts]
    df["name_tokens"] = [d["name_tokens"] for d in name_dicts]

    addr_dicts = df["business_address"].apply(normalize_address)
    df["address_norm"] = [d["address_normalized"] for d in addr_dicts]
    df["address_alpha"] = [d["address_alphanumeric"] for d in addr_dicts]
    df["numeric_tokens"] = [d["numeric_tokens"] for d in addr_dicts]
    df["postal_code"] = [d["postal_code"] for d in addr_dicts]

    return df