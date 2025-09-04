from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
from rapidfuzz import fuzz

from constants import (
    SYNONYMS,
    VIN_REGEX,
    EMAIL_REGEX,
    US_PHONE_REGEX,
    ZIP5_REGEX,
    DELIVERYDATE_PRECEDENCE,
    FUZZY_STRONG,
    FUZZY_CANDIDATE,
    NEGATIVE_KEYWORDS,
    EXCLUDE_OEMS,
    POSITIVE_KEYWORDS,
)


class SchemaError(Exception):
    pass


class SchemaAmbiguityError(SchemaError):
    pass


def normalize_label(label: str) -> str:
    if label is None:
        return ""
    # Lowercase, strip non-alphanumeric except spaces, collapse whitespace
    cleaned = re.sub(r"[^a-z0-9]+", " ", str(label).lower()).strip()
    return re.sub(r"\s+", " ", cleaned)


def compile_regex(pattern: str):
    return re.compile(pattern, flags=re.IGNORECASE)


VIN_RE = compile_regex(VIN_REGEX)
EMAIL_RE = compile_regex(EMAIL_REGEX)
PHONE_RE = compile_regex(US_PHONE_REGEX)
ZIP_RE = compile_regex(ZIP5_REGEX)


def sample_series_values(series: pd.Series, max_rows: int = 500) -> pd.Series:
    """Return up to max_rows non-null values for quick scanning."""
    if not isinstance(series, pd.Series):
        return pd.Series([], dtype=object)
    return series.dropna().astype(str).head(max_rows)


def _looks_like_city(series: pd.Series) -> float:
    vals = sample_series_values(series)
    if vals.empty:
        return 0.0
    # City-like: alphabetic with spaces/hyphens, not mostly digits
    def is_city(v: str) -> bool:
        v = v.strip()
        if not v:
            return False
        if re.search(r"\d", v):
            return False
        return bool(re.fullmatch(r"[A-Za-z][A-Za-z\-\s\.']+", v))
    rate = vals.apply(is_city).mean()
    return float(rate)


def _looks_like_state(series: pd.Series) -> float:
    from constants import US_STATE_ABBR
    vals = sample_series_values(series)
    if vals.empty:
        return 0.0
    abbr = {s.upper() for s in US_STATE_ABBR}
    def is_state(v: str) -> bool:
        v = v.strip().upper()
        return v in abbr
    return float(vals.apply(is_state).mean())


def _looks_like_zip(series: pd.Series) -> float:
    vals = sample_series_values(series)
    if vals.empty:
        return 0.0
    def is_zip(v: str) -> bool:
        return bool(re.fullmatch(r"\d{5}(?:-\d{4})?", v.strip()))
    return float(vals.apply(is_zip).mean())


def _oem_rate(series: pd.Series) -> float:
    vals = sample_series_values(series)
    if vals.empty:
        return 0.0
    oems = {o.upper() for o in EXCLUDE_OEMS}
    def is_oem(v: str) -> bool:
        v = v.strip().upper()
        return v in oems
    return float(vals.apply(is_oem).mean())


def _is_state_value(v: str) -> bool:
    from constants import US_STATE_ABBR
    if v is None:
        return False
    vv = str(v).strip().upper()
    return vv in US_STATE_ABBR


def _is_zip_value(v: str) -> bool:
    if v is None:
        return False
    return bool(re.fullmatch(r"\d{5}(?:-\d{4})?", str(v).strip()))


STREET_SUFFIXES = {
    "ST","STREET","RD","ROAD","AVE","AV","AVENUE","BLVD","DR","DRIVE","LN","LANE","CT","COURT","HWY","HIGHWAY","PKWY","WAY","TER","TERRACE","PL","PLACE","CIR","CIRCLE","TRL","TRAIL","LOOP",
    "BND","BEND","CV","COVE","CMN","COMMONS","SQ","SQUARE","RUN","PASS","ALY","ALLEY","XING","CROSSING","HL","HILL","HOLW","HOLLOW","MDW","MEADOW","RTE","ROUTE","VLG","VILLAGE","RIV","RIVER",
    "CRK","CREEK","GRV","GROVE","GDNS","GARDENS","IS","ISLAND","LNDG","LANDING","LK","LAKE","LGT","LIGHT","MTN","MOUNTAIN","PR","PRAIRIE","PT","POINT","RDG","RIDGE","STA","STATION","VIS","VISTA"
}


def _looks_like_street(series: pd.Series) -> float:
    vals = sample_series_values(series)
    if vals.empty:
        return 0.0
    def is_street(v: str) -> bool:
        v = v.strip().upper()
        if not v:
            return False
        # Leading number or PO BOX
        if re.match(r"^(\d+\s+|P\.?O\.?\s*BOX\b)", v):
            return True
        # Contains a known street suffix
        toks = re.split(r"[^A-Z0-9]+", v)
        return any(tok in STREET_SUFFIXES for tok in toks)
    return float(vals.apply(is_street).mean())


def header_score(canonical: str, header_norm: str) -> int:
    # Exact or synonym contains check
    synonyms = SYNONYMS.get(canonical, set())
    if header_norm in synonyms or header_norm == canonical.lower():
        return FUZZY_STRONG
    # Fuzzy against each synonym and canonical token
    pool = list(synonyms) + [canonical.replace("_", " ")]
    scores = [
        fuzz.token_set_ratio(header_norm, s) for s in pool
    ]
    base = max(scores) if scores else 0
    # Apply negative keyword penalty for misleading columns
    negatives = NEGATIVE_KEYWORDS.get(canonical, set())
    if any(tok in header_norm.split() for tok in negatives) or any(tok in header_norm for tok in negatives):
        penalty = 20
        # Stronger penalty for Store to avoid mapping fees/ids as Store
        if canonical == "Store":
            penalty = 40
        base = max(0, base - penalty)
    # Apply small boost for positive keywords
    positives = POSITIVE_KEYWORDS.get(canonical, set())
    if any(tok in header_norm for tok in positives):
        base = min(100, base + 5)
    return base


def value_pattern_score(canonical: str, series: pd.Series) -> int:
    values = sample_series_values(series)
    if values.empty:
        return 0
    if canonical == "VIN":
        # Require a meaningful fraction of 17-char VINs
        vin_hits = values.str.contains(VIN_RE).sum()
        frac = vin_hits / max(1, len(values))
        if frac >= 0.01:
            return 4
        elif vin_hits > 0:
            return 2
        return 0
    if canonical == "Email":
        hits = values.str.contains(EMAIL_RE).sum()
        return 3 if hits > 0 else 0
    if canonical in {"Home_Phone", "Mobile_Phone", "Work_Phone", "Phone2", "Phone"}:
        hits = values.str.contains(PHONE_RE).sum()
        return 2 if hits > 0 else 0
    if canonical == "Zip":
        hits = values.str.contains(ZIP_RE).sum()
        return 2 if hits > 0 else 0
    if canonical in {"Distance", "Mileage", "Delivery_Miles"}:
        # Numeric leaning
        def looks_numeric(v: str) -> bool:
            v = v.replace(",", "").replace(" ", "")
            return bool(re.fullmatch(r"-?\d+(?:\.\d+)?", v))

        nums = values.apply(looks_numeric).sum()
        return 2 if nums > 0 else 0
    if canonical == "Year":
        # Must look like plausible model year (e.g., 19xx or 20xx within range)
        try:
            parsed = pd.to_numeric(values.str.replace(r"[^0-9]", "", regex=True), errors="coerce")
            hits = parsed.between(1980, 2030).sum()
            return 3 if hits > 0 else 0
        except Exception:
            return 0
    if canonical in {"DeliveryDate"}:
        # Try pandas datetime parse success rate
        try:
            parsed = pd.to_datetime(values, errors="coerce")
            hits = parsed.notna().sum()
            return 2 if hits > 0 else 0
        except Exception:
            return 0
    # Names/Address: no strong value pattern; return 0
    return 0


def detect_schema(df: pd.DataFrame) -> Tuple[Dict[str, str], List[str]]:
    """
    Return mapping of canonical field -> source column name, and a list of warnings.
    Applies staged scoring: header match + value pattern + precedence rules.
    Allows coexistence for some roles (e.g., FullName with First/Last).
    """
    if df is None or df.empty:
        raise SchemaError("Empty DataFrame provided for schema detection")

    warnings: List[str] = []
    header_norms = {col: normalize_label(col) for col in df.columns}

    # Candidate scores per canonical: list of (source_col, score)
    candidates: Dict[str, List[Tuple[str, int]]] = {}

    for col, norm in header_norms.items():
        series = df[col]
        for canonical in SYNONYMS.keys():
            hs = header_score(canonical, norm)
            if hs >= FUZZY_CANDIDATE:
                vs = value_pattern_score(canonical, series)
                score = max(hs, FUZZY_CANDIDATE) + vs
                candidates.setdefault(canonical, []).append((col, score))

    # Identify best State/Zip by value-likeness for co-occurrence
    state_best_col = None
    state_best_rate = 0.0
    zip_best_col = None
    zip_best_rate = 0.0
    for col in df.columns:
        s = df[col]
        r_state = _looks_like_state(s)
        if r_state > state_best_rate:
            state_best_rate = r_state
            state_best_col = col
        r_zip = _looks_like_zip(s)
        if r_zip > zip_best_rate:
            zip_best_rate = r_zip
            zip_best_col = col

    state_mask = None
    zip_mask = None
    if state_best_col is not None and state_best_rate >= 0.3:
        state_mask = df[state_best_col].apply(_is_state_value)
    if zip_best_col is not None and zip_best_rate >= 0.3:
        zip_mask = df[zip_best_col].apply(_is_zip_value)

    # Value-driven candidates for address fields regardless of header names
    for col in df.columns:
        s = df[col]
        # Skip entirely numeric-like columns
        if s.dropna().astype(str).str.fullmatch(r"\s*-?\d+(?:\.\d+)?\s*").mean() > 0.8:
            continue
        street_score = _looks_like_street(s)
        if street_score >= 0.2:
            base_a1 = int(80 + street_score * 20)
            base_a2 = int(70 + street_score * 20)
            norm = header_norms.get(col, "")
            negatives_a1 = NEGATIVE_KEYWORDS.get("Address1", set())
            negatives_a2 = NEGATIVE_KEYWORDS.get("Address2", set())
            if any(tok in norm for tok in negatives_a1):
                base_a1 = max(0, base_a1 - 20)
            if any(tok in norm for tok in negatives_a2):
                base_a2 = max(0, base_a2 - 20)
            candidates.setdefault("Address1", []).append((col, base_a1))
            candidates.setdefault("Address2", []).append((col, base_a2))
        city_score = _looks_like_city(s)
        if city_score >= 0.3:
            oem_penalty = _oem_rate(s) * 60  # heavy penalty if column contains make names
            norm = header_norms.get(col, "")
            neg_city = NEGATIVE_KEYWORDS.get("City", set())
            neg_penalty = 20 if any(tok in norm for tok in neg_city) else 0
            co_rate = 0.0
            if state_mask is not None and zip_mask is not None:
                nonempty = s.fillna("").astype(str).str.strip() != ""
                co_rate = ((nonempty & state_mask & zip_mask).mean())
            uniq_ratio = s.nunique(dropna=True) / max(1, len(s))
            low_unique_penalty = 20 if uniq_ratio < 0.001 else 0
            score_city = int(max(0, (80 + city_score * 20 + co_rate * 20) - oem_penalty - neg_penalty - low_unique_penalty))
            candidates.setdefault("City", []).append((col, score_city))
        state_score = _looks_like_state(s)
        if state_score >= 0.3:
            candidates.setdefault("State", []).append((col, int(85 + state_score * 15)))
        zip_score = _looks_like_zip(s)
        if zip_score >= 0.3:
            candidates.setdefault("Zip", []).append((col, int(85 + zip_score * 15)))

    # For DeliveryDate, also consider precedence keywords even if fuzzy below threshold
    for col, norm in header_norms.items():
        if any(tok in norm for tok in DELIVERYDATE_PRECEDENCE):
            vs = value_pattern_score("DeliveryDate", df[col])
            candidates.setdefault("DeliveryDate", []).append((col, FUZZY_CANDIDATE + vs))

    mapping: Dict[str, str] = {}

    def pick_best(canon: str, pairs: List[Tuple[str, int]]):
        if not pairs:
            return None
        # Sort by score desc, then shortest header label
        pairs_sorted = sorted(pairs, key=lambda x: (-x[1], len(header_norms[x[0]])))
        best_col, best_score = pairs_sorted[0]
        # Check for true ambiguity: another candidate within 2 points for SAME role (not for coexist roles)
        coexist_ok = canon in {"FullName", "First_Name", "Last_Name", "Home_Phone", "Mobile_Phone", "Work_Phone", "Phone2"}
        if not coexist_ok and len(pairs_sorted) > 1:
            second_col, second_score = pairs_sorted[1]
            if abs(best_score - second_score) <= 2 and header_norms[best_col] != header_norms[second_col]:
                # Try value-based tie-breaker: choose column with fewer nulls and more distinct values
                s1 = df[best_col]
                s2 = df[second_col]
                s1_nonnull = s1.notna().sum()
                s2_nonnull = s2.notna().sum()
                if s1_nonnull != s2_nonnull:
                    return best_col if s1_nonnull > s2_nonnull else second_col
                s1_unique = s1.nunique(dropna=True)
                s2_unique = s2.nunique(dropna=True)
                if s1_unique != s2_unique:
                    return best_col if s1_unique > s2_unique else second_col
                raise SchemaAmbiguityError(
                    f"Ambiguity for {canon}: '{best_col}' vs '{second_col}' with close scores {best_score} vs {second_score}."
                )
        return best_col

    # Primary pass: choose best per canonical
    for canon, pairs in candidates.items():
        chosen = pick_best(canon, pairs)
        if chosen:
            mapping[canon] = chosen

    # DeliveryDate precedence: if multiple detected synonyms, prefer by DELIVERYDATE_PRECEDENCE appearance order
    if "DeliveryDate" in mapping:
        # Nothing else to do here; precedence already influenced by header token check
        pass

    # Prefer synthetic CSZ splits when present and better than chosen mapping
    def _maybe_use_csz(canon: str, like_fn, min_thresh: float) -> None:
        col = mapping.get(canon)
        csz_col = f"__CSZ_{canon}"
        try:
            if csz_col in df.columns:
                chosen_like = like_fn(df[col]) if col in df.columns else 0.0
                csz_like = like_fn(df[csz_col])
                chosen_norm = header_norms.get(col or "", "")
                is_composite_header = "city state zip" in chosen_norm
                if (is_composite_header or chosen_like < min_thresh) and csz_like >= min_thresh:
                    mapping[canon] = csz_col
                    warnings.append(f"Remapped {canon} to {csz_col} based on value-likeness ({csz_like:.2f})")
        except Exception:
            pass

    _maybe_use_csz("State", _looks_like_state, 0.7)
    _maybe_use_csz("Zip", _looks_like_zip, 0.7)
    _maybe_use_csz("City", _looks_like_city, 0.5)

    # Confidence gating for critical fields with actionable errors
    def top_candidates_str(canon: str, k: int = 3) -> str:
        pairs = candidates.get(canon, [])
        if not pairs:
            return "<none>"
        pairs_sorted = sorted(pairs, key=lambda x: -x[1])[:k]
        return ", ".join([f"{col} (score {score})" for col, score in pairs_sorted])

    def series_of(canon: str) -> Optional[pd.Series]:
        col = mapping.get(canon)
        if not col:
            return None
        try:
            return df[col]
        except Exception:
            return None

    # Address1
    addr1_s = series_of("Address1")
    if addr1_s is not None:
        street_like = _looks_like_street(addr1_s)
        # Allow lower threshold if PO BOX is common
        po_rate = 0.0
        try:
            vals = sample_series_values(addr1_s)
            po_rate = vals.str.contains(r"(?i)\bP\.?\s*O\.?\s*BOX\b|\bPO\s*BOX\b", regex=True).mean()
        except Exception:
            pass
        min_thresh = 0.2 if po_rate >= 0.2 else 0.3
        if street_like < min_thresh:
            raise SchemaError(
                "Address1 mapping is low confidence: street-likeness "
                f"{street_like:.2f}. Top candidates: {top_candidates_str('Address1')}"
            )

    # City requires co-occurrence with valid State/Zip
    city_s = series_of("City")
    if city_s is not None:
        # Guardrail: if City header clearly refers to a name column, prefer CSZ split
        try:
            chosen_col = mapping.get("City")
            chosen_norm = header_norms.get(chosen_col or "", "")
            looks_like_name_header = any(tok in chosen_norm for tok in ["first", "last", "fullname", "customer name"]) or chosen_norm in {"first name", "last name"}
            looks_like_vehicle_header = any(tok in chosen_norm for tok in ["model", "series", "trim"]) or chosen_norm in {"model"}
            eq_rate_to_last = 0.0
            if "Last_Name" in mapping and mapping["Last_Name"] in df.columns:
                try:
                    eq_rate_to_last = (df[mapping["Last_Name"]].fillna("").astype(str).str.strip() == city_s.fillna("").astype(str).str.strip()).mean()
                except Exception:
                    eq_rate_to_last = 0.0
            if looks_like_name_header or looks_like_vehicle_header or eq_rate_to_last >= 0.2:
                if "__CSZ_City" in df.columns and _looks_like_city(df["__CSZ_City"]) >= 0.3:
                    mapping["City"] = "__CSZ_City"
                    city_s = df["__CSZ_City"]
                    warnings.append("Remapped City to __CSZ_City due to name-like chosen column")
                else:
                    raise SchemaError(
                        "City mapping appears name/model-like (equality with Last_Name or header). "
                        "Provide a City column or a composite City/State/Zip that parses into __CSZ_City."
                    )
        except Exception:
            pass
        city_like = _looks_like_city(city_s)
        co_rate = 0.0
        try:
            nonempty = city_s.fillna("").astype(str).str.strip() != ""
            if state_mask is not None and zip_mask is not None:
                co_rate = (nonempty & state_mask & zip_mask).mean()
        except Exception:
            pass
        if city_like < 0.5 or co_rate < 0.3:
            raise SchemaError(
                "City mapping is low confidence: city-like "
                f"{city_like:.2f}, co-occurrence with valid State/Zip {co_rate:.2f}. "
                f"Top candidates: {top_candidates_str('City')}"
            )

    # State
    state_s = series_of("State")
    if state_s is not None:
        state_like = _looks_like_state(state_s)
        if state_like < 0.7:
            raise SchemaError(
                "State mapping is low confidence: state-like "
                f"{state_like:.2f}. Top candidates: {top_candidates_str('State')}"
            )

    # Zip
    zip_s = series_of("Zip")
    if zip_s is not None:
        zip_like = _looks_like_zip(zip_s)
        if zip_like < 0.7:
            raise SchemaError(
                "Zip mapping is low confidence: zip-like "
                f"{zip_like:.2f}. Top candidates: {top_candidates_str('Zip')}"
            )

    # VIN
    vin_s = series_of("VIN")
    if vin_s is not None:
        try:
            values = sample_series_values(vin_s)
            frac17 = values.astype(str).str.contains(VIN_RE).mean()
        except Exception:
            frac17 = 0.0
        if frac17 < 0.01:
            raise SchemaError(
                "VIN mapping is low confidence: fraction of 17-char VINs "
                f"{frac17:.2%}. Top candidates: {top_candidates_str('VIN')}"
            )

    # DeliveryDate (warn or fail)
    dd_s = series_of("DeliveryDate")
    if dd_s is not None:
        try:
            parsed = pd.to_datetime(sample_series_values(dd_s), errors="coerce")
            dd_rate = parsed.notna().mean()
        except Exception:
            dd_rate = 0.0
        if dd_rate < 0.2:
            raise SchemaError(
                "DeliveryDate mapping is low confidence: parse success rate "
                f"{dd_rate:.2%}. Top candidates: {top_candidates_str('DeliveryDate')}"
            )

    return mapping, warnings


