from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

from constants import (
    PRESETS,
    VIN_REGEX,
    VIN_EXPLOSION_SYNONYMS,
    VIN_EXPLOSION_DELIMITERS,
    EXCLUDE_BRANDS,
    EXCLUDE_OEMS,
    EXCLUDE_KEYWORDS,
    CORPORATE_SUFFIXES,
)
from schema_detection import normalize_label


VIN_RE = re.compile(VIN_REGEX, flags=re.IGNORECASE)


def _safe_str(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def _effective_date_series(df: pd.DataFrame) -> pd.Series:
    d = pd.to_datetime(df.get("DeliveryDate"), errors="coerce")
    for col in ["SoldDate", "SaleDate", "Last_Date"]:
        if col in df.columns:
            d = d.combine_first(pd.to_datetime(df[col], errors="coerce"))
    return d


def find_vin_explosion_column(df: pd.DataFrame) -> Optional[str]:
    normed = {col: normalize_label(col) for col in df.columns}
    candidates = []
    for col, norm in normed.items():
        if norm in VIN_EXPLOSION_SYNONYMS:
            candidates.append(col)
    if not candidates:
        return None
    # Prefer shortest normalized label
    return sorted(candidates, key=lambda c: len(normed[c]))[0]


def explode_vins_on_raw(df: pd.DataFrame, vin_col: Optional[str], vin_list_col: Optional[str]) -> pd.DataFrame:
    """
    If vin_list_col present, explode rows by the list of VIN-like tokens.
    Union with a single VIN in vin_col when present. Drop rows with no valid VINs.
    """
    if vin_list_col is None:
        return df

    records: List[pd.Series] = []
    for idx, row in df.iterrows():
        tokens: List[str] = []
        cell = row[vin_list_col]
        if pd.notna(cell):
            text = str(cell)
            # Split by known delimiters
            for d in VIN_EXPLOSION_DELIMITERS:
                text = text.replace(d, " ")
            parts = [p.strip().upper() for p in text.split() if p.strip()]
            # Keep only VIN-like
            tokens.extend([p for p in parts if VIN_RE.fullmatch(p)])
        # Union with single VIN if present
        if vin_col is not None and pd.notna(row[vin_col]):
            v = str(row[vin_col]).strip().upper()
            if VIN_RE.fullmatch(v):
                tokens.append(v)
        tokens = sorted(set(tokens))
        if not tokens:
            # No usable VINs; drop row silently
            continue
        for v in tokens:
            new_row = row.copy()
            if vin_col is not None:
                new_row[vin_col] = v
            records.append(new_row)

    if not records:
        # All dropped → return empty DataFrame with same columns
        return df.iloc[0:0].copy()
    return pd.DataFrame.from_records(records, columns=df.columns)


def _normalize_address_key(a1: str, city: str, state: str, z: str) -> str:
    def norm(s: str) -> str:
        if s is None:
            return ""
        v = str(s).strip().upper()
        # Standardize PO BOX variants
        v = re.sub(r"\bP\.?\s*O\.?\s*BOX\b", "PO BOX", v)
        # Normalize unit labels to UNIT
        v = re.sub(r"\b(APT|APARTMENT|UNIT|STE|SUITE|#|BLDG|BUILDING|RM|ROOM)\b", "UNIT", v)
        # Remove punctuation; keep letters/numbers/spaces
        v = re.sub(r"[^A-Z0-9\s]", " ", v)
        # Collapse whitespace
        v = re.sub(r"\s+", " ", v).strip()
        return v
    a1n = norm(a1)
    cityn = norm(city)
    staten = norm(state)
    zip5 = re.sub(r"[^0-9]", "", str(z or ""))[:5]
    if not (a1n and cityn and staten and zip5):
        return ""
    return f"{a1n}|{cityn}|{staten}|{zip5}"


def delete_duplicates(df_can: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Two-pass to match client: first by VIN, then by Address; keep most recent DeliveryDate per group."""
    initial = len(df_can)
    if initial == 0:
        return df_can, 0

    work = df_can.copy()
    # Helper columns
    work["___DATE"] = _effective_date_series(work)
    work["___ORDER"] = range(len(work))

    # Build VIN validity and address key
    vin_valid = pd.Series([False] * len(work))
    if "VIN" in work.columns:
        vin_valid = _safe_str(work["VIN"]).str.upper().apply(lambda v: bool(VIN_RE.fullmatch(v)))
    addr_key = pd.Series([""] * len(work))
    if all(c in work.columns for c in ["Address1", "City", "State", "Zip"]):
        addr_key = work.apply(lambda r: _normalize_address_key(r.get("Address1"), r.get("City"), r.get("State"), r.get("Zip")), axis=1)

    # Track address groups to prune if VIN-pass dropped a newer row than any remaining at that address
    prune_addr_keys: set[str] = set()

    # Pass 1: VIN-only dedupe (valid VINs). Keep most recent DeliveryDate with deterministic tiebreaks.
    if vin_valid.any():
        with_vin = work.loc[vin_valid].copy()
        without_vin = work.loc[~vin_valid].copy()

        # Helper tie-break columns
        if "Deal_Number" in with_vin.columns:
            with_vin["__HAS_DEAL"] = _safe_str(with_vin["Deal_Number"]).ne("")
            with_vin["__DN_NUM"] = pd.to_numeric(_safe_str(with_vin["Deal_Number"]), errors="coerce")
        else:
            with_vin["__HAS_DEAL"] = False
            with_vin["__DN_NUM"] = pd.Series([pd.NA] * len(with_vin))
        with_vin["__DN_NUM_FILLED"] = with_vin["__DN_NUM"].astype(float).fillna(float("-inf"))

        with_vin["___VIN_UP"] = _safe_str(with_vin["VIN"]).str.upper()
        # Sort within groups then take the last row per VIN
        with_vin_sorted = with_vin.sort_values(
            ["___VIN_UP", "___DATE", "__HAS_DEAL", "__DN_NUM_FILLED", "___ORDER"],
            ascending=[True, True, True, True, True],
        )
        with_vin_dedup = with_vin_sorted.groupby("___VIN_UP", sort=False).tail(1)
        # Identify VIN-dropped rows and compute address keys for pruning logic
        dropped_vin = with_vin.loc[~with_vin.index.isin(with_vin_dedup.index)].copy()
        if not dropped_vin.empty and all(c in with_vin.columns for c in ["Address1", "City", "State", "Zip"]):
            dropped_vin["___ADDR_KEY_FOR_DROP"] = dropped_vin.apply(
                lambda r: _normalize_address_key(r.get("Address1"), r.get("City"), r.get("State"), r.get("Zip")), axis=1
            )
        # Remaining after VIN-pass
        remaining_after_vin = pd.concat([with_vin_dedup, without_vin], ignore_index=False)
        if not dropped_vin.empty:
            # Compute address keys and max date among remaining for comparison
            if all(c in remaining_after_vin.columns for c in ["Address1", "City", "State", "Zip"]):
                rem_addr_keys = remaining_after_vin.apply(
                    lambda r: _normalize_address_key(r.get("Address1"), r.get("City"), r.get("State"), r.get("Zip")), axis=1
                )
                rem_dates = remaining_after_vin["___DATE"]
                max_date_by_addr = pd.Series(rem_dates.values, index=rem_addr_keys).groupby(level=0).max()
                for _, r in dropped_vin.iterrows():
                    k = r.get("___ADDR_KEY_FOR_DROP", "")
                    if not k:
                        continue
                    drop_date = r.get("___DATE")
                    rem_max = max_date_by_addr.get(k, pd.NaT)
                    try:
                        if pd.notna(drop_date) and (pd.isna(rem_max) or drop_date > rem_max):
                            prune_addr_keys.add(k)
                    except Exception:
                        # Fallback safe compare via string
                        if str(drop_date) > str(rem_max):
                            prune_addr_keys.add(k)
        # Update work to remaining
        work = remaining_after_vin

    # Pass 2: Address-only dedupe. Keep most recent DeliveryDate with deterministic tiebreaks.
    addr_mask = addr_key != ""
    if addr_mask.any():
        with_addr = work.loc[addr_mask].copy()
        # If VIN-pass dropped a newer row at an address, prune that whole address group here
        if prune_addr_keys:
            with_addr["___ADDR_KEY"] = with_addr.apply(lambda r: _normalize_address_key(r.get("Address1"), r.get("City"), r.get("State"), r.get("Zip")), axis=1)
            with_addr = with_addr.loc[~with_addr["___ADDR_KEY"].isin(prune_addr_keys)].copy()
            # Recompute mask for without_addr based on pruning
            without_addr = work.loc[~addr_mask].copy()
        else:
            without_addr = work.loc[~addr_mask].copy()
        without_addr = work.loc[~addr_mask].copy()

        # Helper tie-break columns
        if "Deal_Number" in with_addr.columns:
            with_addr["__HAS_DEAL"] = _safe_str(with_addr["Deal_Number"]).ne("")
            with_addr["__DN_NUM"] = pd.to_numeric(_safe_str(with_addr["Deal_Number"]), errors="coerce")
        else:
            with_addr["__HAS_DEAL"] = False
            with_addr["__DN_NUM"] = pd.Series([pd.NA] * len(with_addr))
        with_addr["__DN_NUM_FILLED"] = with_addr["__DN_NUM"].astype(float).fillna(float("-inf"))

        if "___ADDR_KEY" not in with_addr.columns:
            with_addr["___ADDR_KEY"] = with_addr.apply(lambda r: _normalize_address_key(r.get("Address1"), r.get("City"), r.get("State"), r.get("Zip")), axis=1)

        with_addr_sorted = with_addr.sort_values(
            ["___ADDR_KEY", "___DATE", "__HAS_DEAL", "__DN_NUM_FILLED", "___ORDER"],
            ascending=[True, True, True, True, True],
        )
        with_addr_dedup = with_addr_sorted.groupby("___ADDR_KEY", sort=False).tail(1)
        work = pd.concat([with_addr_dedup, without_addr], ignore_index=False)

    # Cleanup helper cols
    work = work.drop(columns=[c for c in ["___DATE", "___ORDER", "___VIN_UP", "___ADDR_KEY", "__HAS_DEAL", "__DN_NUM", "__DN_NUM_FILLED"] if c in work.columns])
    removed = initial - len(work)
    return work, removed


def filter_address_present(df_can: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    required = ["City", "State", "Zip"]
    have = [c for c in required if c in df_can.columns]
    if len(have) < 3 and "Address1" not in df_can.columns and "Address2" not in df_can.columns:
        return df_can, 0
    a1 = _safe_str(df_can["Address1"]) if "Address1" in df_can.columns else pd.Series([""] * len(df_can))
    a2 = _safe_str(df_can["Address2"]) if "Address2" in df_can.columns else pd.Series([""] * len(df_can))
    city = _safe_str(df_can["City"]) if "City" in df_can.columns else pd.Series([""] * len(df_can))
    state = _safe_str(df_can["State"]) if "State" in df_can.columns else pd.Series([""] * len(df_can))
    zipc = _safe_str(df_can["Zip"]) if "Zip" in df_can.columns else pd.Series([""] * len(df_can))
    # PO BOX counts as address
    po_mask = a2.str.contains(r"(?i)\bP\.?O\.?\s*BOX\b|\bPO\s*BOX\b")
    a1_eff = a1.where(a1 != "", a2.where(po_mask, ""))
    keep = (a1_eff != "") & (city != "") & (state != "") & (zipc != "")
    out = df_can.loc[keep].copy()
    return out, len(df_can) - len(out)


def filter_name_present(df_can: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    if "Last_Name" not in df_can.columns:
        return df_can, 0
    last = _safe_str(df_can["Last_Name"]) if "Last_Name" in df_can.columns else pd.Series([""] * len(df_can))
    keep = last != ""
    out = df_can.loc[keep].copy()
    return out, len(df_can) - len(out)


def filter_cobuyers(df_can: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Drop rows that appear to contain multiple names (co-buyer patterns) when names are combined in a single field.
    Heuristics: contains ' & ', ' and ', or '/' as person joiners in the effective name.
    Do NOT treat commas as co-buyer signals (to avoid 'LAST, FIRST' false positives).
    If explicit co-buyer columns are present (Co_First_Name/Co_Last_Name/Co_FullName), do not drop.
    """
    # Build an effective name string
    full = _safe_str(df_can["FullName"]) if "FullName" in df_can.columns else pd.Series(["" for _ in range(len(df_can))])
    first = _safe_str(df_can["First_Name"]) if "First_Name" in df_can.columns else pd.Series(["" for _ in range(len(df_can))])
    last = _safe_str(df_can["Last_Name"]) if "Last_Name" in df_can.columns else pd.Series(["" for _ in range(len(df_can))])
    eff = full
    empty_full = eff.eq("")
    eff = eff.where(~empty_full, (first + " " + last).str.replace(r"\s+", " ", regex=True).str.strip())
    # If explicit co-buyer columns exist, do not drop based on combined name pattern
    if any(c in df_can.columns for c in ["Co_First_Name", "Co_Last_Name", "Co_FullName"]):
        return df_can, 0
    # Co-buyer patterns (exclude comma to avoid 'LAST, FIRST' names)
    cobuyer_mask = eff.str.contains(r"\s+&\s+|\sand\s|\s*/\s*", regex=True, case=False, na=False)
    out = df_can.loc[~cobuyer_mask].copy()
    return out, int(cobuyer_mask.sum())


def filter_out_of_state(df_can: pd.DataFrame, home_state: str) -> Tuple[pd.DataFrame, int]:
    if "State" not in df_can.columns:
        return df_can, 0
    state = _safe_str(df_can["State"]).str.upper()
    keep = state == str(home_state).upper()
    out = df_can.loc[keep].copy()
    return out, len(df_can) - len(out)


def filter_model_year(df_can: pd.DataFrame, operator: str, year: Optional[int]) -> Tuple[pd.DataFrame, int]:
    if year is None or "Year" not in df_can.columns:
        return df_can, 0
    def to_int(v):
        try:
            return int(str(v).strip())
        except Exception:
            return None
    years = df_can["Year"].apply(to_int)
    if operator == "newer":
        keep = years > year
    elif operator == "older":
        keep = years < year
    else:
        keep = years == year
    out = df_can.loc[keep.fillna(False)].copy()
    return out, len(df_can) - len(out)


def filter_delivery_age(df_can: pd.DataFrame, months: int) -> Tuple[pd.DataFrame, int]:
    # Require an effective date when delivery-age is enabled
    eff = _effective_date_series(df_can)
    cutoff = pd.Timestamp.today() - pd.DateOffset(months=months)
    has_date = eff.notna()
    keep = has_date & (eff <= cutoff)
    out = df_can.loc[keep].copy()
    return out, len(df_can) - len(out)


def filter_distance(df_can: pd.DataFrame, max_miles: float) -> Tuple[pd.DataFrame, int]:
    if "Distance" not in df_can.columns:
        return df_can, 0
    def to_float(v):
        try:
            return float(str(v).replace(",", ""))
        except Exception:
            return None
    miles = df_can["Distance"].apply(to_float)
    # Treat implausible distances as invalid (e.g., > 1000 miles)
    miles = miles.where(~(miles.notna() & (miles > 1000)), other=None)
    valid_ratio = miles.notna().mean()
    # Gate: if too few valid numeric distances, skip this filter
    if valid_ratio < 0.05:
        return df_can, 0
    # Keep rows with distance <= threshold OR missing/invalid distance
    keep = (miles <= max_miles) | miles.isna()
    out = df_can.loc[keep.fillna(True)].copy()
    return out, len(df_can) - len(out)


def _corporate_score(text: str) -> int:
    if not text:
        return 0
    score = 0
    t = text.upper()
    tokens = re.split(r"[^A-Z0-9]+", t)
    # Word-boundary match to avoid false positives (e.g., DAVIS vs AVIS)
    if any(re.search(r"\\b" + re.escape(b) + r"\\b", t) for b in EXCLUDE_BRANDS):
        score += 3
    if any(re.search(r"\\b" + re.escape(k) + r"\\b", t) for k in EXCLUDE_KEYWORDS):
        score += 2
    if any(s in tokens for s in CORPORATE_SUFFIXES):
        score += 2
    return score


def filter_corporate(df_can: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    # Use FullName/First/Last/Store signals
    full = _safe_str(df_can["FullName"]) if "FullName" in df_can.columns else pd.Series([""] * len(df_can))
    first = _safe_str(df_can["First_Name"]) if "First_Name" in df_can.columns else pd.Series([""] * len(df_can))
    last = _safe_str(df_can["Last_Name"]) if "Last_Name" in df_can.columns else pd.Series([""] * len(df_can))
    store = _safe_str(df_can["Store"]) if "Store" in df_can.columns else pd.Series([""] * len(df_can))

    scores = []
    for i in range(len(df_can)):
        s = 0
        text_full = str(full.iloc[i])
        text_first_last = str(first.iloc[i] + " " + last.iloc[i])

        # Score ONLY name fields
        s += _corporate_score(text_full)
        s += _corporate_score(text_first_last)

        # Person-likeness (two title-cased tokens in First/Last)
        person_like = bool(first.iloc[i]) and bool(last.iloc[i]) and first.iloc[i].istitle() and last.iloc[i].istitle()
        if person_like:
            s -= 3

        # HARD RULE: if name fields contain OEM/brand or corporate suffix/keywords → force corporate
        try:
            name_fields_up = (text_full.upper(), text_first_last.upper())
            # Substring match in names (more permissive to avoid misses)
            oem_hit = any(any(tok in f for tok in EXCLUDE_OEMS) for f in name_fields_up)
            brand_hit = any(any(tok in f for tok in EXCLUDE_BRANDS) for f in name_fields_up)
            keyword_hit = any(any(tok in f for tok in EXCLUDE_KEYWORDS) for f in name_fields_up)
            tokens_split = [re.split(r"[^A-Z0-9]+", f) for f in name_fields_up]
            suffix_hit = any(any(suf in parts for suf in CORPORATE_SUFFIXES) for parts in tokens_split)
            if oem_hit or brand_hit or keyword_hit or suffix_hit:
                s = 999
        except Exception:
            pass
        scores.append(s)
    keep = pd.Series(scores) < 3
    out = df_can.loc[keep].copy()
    return out, len(df_can) - len(out)


