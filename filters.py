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

    # Pass 1: VIN-only dedupe (valid VINs). Keep most recent DeliveryDate.
    if vin_valid.any():
        with_vin = work.loc[vin_valid].copy()
        without_vin = work.loc[~vin_valid].copy()

        def pick_idx_vin(group: pd.DataFrame) -> int:
            dn_flag = group.get("Deal_Number", pd.Series([""]*len(group), index=group.index)).fillna("").astype(str).str.strip() != ""
            order = group["___ORDER"]
            sort_df = group.copy()
            sort_df["__DN_FLAG"] = dn_flag.astype(int)
            # Sort by date asc to handle NaT as smallest, then DN flag asc to keep rows with DN last, then order asc
            sort_df = sort_df.sort_values(["___DATE", "__DN_FLAG", "___ORDER"], ascending=[True, True, True])
            # Take last (max date, has DN, latest order)
            return sort_df.index[-1]

        with_vin["___VIN_UP"] = _safe_str(with_vin["VIN"]).str.upper()
        keep_indices_vin = with_vin.groupby("___VIN_UP", sort=False).apply(pick_idx_vin).values.tolist()
        with_vin_dedup = with_vin.loc[keep_indices_vin]
        work = pd.concat([with_vin_dedup, without_vin], ignore_index=False)

    # Pass 2: Address-only dedupe. Keep most recent DeliveryDate.
    addr_mask = addr_key != ""
    if addr_mask.any():
        with_addr = work.loc[addr_mask].copy()
        without_addr = work.loc[~addr_mask].copy()

        def pick_idx_addr(group: pd.DataFrame) -> int:
            dn_flag = group.get("Deal_Number", pd.Series([""]*len(group), index=group.index)).fillna("").astype(str).str.strip() != ""
            sort_df = group.copy()
            sort_df["__DN_FLAG"] = dn_flag.astype(int)
            sort_df = sort_df.sort_values(["___DATE", "__DN_FLAG", "___ORDER"], ascending=[True, True, True])
            return sort_df.index[-1]

        with_addr["___ADDR_KEY"] = with_addr.index.map(addr_key)
        keep_indices_addr = with_addr.groupby("___ADDR_KEY", sort=False).apply(pick_idx_addr).values.tolist()
        with_addr_dedup = with_addr.loc[keep_indices_addr]
        work = pd.concat([with_addr_dedup, without_addr], ignore_index=False)

    # Cleanup helper cols
    work = work.drop(columns=[c for c in ["___DATE", "___ORDER", "___VIN_UP", "___ADDR_KEY"] if c in work.columns])
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
    if any(b in t for b in EXCLUDE_BRANDS):
        score += 3
    if any(k in t for k in EXCLUDE_KEYWORDS):
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
        s += _corporate_score(full.iloc[i])
        s += _corporate_score(first.iloc[i] + " " + last.iloc[i])
        s += _corporate_score(store.iloc[i])
        # Person override: simple Title-Case two token name reduces score
        if first.iloc[i].istitle() and last.iloc[i].istitle() and first.iloc[i] and last.iloc[i]:
            s -= 3
        scores.append(s)
    keep = pd.Series(scores) < 3
    out = df_can.loc[keep].copy()
    return out, len(df_can) - len(out)


