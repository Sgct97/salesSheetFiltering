from __future__ import annotations

import re
from typing import Dict, List, Tuple

import pandas as pd

from constants import (
    CANONICAL_OUTPUT_ORDER,
    SYNONYMS,
    US_STATE_ABBR,
    DELIVERYDATE_PRECEDENCE,
)
from schema_detection import detect_schema, normalize_label


def coerce_str(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series([], dtype=object)
    return s.fillna("").astype(str).str.strip()


def normalize_state_value(value: str) -> str:
    if value is None:
        return ""
    v = str(value).strip().upper()
    return v


def assemble_address(df: pd.DataFrame, mapping: Dict[str, str]) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    addr1 = coerce_str(df[mapping["Address1"]]) if "Address1" in mapping else pd.Series(["" for _ in range(len(df))])
    addr2 = coerce_str(df[mapping["Address2"]]) if "Address2" in mapping else pd.Series(["" for _ in range(len(df))])
    city = coerce_str(df[mapping["City"]]) if "City" in mapping else pd.Series(["" for _ in range(len(df))])
    state = coerce_str(df[mapping["State"]]) if "State" in mapping else pd.Series(["" for _ in range(len(df))])
    zipc = coerce_str(df[mapping["Zip"]]) if "Zip" in mapping else pd.Series(["" for _ in range(len(df))])

    # If addr1 empty but addr2 contains PO BOX, promote
    po_mask = addr2.str.contains(r"(?i)\bP\.?O\.?\s*BOX\b|\bPO\s*BOX\b")
    addr1 = addr1.where(~(addr1.eq("") & po_mask), addr2)
    # Normalize state to uppercase
    state = state.apply(normalize_state_value)
    return addr1, addr2, city, state, zipc


def choose_delivery_date(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.Series:
    if "DeliveryDate" in mapping:
        return pd.to_datetime(df[mapping["DeliveryDate"]], errors="coerce")
    # Fallback: search columns by precedence
    norm_cols = {col: normalize_label(col) for col in df.columns}
    for tok in DELIVERYDATE_PRECEDENCE:
        for col, norm in norm_cols.items():
            if tok in norm:
                return pd.to_datetime(df[col], errors="coerce")
    return pd.to_datetime(pd.Series([None] * len(df)))


def derive_fullname(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.Series:
    first = coerce_str(df[mapping["First_Name"]]) if "First_Name" in mapping else pd.Series(["" for _ in range(len(df))])
    last = coerce_str(df[mapping["Last_Name"]]) if "Last_Name" in mapping else pd.Series(["" for _ in range(len(df))])
    # Prefer constructed FullName from First + Last when either exists
    constructed = (first + " " + last).str.replace(r"\s+", " ", regex=True).str.strip()
    if constructed.eq("").all() and "FullName" in mapping:
        # Fallback to provided FullName only if both First and Last are absent
        return coerce_str(df[mapping["FullName"]])
    return coerce_str(constructed)


def build_canonical_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    mapping, warnings = detect_schema(df)

    # Initialize canonical DataFrame with only the locked output order
    data: Dict[str, pd.Series] = {}

    # Names
    data["First_Name"] = coerce_str(df[mapping["First_Name"]]) if "First_Name" in mapping else pd.Series(["" for _ in range(len(df))])
    data["Last_Name"] = coerce_str(df[mapping["Last_Name"]]) if "Last_Name" in mapping else pd.Series(["" for _ in range(len(df))])
    data["FullName"] = derive_fullname(df, mapping)

    # Contact
    for canon in ["Email", "Home_Phone", "Mobile_Phone", "Work_Phone", "Phone2"]:
        data[canon] = coerce_str(df[mapping[canon]]) if canon in mapping else pd.Series(["" for _ in range(len(df))])

    # Address
    a1, a2, city, state, zipc = assemble_address(df, mapping)
    data["Address1"], data["Address2"], data["City"], data["State"], data["Zip"] = a1, a2, city, state, zipc

    # Vehicle basics
    for canon in ["VIN", "Make", "Model", "Year", "Vehicle_Condition", "Mileage", "Term"]:
        data[canon] = df[mapping[canon]] if canon in mapping else pd.Series([None for _ in range(len(df))])

    # Store/Deal/CustomerID
    for canon in ["Store", "Deal_Number", "CustomerID"]:
        data[canon] = coerce_str(df[mapping[canon]]) if canon in mapping else pd.Series(["" for _ in range(len(df))])

    # Distance / Delivery
    data["Distance"] = df[mapping["Distance"]] if "Distance" in mapping else pd.Series([None for _ in range(len(df))])
    data["Delivery_Miles"] = df[mapping["Delivery_Miles"]] if "Delivery_Miles" in mapping else pd.Series([None for _ in range(len(df))])
    data["DeliveryDate"] = choose_delivery_date(df, mapping)

    # Preserve original row number if present
    if "__ROWNUM" in df.columns:
        data["__ROWNUM"] = df["__ROWNUM"].astype("Int64")

    # Build final frame in locked order, dropping columns that are entirely missing
    out_cols: List[str] = []
    out_data: Dict[str, pd.Series] = {}
    for col in CANONICAL_OUTPUT_ORDER:
        if col in data:
            out_cols.append(col)
            out_data[col] = data[col]

    out_df = pd.DataFrame(out_data)
    if "__ROWNUM" in data and "__ROWNUM" not in out_df.columns:
        out_df["__ROWNUM"] = data["__ROWNUM"]
    return out_df, mapping, warnings


