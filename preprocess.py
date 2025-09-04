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


def _pre_trim_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Lightweight normalization before detection.
    - Trim whitespace
    - Collapse internal whitespace
    - Normalize common non-breaking spaces
    """
    work = df.copy()
    for col in work.columns:
        try:
            if pd.api.types.is_string_dtype(work[col]) or work[col].dtype == object:
                s = work[col].astype(str)
                s = s.str.replace("\u00A0", " ", regex=False)
                s = s.str.replace(r"\s+", " ", regex=True).str.strip()
                work[col] = s
        except Exception:
            # Best effort only
            pass
    return work


def _split_csz_value(text: str) -> Tuple[str, str, str]:
    """Parse composite City/State/Zip strings into components.
    Supports patterns like:
      - City, ST 12345
      - City ST 12345
      - City, ST
      - ST 12345
      - City 12345
    Returns (city, state, zip).
    """
    if text is None:
        return "", "", ""
    t = str(text).strip()
    if not t:
        return "", "", ""
    # Fast path: find zip at end
    m_zip = re.search(r"\b(\d{5})(?:-\d{4})?$", t)
    zip5 = m_zip.group(1) if m_zip else ""
    t_wo_zip = t[: m_zip.start()] if m_zip else t
    # If the token before zip is just a label like "ZIP"/"POSTAL", do not use it as City
    if zip5 and t_wo_zip.strip().upper() in {"ZIP", "ZIP CODE", "POSTAL", "POSTAL CODE"}:
        return "", "", zip5
    # Try City, ST
    m_city_state = re.search(r"^(?P<city>[A-Za-z][A-Za-z \-\.'/]*)[,\s]+(?P<st>[A-Za-z]{2})$", t_wo_zip.strip())
    if m_city_state:
        return (
            m_city_state.group("city").strip(),
            m_city_state.group("st").strip().upper(),
            zip5,
        )
    # Try ST <zip>
    m_state_only = re.search(r"^(?P<st>[A-Za-z]{2})$", t_wo_zip.strip())
    if m_state_only and zip5:
        return "", m_state_only.group("st").strip().upper(), zip5
    # Try City <zip>
    if zip5:
        m_city_only = re.search(r"^(?P<city>[A-Za-z][A-Za-z \-\.'/]*)$", t_wo_zip.strip())
        if m_city_only:
            city_val = m_city_only.group("city").strip()
            if city_val.upper() in {"ZIP", "ZIP CODE", "POSTAL", "POSTAL CODE", "CITY", "STATE"}:
                return "", "", zip5
            return city_val, "", zip5
    # Fallback: attempt split by comma first then space
    if "," in t_wo_zip:
        parts = [p.strip() for p in t_wo_zip.split(",") if p.strip()]
        if len(parts) >= 2 and re.fullmatch(r"[A-Za-z]{2}", parts[-1]):
            return ", ".join(parts[:-1]), parts[-1].upper(), zip5
    toks = t_wo_zip.strip().split()
    if len(toks) >= 2 and re.fullmatch(r"[A-Za-z]{2}", toks[-1]):
        return " ".join(toks[:-1]), toks[-1].upper(), zip5
    return "", "", zip5


def _pre_split_city_state_zip(df: pd.DataFrame) -> pd.DataFrame:
    """Add synthetic columns __CSZ_City/__CSZ_State/__CSZ_Zip by splitting any composite CSZ columns.
    Detection will consider these via value-pattern scoring.
    """
    work = df.copy()
    cand_cols: List[str] = []
    normed = {col: normalize_label(col) for col in work.columns}
    for col, norm in normed.items():
        if any(tok in norm for tok in ["city state zip", "city st zip", "csz", "city state", "city st", "city/ state", "city/state"]):
            cand_cols.append(col)
        else:
            # Heuristic: values frequently match City, ST 12345
            try:
                s = work[col].astype(str)
                sample = s.dropna().astype(str).head(200)
                rate = sample.str.contains(r"[A-Za-z].*,?\s*[A-Za-z]{2}\s+\d{5}(?:-\d{4})?", regex=True).mean()
                if rate >= 0.3:
                    cand_cols.append(col)
            except Exception:
                pass
    if not cand_cols:
        return work
    city_acc = pd.Series([None] * len(work))
    state_acc = pd.Series([None] * len(work))
    zip_acc = pd.Series([None] * len(work))
    for col in cand_cols:
        try:
            triples = work[col].apply(_split_csz_value)
            c = triples.apply(lambda t: t[0] if isinstance(t, tuple) else "")
            s = triples.apply(lambda t: t[1] if isinstance(t, tuple) else "")
            z = triples.apply(lambda t: t[2] if isinstance(t, tuple) else "")
            city_acc = city_acc.where(city_acc.notna(), c.where(c != "", None))
            state_acc = state_acc.where(state_acc.notna(), s.where(s != "", None))
            zip_acc = zip_acc.where(zip_acc.notna(), z.where(z != "", None))
        except Exception:
            pass
    # Materialize synthetic columns only if we extracted anything
    if city_acc.notna().any():
        work["__CSZ_City"] = city_acc.fillna("")
    if state_acc.notna().any():
        work["__CSZ_State"] = state_acc.fillna("")
    if zip_acc.notna().any():
        work["__CSZ_Zip"] = zip_acc.fillna("")
    return work


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
    # If Address1 and Address2 came from the same source column, only keep Address2 when it contains unit tokens
    try:
        if mapping.get("Address1") and mapping.get("Address2") and mapping.get("Address1") == mapping.get("Address2"):
            unit_mask = addr2.str.contains(r"(?i)\b(apt|apartment|unit|ste|suite|#|bldg|building|fl|floor|rm|room)\b")
            # Drop Address2 when it's identical to Address1 or lacks unit signal
            addr2 = addr2.where(unit_mask & addr2.str.lower().ne(addr1.str.lower()), "")
    except Exception:
        pass
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


def _normalize_phone_value(value: str) -> str:
    """Normalize a phone-like value to either 10-digit AAA-NNN-NNNN, 7-digit NNN-NNNN, or empty.
    Preserves extension if present (x1234). Drops country code by taking the last 10 digits when >=10.
    """
    if value is None:
        return ""
    text = str(value)
    m_ext = re.search(r"(?:ext\.?|x)\s*(\d{1,5})$", text, flags=re.IGNORECASE)
    ext = m_ext.group(1) if m_ext else None
    digits = re.sub(r"\D", "", text)
    core = digits[-10:] if len(digits) >= 10 else digits
    if len(core) == 10:
        out = f"{core[0:3]}-{core[3:6]}-{core[6:10]}"
    elif len(core) == 7:
        out = f"{core[0:3]}-{core[3:7]}"
    else:
        out = core
    if ext:
        out = f"{out} x{ext}"
    return out


def _merge_area_code(area_series: pd.Series | None, number_series: pd.Series) -> pd.Series:
    """Combine area code with 7-digit numbers; leave 10+ digits as-is (normalized).
    If area_series is None, just normalize the number.
    """
    if number_series is None:
        return pd.Series(["" for _ in range(0)])
    area = coerce_str(area_series) if area_series is not None else pd.Series(["" for _ in range(len(number_series))])
    nums = number_series.fillna("").astype(str)
    out_vals: List[str] = []
    for a_raw, n_raw in zip(area.tolist(), nums.tolist()):
        n_norm = _normalize_phone_value(n_raw)
        # If n_norm is 7-digit like NNN-NNNN, and area is 3 digits, build AAA-NNN-NNNN
        if re.fullmatch(r"\d{3}-\d{4}", n_norm):
            a_digits = re.sub(r"\D", "", a_raw)
            if re.fullmatch(r"\d{3}", a_digits):
                n_norm = f"{a_digits}-{n_norm}"
        out_vals.append(n_norm)
    return pd.Series(out_vals)


def build_canonical_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    # Pre-normalize and split composites before detection
    df = _pre_trim_normalize(df)
    df = _pre_split_city_state_zip(df)
    mapping, warnings = detect_schema(df)

    # Initialize canonical DataFrame with only the locked output order
    data: Dict[str, pd.Series] = {}

    # Names
    data["First_Name"] = coerce_str(df[mapping["First_Name"]]) if "First_Name" in mapping else pd.Series(["" for _ in range(len(df))])
    data["Last_Name"] = coerce_str(df[mapping["Last_Name"]]) if "Last_Name" in mapping else pd.Series(["" for _ in range(len(df))])
    data["FullName"] = derive_fullname(df, mapping)

    # Contact
    data["Email"] = coerce_str(df[mapping["Email"]]) if "Email" in mapping else pd.Series(["" for _ in range(len(df))])
    # Numbers
    home_num = df[mapping["Home_Phone"]] if "Home_Phone" in mapping else pd.Series(["" for _ in range(len(df))])
    mobile_num = df[mapping["Mobile_Phone"]] if "Mobile_Phone" in mapping else pd.Series(["" for _ in range(len(df))])
    work_num = df[mapping["Work_Phone"]] if "Work_Phone" in mapping else pd.Series(["" for _ in range(len(df))])
    phone2_num = df[mapping["Phone2"]] if "Phone2" in mapping else pd.Series(["" for _ in range(len(df))])
    # Area codes (scoped, else generic AreaCode)
    ac_generic = df[mapping["AreaCode"]] if "AreaCode" in mapping else None
    ac_home = df[mapping["Home_AreaCode"]] if "Home_AreaCode" in mapping else ac_generic
    ac_mobile = df[mapping["Mobile_AreaCode"]] if "Mobile_AreaCode" in mapping else ac_generic
    ac_work = df[mapping["Work_AreaCode"]] if "Work_AreaCode" in mapping else ac_generic
    ac_p2 = df[mapping["Phone2_AreaCode"]] if "Phone2_AreaCode" in mapping else ac_generic

    data["Home_Phone"] = _merge_area_code(ac_home, home_num)
    data["Mobile_Phone"] = _merge_area_code(ac_mobile, mobile_num)
    data["Work_Phone"] = _merge_area_code(ac_work, work_num)
    data["Phone2"] = _merge_area_code(ac_p2, phone2_num)

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


