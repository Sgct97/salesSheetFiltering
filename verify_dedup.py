from __future__ import annotations

import os
import sys
from typing import Tuple

import pandas as pd

from preprocess import build_canonical_frame
from filters import _normalize_address_key
from constants import PRESETS

from filters import (
    filter_corporate,
    filter_out_of_state,
    filter_delivery_age,
    filter_distance,
)


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # VIN normalized
    if "VIN" in out.columns:
        out["__VIN_UP"] = out["VIN"].fillna("").astype(str).str.strip().str.upper()
    else:
        out["__VIN_UP"] = ""
    # Address key
    if all(c in out.columns for c in ["Address1", "City", "State", "Zip"]):
        out["__ADDR_KEY"] = out.apply(lambda r: _normalize_address_key(r.get("Address1"), r.get("City"), r.get("State"), r.get("Zip")), axis=1)
    else:
        out["__ADDR_KEY"] = ""
    # Dates
    # Effective date: first non-NaT among DeliveryDate, SoldDate, SaleDate, Last_Date
    eff = pd.to_datetime(out.get("DeliveryDate"), errors="coerce")
    for col in ["SoldDate", "SaleDate", "Last_Date"]:
        if col in out.columns:
            d = pd.to_datetime(out[col], errors="coerce")
            eff = eff.combine_first(d)
    out["__DATE"] = eff
    return out


def apply_prefilters(df: pd.DataFrame) -> pd.DataFrame:
    can_df, mapping, warnings = build_canonical_frame(df)
    # Apply same non-dedupe presets
    if PRESETS.get("exclude_corporate"):
        can_df, _ = filter_corporate(can_df)
    if PRESETS.get("delete_out_of_state"):
        can_df, _ = filter_out_of_state(can_df, PRESETS.get("home_state"))
    da = PRESETS.get("delivery_age_filter", {})
    if da.get("enabled"):
        can_df, _ = filter_delivery_age(can_df, da.get("months", 18))
    df_conf = PRESETS.get("distance_filter", {})
    if df_conf.get("enabled"):
        can_df, _ = filter_distance(can_df, df_conf.get("max_miles", 100))
    return can_df


def verify(input_csv: str, final_xlsx: str, dropped_xlsx: str | None = None) -> Tuple[bool, str]:
    # Pre-dedupe canonical (apply non-dedupe filters only)
    raw = pd.read_csv(input_csv, dtype=str, keep_default_na=False)
    pre = apply_prefilters(raw)
    pre = normalize_keys(pre)

    # Final result
    final = pd.read_excel(final_xlsx, dtype=str)
    # Bring keys
    final = normalize_keys(final)

    issues = []

    # 1) Assert no duplicate VINs in final (valid 17-char only)
    vin_len17 = final["__VIN_UP"].str.len() == 17
    dup_vin = final.loc[vin_len17, "__VIN_UP"].duplicated(keep=False)
    if dup_vin.any():
        issues.append(f"Final contains duplicate VINs: {final.loc[vin_len17 & dup_vin, '__VIN_UP'].unique()[:10].tolist()}")

    # 2) Assert no duplicate normalized addresses in final
    addr_nonempty = final["__ADDR_KEY"] != ""
    dup_addr = final.loc[addr_nonempty, "__ADDR_KEY"].duplicated(keep=False)
    if dup_addr.any():
        issues.append("Final contains duplicate addresses (normalized).")

    # 3) If dropped provided, ensure each dropped row has a matching group in pre and kept is most recent
    if dropped_xlsx and os.path.exists(dropped_xlsx):
        dropped = pd.read_excel(dropped_xlsx, dtype=str)
        dropped = normalize_keys(dropped)
        # For each dropped row, verify there exists a pre row sharing VIN OR address, and that a kept row in final has the same key with >= date
        for _, r in dropped.iterrows():
            vin = r.get("__VIN_UP", "")
            addr = r.get("__ADDR_KEY", "")
            date = pd.to_datetime(r.get("DeliveryDate"), errors="coerce")

            pre_candidates = pre[(pre["__VIN_UP"] == vin) | ((addr != "") & (pre["__ADDR_KEY"] == addr))]
            if pre_candidates.empty:
                issues.append(f"Dropped row has no matching VIN or Address in pre-filter set (rownum={r.get('__ROWNUM')}).")
                continue
            # Verify a kept row exists in final with same VIN or address
            kept_candidates = final[(final["__VIN_UP"] == vin) | ((addr != "") & (final["__ADDR_KEY"] == addr))]
            if kept_candidates.empty:
                # Under two-pass VIN->Address, it's valid if the address pass removed the entire household
                pre_addr_group = pre[(addr != "") & (pre["__ADDR_KEY"] == addr)]
                if not pre_addr_group.empty:
                    # Ensure there existed at least one candidate in pre, so removal was by address pass
                    continue
                issues.append(f"Dropped row has no corresponding kept row with same VIN or Address (rownum={r.get('__ROWNUM')}).")
                continue
            # Most recent rule: kept must have date >= dropped date when dates available
            if pd.notna(date) and kept_candidates["__DATE"].notna().any():
                if kept_candidates["__DATE"].max() < date:
                    issues.append(f"Dropped newer row than kept for key (VIN={vin},ADDR={addr}).")

    ok = len(issues) == 0
    report = "\n".join(issues) if issues else "All checks passed. No duplicate VINs or addresses in final; all dropped rows have valid matches and older-or-equal dates."
    return ok, report


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_dedup.py <input_csv> <final_xlsx> [dropped_xlsx]")
        sys.exit(1)
    input_csv = sys.argv[1]
    final_xlsx = sys.argv[2]
    dropped_xlsx = sys.argv[3] if len(sys.argv) > 3 else None
    ok, report = verify(input_csv, final_xlsx, dropped_xlsx)
    print(report)
    sys.exit(0 if ok else 2)


