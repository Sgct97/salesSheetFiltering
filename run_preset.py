from __future__ import annotations

import os
from typing import Tuple
from datetime import datetime

import pandas as pd

from constants import PRESETS, CANONICAL_OUTPUT_ORDER
from preprocess import build_canonical_frame
from schema_detection import detect_schema
from filters import (
    find_vin_explosion_column,
    explode_vins_on_raw,
    delete_duplicates,
    filter_address_present,
    filter_name_present,
    filter_out_of_state,
    filter_model_year,
    filter_delivery_age,
    filter_distance,
    filter_corporate,
)
from write_results import write_xlsx


def run_pipeline(input_csv_path: str) -> Tuple[pd.DataFrame, str]:
    raw = pd.read_csv(input_csv_path, dtype=str, keep_default_na=False)
    # Track original CSV row numbers (header is line 1)
    raw["__ROWNUM"] = (raw.reset_index().index + 2).astype(int)

    # Optional VIN explosion on raw
    vin_col = None
    for c in raw.columns:
        if c.strip().lower() == "vin":
            vin_col = c
            break
    vin_list_col = find_vin_explosion_column(raw) if PRESETS.get("vin_explosion") else None
    if vin_list_col is not None:
        raw = explode_vins_on_raw(raw, vin_col=vin_col, vin_list_col=vin_list_col)

    # Build canonical frame
    can_df, mapping, warnings = build_canonical_frame(raw)
    # Mapping report for key fields
    report_keys = ["VIN", "Address1", "Address2", "City", "State", "Zip"]
    print("MAPPING:")
    for k in report_keys:
        src = mapping.get(k, "<none>")
        print(f"  {k}: {src}")

    steps = []
    steps.append(("initial", len(can_df)))

    # Corporate/dealer exclusion
    if PRESETS.get("exclude_corporate"):
        before = len(can_df)
        can_df, removed = filter_corporate(can_df)
        steps.append(("exclude_corporate", before, len(can_df)))

    # Name present
    if PRESETS.get("name_present"):
        before = len(can_df)
        # Debug: compute mask before filtering to show what will be dropped
        if "Last_Name" in can_df.columns:
            last_dbg = can_df["Last_Name"].fillna("").astype(str).str.strip()
            name_keep_mask = last_dbg != ""
            name_drop = can_df.loc[~name_keep_mask, [c for c in ["First_Name", "Last_Name", "FullName", "Store", "VIN"] if c in can_df.columns]].head(20)
            if not name_drop.empty:
                print("NAME DROPPED SAMPLE (first 20):")
                print(name_drop.to_string(index=False))
        can_df, removed = filter_name_present(can_df)
        steps.append(("name_present", before, len(can_df)))

    # Address present
    if PRESETS.get("address_present"):
        before = len(can_df)
        # Debug: compute mask before filtering to show what will be dropped
        a1 = can_df["Address1"].fillna("").astype(str).str.strip() if "Address1" in can_df.columns else None
        a2 = can_df["Address2"].fillna("").astype(str).str.strip() if "Address2" in can_df.columns else None
        city = can_df["City"].fillna("").astype(str).str.strip() if "City" in can_df.columns else None
        state = can_df["State"].fillna("").astype(str).str.strip() if "State" in can_df.columns else None
        zipc = can_df["Zip"].fillna("").astype(str).str.strip() if "Zip" in can_df.columns else None
        if a1 is not None and a2 is not None and city is not None and state is not None and zipc is not None:
            po_mask_dbg = a2.str.contains(r"(?i)\bP\.?O\.?\s*BOX\b|\bPO\s*BOX\b")
            a1_eff_dbg = a1.where(a1 != "", a2.where(po_mask_dbg, ""))
            addr_keep_mask = (a1_eff_dbg != "") & (city != "") & (state != "") & (zipc != "")
            addr_drop_cols = [c for c in ["__ROWNUM", "Address1", "Address2", "City", "State", "Zip", "Store", "VIN"] if c in can_df.columns]
            addr_drop = can_df.loc[~addr_keep_mask, addr_drop_cols].head(20)
            if not addr_drop.empty:
                print("ADDRESS DROPPED SAMPLE (first 20):")
                print(addr_drop.to_string(index=False))
            # Save full dropped list to CSV with original row numbers
            try:
                base_dir = os.path.dirname(os.path.abspath(input_csv_path))
                base_name = os.path.splitext(os.path.basename(input_csv_path))[0]
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                drop_path = os.path.join(base_dir, f"{base_name}_address_dropped_{ts}.csv")
                can_df.loc[~addr_keep_mask, addr_drop_cols].to_csv(drop_path, index=False)
                print(f"ADDRESS DROPPED: wrote full list to {drop_path}")
            except Exception as e:
                print(f"ADDRESS DROPPED: failed to write CSV: {e}")
        can_df, removed = filter_address_present(can_df)
        steps.append(("address_present", before, len(can_df)))

    # Out of state
    if PRESETS.get("delete_out_of_state"):
        before = len(can_df)
        can_df, removed = filter_out_of_state(can_df, PRESETS.get("home_state"))
        steps.append(("out_of_state", before, len(can_df)))

    # Model year
    my = PRESETS.get("model_year_filter", {})
    if my.get("enabled"):
        before = len(can_df)
        can_df, removed = filter_model_year(can_df, my.get("operator", "newer"), my.get("year"))
        steps.append(("model_year", before, len(can_df)))

    # Delivery age
    da = PRESETS.get("delivery_age_filter", {})
    if da.get("enabled"):
        before = len(can_df)
        can_df, removed = filter_delivery_age(can_df, da.get("months", 18))
        steps.append(("delivery_age", before, len(can_df)))

    # Distance
    df_conf = PRESETS.get("distance_filter", {})
    if df_conf.get("enabled"):
        before = len(can_df)
        can_df, removed = filter_distance(can_df, df_conf.get("max_miles", 100))
        steps.append(("distance", before, len(can_df)))

    # VIN diagnostics before dedupe
    if "VIN" in can_df.columns:
        vin_series = can_df["VIN"].fillna("").astype(str).str.strip().str.upper()
        unique_vins = vin_series.nunique(dropna=True)
        total_rows = len(vin_series)
        vin17_ratio = (vin_series.str.len() == 17).mean()
        print(f"VIN DIAG: unique={unique_vins} of {total_rows}, pct_len17={vin17_ratio:.2%}")
        try:
            top_vins = vin_series.value_counts().head(10)
            print("Top VINs by frequency (pre-dedupe):")
            print(top_vins.to_string())
        except Exception:
            pass

    # Dedupe
    if PRESETS.get("delete_duplicates"):
        before = len(can_df)
        # Track pre-dedupe indices to identify dropped rows
        can_df = can_df.copy()
        can_df["___IDX"] = range(len(can_df))
        df_before = can_df.copy()
        can_df, removed = delete_duplicates(can_df)
        kept_idx = set(can_df.get("___IDX", pd.Series([], dtype=int)).tolist())
        drop_mask = ~df_before["___IDX"].isin(kept_idx)
        drop_cols = [c for c in ["__ROWNUM", "VIN", "Deal_Number", "DeliveryDate", "Store", "FullName", "Address1", "City", "State", "Zip", "Year"] if c in df_before.columns]
        dropped_rows = df_before.loc[drop_mask, drop_cols]
        # Print sample
        if not dropped_rows.empty:
            print("DEDUPE DROPPED SAMPLE (first 20):")
            print(dropped_rows.head(20).to_string(index=False))
        # Write full list to CSV and XLSX
        try:
            base_dir = os.path.dirname(os.path.abspath(input_csv_path))
            base_name = os.path.splitext(os.path.basename(input_csv_path))[0]
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            dedupe_drop_path = os.path.join(base_dir, f"{base_name}_dedupe_dropped_{ts}.csv")
            dropped_rows.to_csv(dedupe_drop_path, index=False)
            print(f"DEDUPE DROPPED: wrote full list to {dedupe_drop_path}")
            # Also write to XLSX for review
            dedupe_drop_xlsx = os.path.join(base_dir, f"{base_name}_dedupe_dropped_{ts}.xlsx")
            try:
                write_xlsx(dropped_rows, input_csv_path, output_path=dedupe_drop_xlsx)
                print(f"DEDUPE DROPPED: wrote Excel to {dedupe_drop_xlsx}")
            except Exception as ex:
                print(f"DEDUPE DROPPED: failed to write Excel: {ex}")

            # Audit: all occurrences for VINs with duplicates (kept vs dropped)
            if "VIN" in df_before.columns:
                vin_counts = df_before["VIN"].value_counts()
                dup_vins = set(vin_counts[vin_counts > 1].index)
                all_dupes = df_before[df_before["VIN"].isin(dup_vins)].copy()
                all_dupes["Status"] = all_dupes["___IDX"].apply(lambda i: "kept" if i in kept_idx else "dropped")
                audit_cols = [c for c in ["__ROWNUM", "Status", "VIN", "Deal_Number", "DeliveryDate", "Store", "FullName", "Address1", "City", "State", "Zip", "Year"] if c in all_dupes.columns]
                all_dupes = all_dupes[audit_cols].sort_values(["VIN", "DeliveryDate"]) if "DeliveryDate" in all_dupes.columns else all_dupes.sort_values(["VIN"]) 
                audit_xlsx = os.path.join(base_dir, f"{base_name}_dedupe_all_occurrences_{ts}.xlsx")
                try:
                    write_xlsx(all_dupes, input_csv_path, output_path=audit_xlsx)
                    print(f"DEDUPE AUDIT: wrote all occurrences (kept+dropped) to {audit_xlsx}")
                except Exception as ex:
                    print(f"DEDUPE AUDIT: failed to write Excel: {ex}")
        except Exception as e:
            print(f"DEDUPE DROPPED: failed to write CSV: {e}")
        # Clean temp col if present
        if "___IDX" in can_df.columns:
            can_df = can_df.drop(columns=["___IDX"])
        steps.append(("dedupe", before, len(can_df)))

    # Enforce canonical output order; drop columns not in the list
    present = [c for c in CANONICAL_OUTPUT_ORDER if c in can_df.columns]
    out_df = can_df.loc[:, present].copy()

    out_path = write_xlsx(out_df, input_csv_path)
    # Print debug steps
    try:
        for step in steps:
            if len(step) == 3:
                name, b, a = step
                print(f"{name}: {b} -> {a}")
            else:
                print(f"{step[0]}: {step[1]}")
    except Exception:
        pass
    return out_df, out_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python run_preset.py <input_csv_path>")
        sys.exit(1)
    df, path = run_pipeline(sys.argv[1])
    print(f"Wrote {len(df)} rows to {path}")


