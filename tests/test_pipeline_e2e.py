from __future__ import annotations

import os
import glob
import pandas as pd

from run_preset import run_pipeline
from verify_dedup import verify


def _fixture_path(name: str) -> str:
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(base, name)


def _no_dup_keys(df: pd.DataFrame) -> bool:
    vin = df.get("VIN")
    if vin is not None:
        vin_up = vin.fillna("").astype(str).str.strip().str.upper()
        ok_vin = ~(vin_up.str.len() == 17)
        if (vin_up[~ok_vin].duplicated(keep=False)).any():
            return False
    if all(c in df.columns for c in ["Address1", "City", "State", "Zip"]):
        addr = (df["Address1"].fillna("").astype(str).str.strip().str.upper() + "|" +
                df["City"].fillna("").astype(str).str.strip().str.upper() + "|" +
                df["State"].fillna("").astype(str).str.strip().str.upper() + "|" +
                df["Zip"].fillna("").astype(str).str.strip())
        addr5 = addr.str.replace(r"[^0-9]", "", regex=True)
        if addr[addr != ""].duplicated(keep=False).any():
            return False
    return True


def test_pipeline_dualstore_end_to_end(tmp_path):
    src = _fixture_path("dualStoreExample.csv")
    df, out_path = run_pipeline(src, with_audits=True)
    assert os.path.exists(out_path)
    # Read final xlsx to double-check invariants
    final = pd.read_excel(out_path, dtype=str)
    assert _no_dup_keys(final)
    ok, report = verify(src, out_path)
    assert ok, report


def test_pipeline_different_format_end_to_end(tmp_path):
    src = _fixture_path("differentFormatExample .csv")
    df, out_path = run_pipeline(src, with_audits=True)
    assert os.path.exists(out_path)
    final = pd.read_excel(out_path, dtype=str)
    assert _no_dup_keys(final)
    ok, report = verify(src, out_path)
    assert ok, report


