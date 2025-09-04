from __future__ import annotations

import os
import pandas as pd

from preprocess import build_canonical_frame


def _load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def test_schema_maps_critical_fields_dualstore():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    path = os.path.join(base, "dualStoreExample.csv")
    df = _load_csv(path)
    _, mapping, _ = build_canonical_frame(df)
    for key in ["Address1", "City", "State", "Zip"]:
        assert key in mapping, f"Expected mapping for {key} in dualStoreExample.csv"


def test_schema_maps_critical_fields_different_format():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    path = os.path.join(base, "differentFormatExample .csv")
    df = _load_csv(path)
    _, mapping, _ = build_canonical_frame(df)
    for key in ["Address1", "City", "State", "Zip"]:
        assert key in mapping, f"Expected mapping for {key} in differentFormatExample .csv"


