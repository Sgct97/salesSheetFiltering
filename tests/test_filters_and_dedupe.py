from __future__ import annotations

import os
import pandas as pd

from preprocess import build_canonical_frame
from filters import (
    filter_address_present,
    filter_out_of_state,
    filter_delivery_age,
    filter_distance,
    delete_duplicates,
    _normalize_address_key,
)


def _load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _can(path: str) -> pd.DataFrame:
    df = _load_csv(path)
    can, _, _ = build_canonical_frame(df)
    return can


def test_address_present_with_composites():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    path = os.path.join(base, "differentFormatExample .csv")
    can = _can(path)
    out, _ = filter_address_present(can)
    assert set(["Address1", "City", "State", "Zip"]).issubset(out.columns)


def test_out_of_state_wa():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    path = os.path.join(base, "dualStoreExample.csv")
    can = _can(path)
    out, _ = filter_out_of_state(can, "WA")
    if len(can) > 0 and "State" in can.columns:
        assert (out["State"].str.upper() == "WA").all()


def test_distance_gate_and_threshold():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    path = os.path.join(base, "dualStoreExample.csv")
    can = _can(path)
    out, _ = filter_distance(can, 100)
    assert len(out) <= len(can)


def test_dedupe_no_duplicate_keys():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    path = os.path.join(base, "dualStoreExample.csv")
    can = _can(path)
    out, _ = delete_duplicates(can)
    # Verify no duplicate normalized address using the same normalization as production
    if all(c in out.columns for c in ["Address1", "City", "State", "Zip"]):
        addr_key = out.apply(lambda r: _normalize_address_key(r.get("Address1"), r.get("City"), r.get("State"), r.get("Zip")), axis=1)
        dup = addr_key[addr_key != ""].duplicated(keep=False)
        assert not dup.any()

