from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import pandas as pd


def _auto_size_columns(writer: pd.ExcelWriter, sheet_name: str):
    ws = writer.sheets[sheet_name]
    for idx, col in enumerate(ws.iter_cols(1, ws.max_column), 1):
        max_len = 0
        for cell in col:
            try:
                v = str(cell.value) if cell.value is not None else ""
            except Exception:
                v = ""
            max_len = max(max_len, len(v))
        # Add padding
        ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 80)


def write_xlsx(df: pd.DataFrame, input_path: str, output_path: Optional[str] = None) -> str:
    base_dir = os.path.dirname(os.path.abspath(input_path))
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"{base_name}_filtered_{ts}.xlsx"
    final_path = output_path or os.path.join(base_dir, out_name)

    with pd.ExcelWriter(final_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Filtered")
        _auto_size_columns(writer, "Filtered")
    return final_path


def write_multi_sheet(dfs: dict, input_path: str, output_path: str) -> str:
    """Write multiple dataframes to one workbook; keys are sheet names."""
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, df in dfs.items():
            df.to_excel(writer, index=False, sheet_name=sheet_name[:31] or "Sheet1")
            _auto_size_columns(writer, sheet_name[:31] or "Sheet1")
    return output_path


