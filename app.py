from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox

from run_preset import run_pipeline


def select_and_run():
    path = filedialog.askopenfilename(
        title="Select file",
        filetypes=[
            ("Supported", "*.csv;*.xlsx;*.xlsm"),
            ("CSV", "*.csv"),
            ("Excel", "*.xlsx;*.xlsm"),
            ("All files", "*.*"),
        ],
    )
    if not path:
        return
    try:
        df, out_path = run_pipeline(path)
        messagebox.showinfo(
            "Done",
            f"Processed {len(df)} rows.\nSaved: {out_path}",
        )
    except Exception as e:
        messagebox.showerror("Error", str(e))


def select_and_run_multi():
    paths = filedialog.askopenfilenames(
        title="Select files",
        filetypes=[
            ("Supported", "*.csv;*.xlsx;*.xlsm"),
            ("CSV", "*.csv"),
            ("Excel", "*.xlsx;*.xlsm"),
            ("All files", "*.*"),
        ],
    )
    if not paths:
        return
    results = []
    for p in paths:
        try:
            df, out_path = run_pipeline(p, with_audits=True)
            results.append((p, len(df), out_path, None))
        except Exception as e:
            results.append((p, None, None, str(e)))
    lines = []
    for p, n, out_path, err in results:
        if err:
            lines.append(f"FAIL: {p}\n  {err}")
        else:
            lines.append(f"OK: {p}\n  {n} rows -> {out_path}")
    messagebox.showinfo("Batch complete", "\n\n".join(lines[:20]))


def main():
    root = tk.Tk()
    root.title("Dealership Sales Filter - Milestone 1")
    root.geometry("480x220")
    tk.Label(root, text="Pick file(s) (.csv/.xlsx/.xlsm) and run fixed preset filters.").pack(pady=12)
    tk.Button(root, text="Select ONE File and Run", command=select_and_run).pack(pady=6)
    tk.Button(root, text="Select MULTIPLE Files and Run (with audits)", command=select_and_run_multi).pack(pady=6)
    tk.Label(root, text="Output: timestamped XLSX per file; with audits in batch mode.").pack(pady=12)
    root.mainloop()


if __name__ == "__main__":
    main()


