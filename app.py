from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox

from run_preset import run_pipeline


def select_and_run():
    path = filedialog.askopenfilename(
        title="Select CSV",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
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


def main():
    root = tk.Tk()
    root.title("Dealership Sales Filter - Milestone 1")
    root.geometry("420x180")
    tk.Label(root, text="Pick a CSV and run fixed preset filters.").pack(pady=20)
    tk.Button(root, text="Select CSV and Run", command=select_and_run).pack(pady=10)
    tk.Label(root, text="Output: timestamped XLSX with canonical columns.").pack(pady=10)
    root.mainloop()


if __name__ == "__main__":
    main()


