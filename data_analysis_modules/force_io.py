# data_analysis_modules/force_io.py (replace find_force_csvs + list_force_designs)

from glob import glob
import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Sequence, List

FORCE_AVG_SUFFIX = "__force_avg.csv"

def find_force_csvs(
    root: Union[str, os.PathLike],
    design: Union[str, Sequence[str], None] = None,
    *,
    include_averages: bool = False,
) -> List[str]:
    """
    Recursively find raw force CSVs under `root`.
    Excludes <design>__force_avg.csv unless include_averages=True.
    If `design` provided, returns only those that match the design (case-insensitive),
    accepting both '<design>.csv' and '<design>_testN.csv' forms.
    """
    root = str(root)
    pattern = os.path.join(root, "**", "*.csv")
    files = sorted(set(glob(pattern, recursive=True)))

    # Exclude averages by default
    if not include_averages:
        files = [f for f in files if not os.path.basename(f).endswith(FORCE_AVG_SUFFIX)]

    if design is None:
        return files

    if isinstance(design, str):
        targets = {design.lower()}
    else:
        targets = {str(d).lower() for d in design}

    out = []
    for fp in files:
        stem = os.path.splitext(os.path.basename(fp))[0].lower()
        # match <design> or <design>_testN
        for d in targets:
            if stem == d or (stem.startswith(d + "_test")):
                out.append(fp)
                break
    return sorted(set(out))


def list_force_designs(root: str) -> List[str]:
    """
    Discover unique design names in force CSVs (ignoring *_force_avg.csv).
    Accepts 'design.csv' and 'design_testN.csv'.
    """
    files = find_force_csvs(root, design=None, include_averages=False)
    designs = set()
    rx = re.compile(r"(.+?)_test\d+$", re.IGNORECASE)
    for fp in files:
        stem = os.path.splitext(os.path.basename(fp))[0]
        m = rx.match(stem)
        designs.add(m.group(1) if m else stem)
    return sorted(designs)


def load_force_timeseries(
    csv_path: str,
    columns: Sequence[str] | None = None,
    normalize_time_to_zero: bool = True,
    normalize_position_to_zero: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    if "time_s" not in df.columns:
        raise ValueError(f"'time_s' column not found in {csv_path}")
    if columns is None:
        wanted = [c for c in DEFAULT_FORCE_COLUMNS if c in df.columns]
    else:
        wanted = ["time_s"]
        for c in columns:
            c = str(c).strip()
            if c == "time_s":
                continue
            if c in df.columns and c not in wanted:
                wanted.append(c)
    out = df.loc[:, wanted].copy()
    out["time_s"] = pd.to_numeric(out["time_s"], errors="coerce")
    for c in out.columns:
        if c == "time_s":
            continue
        if c.startswith("gpio."):
            if out[c].dtype == object:
                out[c] = out[c].astype(str).str.strip().str.lower().map({"true": True, "false": False})
            out[c] = out[c].astype("boolean")
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["time_s"]).sort_values("time_s").reset_index(drop=True)
    if normalize_time_to_zero and not out.empty:
        t0 = float(out.loc[0, "time_s"])
        out["time_s"] = out["time_s"] - t0 + 0.0
    if normalize_position_to_zero and "position" in out.columns and not out.empty:
        p0 = float(out.loc[0, "position"])
        out["position"] = out["position"] - p0 + 0.0
    return out

def unify_time_grid(
    dfs: list[pd.DataFrame],
    time_col: str = "time_s",
    step: float | None = None
) -> tuple[np.ndarray, list[pd.DataFrame]]:
    if not dfs:
        return np.array([]), []
    times = []
    for df in dfs:
        if time_col not in df.columns:
            raise ValueError(f"Missing '{time_col}' in DataFrame")
        t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(t)
        t = t[ok]
        if t.size == 0:
            raise ValueError("Empty/invalid time column found")
        times.append(t)
    t_min = max(t[0] for t in times)
    t_max = min(t[-1] for t in times)
    if t_max <= t_min:
        return np.array([]), []
    if step is None:
        dts = []
        for t in times:
            dt = np.diff(t)
            dt = dt[(dt > 0) & np.isfinite(dt)]
            if dt.size:
                dts.append(np.median(dt))
        step = float(np.median(dts)) if dts else 1/30.0
    step = float(step)
    grid = np.arange(t_min, t_max + 1e-12, step)
    resampled = []
    for df in dfs:
        out = pd.DataFrame({"time_s": grid})
        for col in df.columns:
            if col == "time_s":
                continue
            if pd.api.types.is_bool_dtype(df[col]):
                vals = df[col].astype(float).to_numpy()
                t = pd.to_numeric(df["time_s"], errors="coerce").to_numpy(dtype=float)
                interp = np.interp(grid, t, vals)
                out[col] = interp >= 0.5
            else:
                vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
                t = pd.to_numeric(df["time_s"], errors="coerce").to_numpy(dtype=float)
                out[col] = np.interp(grid, t, vals)
        resampled.append(out)
    return grid, resampled