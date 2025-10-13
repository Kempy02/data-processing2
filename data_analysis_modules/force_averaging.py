# data_analysis_modules/force_averaging.py
from __future__ import annotations
import os, re
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from pathlib import Path

from .force_io import find_force_csvs, list_force_designs, load_force_timeseries, unify_time_grid
from .force_io import FORCE_AVG_SUFFIX  

def average_design_force(
    files: List[str],
    metrics: List[str],
    out_csv: str | None = None,
    time_step: float | None = None,
    derived: Dict[str, str] | None = None,
) -> pd.DataFrame:
    if not files:
        raise ValueError("No files provided")
    dfs = [load_force_timeseries(fp, columns=["time_s"] + metrics) for fp in files]
    grid, resampled = unify_time_grid(dfs, time_col="time_s", step=time_step)
    if grid.size == 0:
        raise ValueError("No overlapping time window across files")
    avg = pd.DataFrame({"time_s": grid})
    for m in metrics:
        avg[m] = np.nanmean([d[m].to_numpy(dtype=float) for d in resampled], axis=0)
    if derived:
        local_ns = {col: avg[col] for col in avg.columns if col != "time_s"}
        for new_col, expr in derived.items():
            try:
                avg[new_col] = pd.eval(expr, engine="python", parser="pandas", local_dict=local_ns)
                local_ns[new_col] = avg[new_col]
            except Exception:
                avg[new_col] = np.nan
    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        avg.to_csv(out_csv, index=False)
    return avg

def average_all_force_designs(
    root: str | os.PathLike,
    metrics: List[str],
    out_dir: str | os.PathLike,
    time_step: float | None = None,
    derived: Dict[str, str] | None = None,
    overwrite: bool = False,
) -> Dict[str, str]:
    """
    Create '<design>__force_avg.csv' for each discovered design.
    Skips designs that already have an average file unless overwrite=True.
    Never treats existing averages as inputs.
    """
    os.makedirs(out_dir, exist_ok=True)
    designs = list_force_designs(root)
    out_paths: Dict[str, str] = {}

    for design in designs:
        out_csv = str(Path(out_dir) / f"{design}{FORCE_AVG_SUFFIX}")
        if (not overwrite) and os.path.exists(out_csv):
            # skip silently
            out_paths[design] = out_csv
            continue

        files = find_force_csvs(root, design=design, include_averages=False)
        if not files:
            continue

        # compute average and write
        from .force_averaging import average_design_force  # local import to avoid cycle
        avg_df = average_design_force(files, metrics, out_csv=out_csv, time_step=time_step, derived=derived)
        out_paths[design] = out_csv

    return out_paths