
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Dict, List

import numpy as np
import pandas as pd

from .io_utils import (
    find_frame_csvs, group_by_design, load_metric_timeseries, unify_time_grid
)
from .derivations import add_derived_columns, DerivedSpec


def _infer_dt_from_series(ts: pd.Series, default: float = 1/30.0) -> float:
    """Robustly infer a timestep from a time Series without casting the whole Series to float."""
    arr = np.diff(ts.values.astype(float))
    arr = arr[np.isfinite(arr) & (arr > 0)]
    return float(np.median(arr)) if arr.size else float(default)

def average_design_timeseries(files: list[str],
                              metrics: list[str],
                              out_csv: str | None = None,
                              time_step: float | None = None,
                              derived: DerivedSpec | None = None) -> pd.DataFrame:
    """
    Given several per-frame CSVs for the *same design*, align on a common time grid and average.
    """
    if not files:
        raise ValueError("No files provided for averaging")

    # Load only what's needed; loader ensures 'time_s' is present & numeric
    dfs = [load_metric_timeseries(fp, columns=metrics) for fp in files]

    # Infer dt if not provided
    if time_step is None:
        dts = [_infer_dt_from_series(df["time_s"]) for df in dfs]
        time_step = float(np.median(dts)) if len(dts) else 1/30.0

    # Common overlapping window
    t_min = max(df["time_s"].min() for df in dfs)
    t_max = min(df["time_s"].max() for df in dfs)
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
        raise ValueError("Design runs have no overlapping time window")

    grid = np.arange(t_min, t_max + 1e-12, time_step)

    # Interpolate each file on the grid
    resampled = []
    for df in dfs:
        interp = {"time_s": grid}
        t = df["time_s"].values.astype(float)
        for m in metrics:
            y = df[m].values.astype(float)
            interp[m] = np.interp(grid, t, y)
        resampled.append(pd.DataFrame(interp))

    # Average across files
    avg = resampled[0][["time_s"]].copy()
    for m in metrics:
        avg[m] = np.mean([d[m].values for d in resampled], axis=0)

    # Add derived columns if requested
    if derived:
        avg = add_derived_columns(avg, derived)

    if out_csv:
        avg.to_csv(out_csv, index=False)

    return avg



def average_all_designs(
    root: str | os.PathLike,
    metrics: list[str],
    out_dir: str | os.PathLike,
    time_step: float | None = None,
    derived: DerivedSpec | None = None,
) -> Dict[str, str]:
    """
    Discover designs under `root`, average each design for the given metrics, and
    write CSVs under `out_dir` as '<design>__avg.csv'. Returns {design: csv_path}.
    """
    files = find_frame_csvs(root)
    groups = group_by_design(files)

    out_paths: Dict[str, str] = {}
    os.makedirs(out_dir, exist_ok=True)

    for design, flist in groups.items():
        out_csv = Path(out_dir) / f"{design}__avg.csv"
        average_design_timeseries(flist, metrics, out_csv=out_csv, time_step=time_step, derived=derived)
        out_paths[design] = str(out_csv)

    return out_paths
