
# data_analysis_modules/plotting.py
from __future__ import annotations
import os
from typing import Iterable, List, Optional, Sequence, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .io_utils import group_by_design, find_frame_csvs, load_metric_timeseries
from .averaging import average_design_timeseries
from .smoothing import smooth_dataframe


# -------------------------
# Helpers
# -------------------------
def _auto_time_step(times: np.ndarray) -> Optional[float]:
    """Infer a representative time step from a time array."""
    try:
        diffs = np.diff(times.astype(float))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size == 0:
            return None
        # use median to reduce effect of outliers
        return float(np.median(diffs))
    except Exception:
        return None


def _ensure_numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _maybe_resample(df: pd.DataFrame, metrics: Sequence[str], time_step: Optional[float]) -> pd.DataFrame:
    """Resample timeseries to a fixed time grid if time_step is provided."""
    if time_step is None or "time_s" not in df.columns or df.empty:
        return df
    try:
        ts = pd.to_numeric(df["time_s"], errors="coerce").astype(float)
        t0, t1 = float(np.nanmin(ts)), float(np.nanmax(ts))
        if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
            return df
        new_t = np.arange(t0, t1 + 1e-9, float(time_step))
        base = df.set_index("time_s").sort_index()
        # numeric-only interpolation
        cols = ["time_s"] + [m for m in metrics if m in df.columns]
        base = _ensure_numeric(base.reset_index(), cols).set_index("time_s")
        interp = base.reindex(base.index.union(new_t)).interpolate(method="time", limit_direction="both")
        out = interp.loc[new_t].reset_index().rename(columns={"index": "time_s"})
        return out
    except Exception:
        return df


def _apply_smoothing(df: pd.DataFrame, cols: Sequence[str], smooth: Optional[Dict[str, Any] | bool]) -> pd.DataFrame:
    """Apply smoothing if requested. See smoothing.smooth_dataframe for kwargs."""
    if not smooth:
        return df
    if isinstance(smooth, bool):
        kwargs = {"method": "ema", "alpha": 0.25}
    else:
        kwargs = dict(smooth)
    return smooth_dataframe(df, cols, **kwargs)


# -------------------------
# Public plotting functions
# -------------------------
def plot_metric_for_design(
    root: str,
    design: str,
    metric: str,
    with_average: bool = True,
    time_step: Optional[float] = None,
    figsize: Tuple[float, float] = (8, 4),
    title: Optional[str] = None,
    smooth: Optional[Dict[str, Any] | bool] = None,
) -> pd.DataFrame:
    """
    Plot one metric over time for all tests of a given design.
    Returns the averaged DataFrame (if computed) for convenience.

    Parameters
    ----------
    root : str
        Root directory containing processed data folders.
    design : str
        Design name (subfolder) to plot.
    metric : str
        Column name to plot (e.g., 'lin_mm', 'rad_mm', 'area_mm', 'bend_deg', 'lin_norm', etc.).
    with_average : bool, default True
        If True and multiple tests exist, overlay the averaged timeseries.
    time_step : float or None
        If provided, resample all series to this fixed time step spacing (in seconds).
        If None, leave native sampling.
    figsize : tuple
        Matplotlib figure size.
    title : str or None
        Plot title. If None, an automatic title is used.
    smooth : dict | bool | None
        Optional smoothing. If True, defaults to EMA(alpha=0.25).
        Or pass dict, e.g. {'method':'moving','window':9} or {'method':'savgol','window':11,'polyorder':2}.
    """
    files = find_frame_csvs(root, design)
    if not files:
        raise FileNotFoundError(f"No *_frames.csv files found for design '{design}' under {root}")

    plt.figure(figsize=figsize)

    # plot each test
    plotted_any = False
    for fp in sorted(files):
        df = load_metric_timeseries(fp, columns=["time_s", metric])
        if df.empty:
            continue

        # optional resample
        if time_step is None:
            time_step_auto = _auto_time_step(df["time_s"].to_numpy())
        else:
            time_step_auto = time_step
        df = _maybe_resample(df, [metric], time_step_auto)

        # optional smooth
        df = _apply_smoothing(df, [metric], smooth)

        plt.plot(df["time_s"], df[metric], lw=1.2, alpha=0.8, label=os.path.basename(fp))
        plotted_any = True

    # overlay average
    avg_df = None
    if with_average and len(files) >= 2:
        avg_df = average_design_timeseries(files, [metric], out_csv=None, time_step=time_step)
        avg_df = _apply_smoothing(avg_df, [metric], smooth)
        plt.plot(avg_df["time_s"], avg_df[metric], lw=2.5, label=f"{design} — avg", zorder=10)

    if not plotted_any and avg_df is None:
        raise ValueError(f"Could not plot metric '{metric}' for design '{design}' (no usable data).")

    plt.xlabel("Time (s)")
    plt.ylabel(metric)
    plt.title(title or f"{design} — {metric}")
    plt.legend(loc="best", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return avg_df if avg_df is not None else pd.DataFrame()


def plot_metrics_across_designs(
    root: str,
    designs: Optional[Sequence[str]] = None,
    metrics: Sequence[str] = ("lin_mm",),
    use_average: bool = True,
    averages_dir: Optional[str] = None,
    time_step: Optional[float] = None,
    cols: int = 2,
    figsize: Tuple[float, float] = (8, 4),
    smooth: Optional[Dict[str, Any] | bool] = None,
    legend: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Plot one or multiple metrics for all designs (or a subset). Returns a dict of the
    DataFrames used for each design (average if use_average=True, else first test).

    Parameters
    ----------
    root : str
        Root directory containing processed data.
    designs : sequence[str] or None
        Subset of designs to include; if None, include all detected.
    metrics : sequence[str]
        Column names to plot.
    use_average : bool, default True
        If True, plot the averaged timeseries per design (requires averages_dir).
        If False, plot the first found test for each design.
    averages_dir : str or None
        Directory where *_avg.csv reside (output of your averaging step).
        Required if use_average=True.
    time_step : float or None
        Optional fixed resampling step for x-axis (seconds).
    cols : int
        Number of subplot columns.
    figsize : tuple
        Size of each subplot. Overall fig size is scaled by subplot grid.
    smooth : dict | bool | None
        Optional smoothing applied to the series before plotting.
    legend : bool
        Show subplot legends.
    """
    all_designs = list(group_by_design(find_frame_csvs(root)).keys())
    # print(f"Found {len(all_designs)} designs: {all_designs}")
    if designs is None:
        designs = all_designs
    else:
        designs = [d for d in designs if d in all_designs]
        if not designs:
            raise ValueError("No matching designs found.")

    rows = int(np.ceil(len(metrics) / float(max(1, cols))))
    fig, axes = plt.subplots(rows, cols, figsize=(figsize[0]*cols, figsize[1]*rows), squeeze=False)

    used_dfs: Dict[str, pd.DataFrame] = {}

    for mi, metric in enumerate(metrics):
        ax = axes[mi // cols][mi % cols]

        for design in designs:
            if use_average:
                if not averages_dir:
                    raise ValueError("averages_dir must be provided when use_average=True")
                avg_path = os.path.join(averages_dir, f"{design}__avg.csv")
                if not os.path.exists(avg_path):
                    # compute on the fly if missing
                    files = find_frame_csvs(root, design)
                    if not files:
                        continue
                    avg_df = average_design_timeseries(files, [metric], out_csv=avg_path, time_step=time_step)
                else:
                    avg_df = pd.read_csv(avg_path)
                avg_df = _ensure_numeric(avg_df, ["time_s", metric])
                # optional smooth
                avg_df = _apply_smoothing(avg_df, [metric], smooth)
                ax.plot(avg_df["time_s"], avg_df[metric], lw=2, label=design)
                used_dfs[design] = avg_df
            else:
                files = find_frame_csvs(root, design)
                if not files:
                    continue
                df = load_metric_timeseries(files[0], columns=["time_s", metric])
                df = _ensure_numeric(df, ["time_s", metric])
                # optional resample
                step = time_step if time_step is not None else _auto_time_step(df["time_s"].to_numpy())
                df = _maybe_resample(df, [metric], step)
                # optional smooth
                df = _apply_smoothing(df, [metric], smooth)
                ax.plot(df["time_s"], df[metric], lw=1.2, label=design)
                used_dfs[design] = df

        ax.set_title(metric)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        if legend:
            ax.legend(fontsize=8, loc="best")

    # hide unused subplots
    for k in range(len(metrics), rows*cols):
        axes[k // cols][k % cols].axis("off")

    plt.tight_layout()
    return used_dfs
