# data_analysis_modules/force_plotting.py
from __future__ import annotations
import os
from typing import Sequence, Optional, Tuple, Dict, Any, List
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .force_io import find_force_csvs, list_force_designs, load_force_timeseries, FORCE_AVG_SUFFIX
from .force_averaging import average_design_force
from .force_smoothing import smooth_dataframe

def _auto_dt(times: np.ndarray) -> Optional[float]:
    diffs = np.diff(times.astype(float))
    diffs = diffs[(diffs > 0) & np.isfinite(diffs)]
    return float(np.median(diffs)) if diffs.size else None

def _apply_smoothing(df: pd.DataFrame, cols: Sequence[str], smooth: Optional[Dict[str, Any] | bool] = None) -> pd.DataFrame:
    if not smooth:
        return df
    if isinstance(smooth, bool):
        kwargs = {"method": "ema", "alpha": 0.25}
    else:
        kwargs = dict(smooth)
    return smooth_dataframe(df, cols, **kwargs)

def plot_force_metric_for_design(
    root: str,
    design: str,
    metric: str = "load_cell.force",
    with_average: bool = True,
    time_step: Optional[float] = None,
    figsize: Tuple[float, float] = (8,4),
    title: Optional[str] = None,
    smooth: Optional[Dict[str, Any] | bool] = None,
) -> pd.DataFrame:
    files = find_force_csvs(root, design=design)
    if not files:
        raise FileNotFoundError(f"No force CSVs found for design '{design}' under {root}")
    plt.figure(figsize=figsize)
    plotted = False
    for fp in sorted(files):
        df = load_force_timeseries(fp, columns=["time_s", metric], normalize_time_to_zero=True)
        if df.empty:
            continue
        step = time_step if time_step is not None else _auto_dt(df["time_s"].to_numpy())
        if step is not None:
            t = df["time_s"].to_numpy()
            new_t = np.arange(t.min(), t.max() + 1e-12, step)
            y = np.interp(new_t, t, df[metric].to_numpy(dtype=float))
            df = pd.DataFrame({"time_s": new_t, metric: y})
        df = _apply_smoothing(df, [metric], smooth)
        plt.plot(df["time_s"], df[metric], lw=1.2, alpha=0.85, label=os.path.basename(fp))
        plotted = True
    avg_df = None
    if with_average and len(files) >= 2:
        avg_df = average_design_force(files, [metric], out_csv=None, time_step=time_step, derived=None)
        avg_df = _apply_smoothing(avg_df, [metric], smooth)
        plt.plot(avg_df["time_s"], avg_df[metric], lw=2.4, label=f"{design} — avg", zorder=10)
    if not plotted and avg_df is None:
        raise ValueError(f"Could not plot '{metric}' for '{design}' (no usable data).")
    plt.xlabel("Time (s) — normalized (t0 = 0s)")
    plt.ylabel(metric)
    plt.title(title or f"{design} — {metric}")
    plt.legend(fontsize=8, loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return avg_df if avg_df is not None else pd.DataFrame()


def plot_force_metrics_across_designs(
    root: str,
    designs: Optional[Sequence[str]] = None,
    metrics: Sequence[str] = ("load_cell.force",),
    use_average: bool = True,
    averages_dir: Optional[str] = None,
    time_step: Optional[float] = None,
    cols: int = 2,
    figsize: Tuple[float, float] = (8,4),
    smooth: Optional[Dict[str, Any] | bool] = None,
    legend: bool = True,
) -> Dict[str, pd.DataFrame]:
    all_designs = list_force_designs(root)
    if designs is None:
        designs = all_designs
    else:
        designs = [d for d in designs if d in all_designs]
        if not designs:
            raise ValueError("No matching designs found")
    rows = int(np.ceil(len(metrics) / float(max(1, cols))))
    fig, axes = plt.subplots(rows, cols, figsize=(figsize[0]*cols, figsize[1]*rows), squeeze=False)
    used: Dict[str, pd.DataFrame] = {}
    for mi, metric in enumerate(metrics):
        ax = axes[mi // cols][mi % cols]
        for design in designs:
            if use_average:
                if not averages_dir:
                    raise ValueError("averages_dir must be provided when use_average=True")
                avg_path = os.path.join(averages_dir, f"{design}{FORCE_AVG_SUFFIX}")
                avg_df = None
                if os.path.exists(avg_path):
                    tmp = pd.read_csv(avg_path)
                    # If this metric isn't in the saved average, compute on the fly (no write)
                    if (metric in tmp.columns) and ("time_s" in tmp.columns):
                        avg_df = tmp[["time_s", metric]].dropna()
                    else:
                        files = find_force_csvs(root, design=design, include_averages=False)
                        if files:
                            avg_df = average_design_force(files, [metric], out_csv=None, time_step=time_step)
                else:
                    # No saved average—compute on the fly
                    files = find_force_csvs(root, design=design, include_averages=False)
                    if files:
                        avg_df = average_design_force(files, [metric], out_csv=None, time_step=time_step)

                if avg_df is None or avg_df.empty:
                    continue

                avg_df = _apply_smoothing(avg_df, [metric], smooth)
                ax.plot(avg_df["time_s"], avg_df[metric], lw=2.0, label=design)
                used[design] = avg_df
            else:
                files = find_force_csvs(root, design=design)
                if not files:
                    continue
                df = load_force_timeseries(files[0], columns=["time_s", metric], normalize_time_to_one=True)
                step = time_step if time_step is not None else _auto_dt(df["time_s"].to_numpy())
                if step is not None:
                    t = df["time_s"].to_numpy()
                    new_t = np.arange(t.min(), t.max() + 1e-12, step)
                    y = np.interp(new_t, t, df[metric].to_numpy(dtype=float))
                    df = pd.DataFrame({"time_s": new_t, metric: y})
                df = _apply_smoothing(df, [metric], smooth)
                ax.plot(df["time_s"], df[metric], lw=1.2, label=design)
                used[design] = df
        ax.set_title(metric)
        ax.set_xlabel("Time (s) — normalized (t0 = 0s)")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        if legend:
            ax.legend(fontsize=8, loc="best")
    for k in range(len(metrics), rows*cols):
        axes[k // cols][k % cols].axis("off")
    plt.tight_layout()
    return used


def _list_force_avg_designs(averages_dir: str) -> list[str]:
    """
    Return design names for files like '<design>__force_avg.csv' in averages_dir.
    """
    pat = os.path.join(averages_dir, "*__force_avg.csv")
    files = sorted(set(glob(pat)))
    designs = []
    for fp in files:
        base = os.path.basename(fp)
        if base.endswith("__force_avg.csv"):
            designs.append(base[:-len("__force_avg.csv")])
    return designs


def plot_force_vs_position_across_designs(
    *,
    averages_dir: str,
    designs: Optional[Sequence[str]] = None,
    position_col: str = "position",
    force_col: str = "load_cell.force",
    style: str = "line",                         # "line" or "scatter"
    figsize: Tuple[float, float] = (8, 5),
    title: Optional[str] = "Force vs Position — averages",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend: bool = True,
    save_path: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Plot position (x) vs force (y) for each design using ONLY '<design>__force_avg.csv'
    files found in `averages_dir`.

    Returns {design: df_used} for convenience.
    """
    if designs is None:
        designs = _list_force_avg_designs(averages_dir)
    else:
        # keep only those with existing avg files
        designs = [d for d in designs
                   if os.path.exists(os.path.join(averages_dir, f"{d}__force_avg.csv"))]

    if not designs:
        raise FileNotFoundError(f"No '*__force_avg.csv' files found in {averages_dir}")

    used: Dict[str, pd.DataFrame] = {}
    plt.figure(figsize=figsize)

    for design in designs:
        path = os.path.join(averages_dir, f"{design}__force_avg.csv")
        if not os.path.exists(path):
            print(f"[skip] missing avg: {path}")
            continue

        df = pd.read_csv(path)
        # ensure required columns exist & numeric
        for c in (position_col, force_col):
            if c not in df.columns:
                print(f"[skip] {design}: missing column '{c}' in {path}")
                break
        else:
            df = df[[position_col, force_col]].copy()
            df[position_col] = pd.to_numeric(df[position_col], errors="coerce")
            df[force_col]    = pd.to_numeric(df[force_col], errors="coerce")
            df = df.dropna().sort_values(position_col)

            if style == "scatter":
                plt.scatter(df[position_col], df[force_col], s=16, alpha=0.9, label=design)
            else:
                plt.plot(df[position_col], df[force_col], lw=2.0, alpha=0.95, label=design)

            used[design] = df

    if not used:
        raise RuntimeError("Nothing plotted (all series missing or empty).")

    plt.grid(True, alpha=0.3)
    plt.xlabel(xlabel or position_col)
    plt.ylabel(ylabel or force_col)
    if title:
        plt.title(title)
    if legend:
        plt.legend(fontsize=8, loc="best")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)

    return used
