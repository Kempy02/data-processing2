# data_analysis_modules/cross_plotting.py
from __future__ import annotations
import os, re
import glob
from typing import Optional, Dict, Any, Tuple, Literal, Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .io_utils import find_frame_csvs, parse_design_and_test_from_filename, load_metric_timeseries
from .force_io import find_force_csvs, load_force_timeseries
from .force_smoothing import smooth_dataframe

from .io_utils import list_designs  # already used elsewhere
from .averaging import average_design_timeseries
# from .force_averaging import average_force_design_timeseries

Source = Literal["force", "force"]

def _parse_force_name(path: str) -> tuple[str, str]:
    """Return (design, test) from 'design_testN.csv' → ('design','testN')."""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"(.+?)_test(\d+)$", stem, flags=re.IGNORECASE)
    if m:
        return m.group(1), f"test{m.group(2)}"
    return stem, "test0"

def _pick_by_test(files: list[str], test: Optional[str], parser) -> Optional[str]:
    """Choose the file whose parsed test matches; else return the first."""
    if not files:
        return None
    if test is None:
        return files[0]
    test = str(test).lower()
    for fp in files:
        _, t = parser(fp)
        if t.lower() == test:
            return fp
    return files[0]

def _load_series(
    *,
    source: Source,
    root: str,
    design: str,
    metric: str,
    test: Optional[str],
    use_average: bool,
    averages_dir: Optional[str],
) -> pd.DataFrame:
    """
    Load a 2-col DataFrame: ['time_s', metric] from either deformation or force data.
    If use_average=True, read from <averages_dir>/<design>__avg.csv (deform)
    or <averages_dir>/<design>__force_avg.csv (force). Otherwise pick the matching test file.
    """
    if source == "deform":
        if use_average:
            if not averages_dir:
                raise ValueError("deform averages_dir is required when use_average=True for deform.")
            path = os.path.join(averages_dir, f"{design}__force_avg.csv")
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            df = pd.read_csv(path)
        else:
            fps = find_frame_csvs(root, design)
            fp  = _pick_by_test(fps, test, parse_design_and_test_from_filename)
            if not fp:
                raise FileNotFoundError(f"No per-frame CSV found for design={design}, test={test}")
            df = load_metric_timeseries(fp, columns=["time_s", metric])
        cols = ["time_s", metric]
        if not all(c in df.columns for c in cols):
            raise KeyError(f"Missing columns for deform: {cols}")
        return df[cols].dropna()

    elif source == "force":
        if use_average:
            if not averages_dir:
                raise ValueError("force averages_dir is required when use_average=True for force.")
            path = os.path.join(averages_dir, f"{design}__force_avg.csv")
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            df = pd.read_csv(path)
        else:
            fps = find_force_csvs(root, design, include_averages=False)
            fp  = _pick_by_test(fps, test, _parse_force_name)
            if not fp:
                raise FileNotFoundError(f"No force CSV found for design={design}, test={test}")
            df = load_force_timeseries(fp, columns=["time_s", metric])
        cols = ["time_s", metric]
        if not all(c in df.columns for c in cols):
            raise KeyError(f"Missing columns for force: {cols}")
        return df[cols].dropna()

    else:
        raise ValueError("source must be 'deform' or 'force'.")

def _infer_dt(ts: np.ndarray, default: float = 1/30.0) -> float:
    dif = np.diff(ts.astype(float))
    dif = dif[np.isfinite(dif) & (dif > 0)]
    return float(np.median(dif)) if dif.size else float(default)

def _resample_to_overlap(
    X: pd.DataFrame, mx: str,
    Y: pd.DataFrame, my: str,
    time_step: Optional[float] = None
) -> pd.DataFrame:
    """Interpolate X[mx] and Y[my] onto a common overlapping time grid."""
    tx = X["time_s"].to_numpy(float)
    ty = Y["time_s"].to_numpy(float)
    t0 = max(np.nanmin(tx), np.nanmin(ty))
    t1 = min(np.nanmax(tx), np.nanmax(ty))
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        raise ValueError("No overlapping time window between the two series.")
    if time_step is None:
        dt = min(_infer_dt(tx), _infer_dt(ty))
    else:
        dt = float(time_step)
    grid = np.arange(t0, t1 + 1e-9, dt)

    xv = np.interp(grid, tx, X[mx].to_numpy(float))
    yv = np.interp(grid, ty, Y[my].to_numpy(float))

    return pd.DataFrame({"time_s": grid, mx: xv, my: yv})

def plot_two_metrics_vs(
    *,
    root: str,
    design: str,
    metric_x: str,
    x_source: Source,
    metric_y: str,
    y_source: Source,
    test: Optional[str] = None,             # e.g. 'test1'; ignored if use_average=True
    use_average: bool = False,
    deform_averages_dir: Optional[str] = None,
    force_averages_dir: Optional[str] = None,
    time_step: Optional[float] = None,
    smooth: Dict[str, Any] | bool | None = None,  # {'method':'ema','alpha':0.25} etc
    style: Literal["scatter","line"] = "scatter",
    color_by_time: bool = False,
    figsize: Tuple[float,float] = (6,5),
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> pd.DataFrame:
    """
    Plot one metric against another (e.g., displacement vs force), aligning by time.
    Returns the aligned DataFrame with columns ['time_s', metric_x, metric_y].
    """
    # load series
    X = _load_series(
        source=x_source, root=root, design=design, metric=metric_x,
        test=test, use_average=use_average,
        averages_dir=(force_averages_dir)
    )
    Y = _load_series(
        source=y_source, root=root, design=design, metric=metric_y,
        test=test, use_average=use_average,
        averages_dir=(force_averages_dir)
    )

    # align / resample
    df = _resample_to_overlap(X, metric_x, Y, metric_y, time_step=time_step)

    # optional smoothing
    if smooth:
        if isinstance(smooth, bool):
            smooth = {"method": "ema", "alpha": 0.25}
        df = smooth_dataframe(df, [metric_x, metric_y], **smooth)

    # plot
    plt.figure(figsize=figsize)
    if style == "scatter":
        if color_by_time:
            sc = plt.scatter(df[metric_x], df[metric_y], c=df["time_s"], s=12)
            cbar = plt.colorbar(sc)
            cbar.set_label("time (s)")
        else:
            plt.scatter(df[metric_x], df[metric_y], s=12, alpha=0.8)
    else:
        # line in time order
        plt.plot(df[metric_x], df[metric_y], lw=1.5, alpha=0.9)

    plt.grid(True, alpha=0.3)
    plt.xlabel(xlabel or metric_x)
    plt.ylabel(ylabel or metric_y)
    plt.title(title or f"{design}: {metric_y} vs {metric_x}")
    plt.tight_layout()
    return df

def plot_displacement_vs_force(
    *,
    root: str,
    design: str,
    test: Optional[str] = None,
    use_average: bool = True,
    deform_averages_dir: Optional[str] = None,
    force_averages_dir: Optional[str] = None,
    force1_metric: str = "position",          # x-axis (from deformation)
    force2_metric: str = "load_cell.force",       # y-axis (from force)
    **kwargs
) -> pd.DataFrame:
    """Convenience wrapper for displacement (x) vs force (y)."""
    return plot_two_metrics_vs(
        root=root,
        design=design,
        metric_x=force1_metric, x_source="force",
        metric_y=force2_metric, y_source="force",
        test=test,
        use_average=use_average,
        deform_averages_dir=deform_averages_dir,
        force_averages_dir=force_averages_dir,
        **kwargs
    )

def plot_displacement_vs_force_across_designs(
    *,
    root: str,
    designs: Optional[Sequence[str]] = None,
    disp_metric: str = "position",               # deformation metric (x-axis)
    force_metric: str = "load_cell.force",            # force metric (y-axis)
    use_average: bool = True,                         # use precomputed averages
    deform_averages_dir: Optional[str] = None,        # where <design>__avg.csv lives
    force_averages_dir: Optional[str] = None,         # where <design>__force_avg.csv lives
    compute_if_missing: bool = False,                  # compute averages on-the-fly if missing
    time_step: Optional[float] = None,                # resample step for time alignment
    smooth: Dict[str, Any] | bool | None = None,      # {'method':'ema','alpha':0.25}, etc.
    style: Literal["scatter","line"] = "line",
    figsize: Tuple[float,float] = (7,5),
    legend: bool = True,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Overlay displacement-vs-force curves for multiple designs.
    Returns {design: aligned_df_with_columns['time_s', disp_metric, force_metric]}.

    If use_average=True:
      - Loads deformation averages from <deform_averages_dir>/<design>__avg.csv
      - Loads force averages      from <force_averages_dir>/<design>__force_avg.csv
      - If a file is missing and compute_if_missing=True, it will average on the fly
        from the raw per-test CSVs and write the average file for future runs.
    """
    if use_average:
        if not deform_averages_dir or not force_averages_dir:
            raise ValueError("Please provide deform_averages_dir and force_averages_dir when use_average=True.")
        os.makedirs(deform_averages_dir, exist_ok=True)
        os.makedirs(force_averages_dir, exist_ok=True)

    # discover designs if not given
    all_d = list_designs(root)
    if designs is None:
        designs = all_d
    else:
        designs = [d for d in designs if d in all_d]
        if not designs:
            raise ValueError("No matching designs found.")

    used: Dict[str, pd.DataFrame] = {}
    plt.figure(figsize=figsize)

    for design in designs:
        # --- prepare deformation average (or single) ---
        if use_average:
            deform_avg_path = os.path.join(deform_averages_dir, f"{design}__avg.csv")
            if not os.path.exists(deform_avg_path) and compute_if_missing:
                # compute deformation average for the needed metric
                f_deform = find_frame_csvs(root, design)
                if f_deform:
                    _ = average_design_timeseries(
                        f_deform, [disp_metric], out_csv=deform_avg_path, time_step=time_step
                    )
            if not os.path.exists(deform_avg_path):
                print(f"[skip] no deformation average for {design}")
                continue
            X = pd.read_csv(deform_avg_path)
            if disp_metric not in X.columns or "time_s" not in X.columns:
                print(f"[skip] {design}: disp metric '{disp_metric}' not in {deform_avg_path}")
                continue
            X = X[["time_s", disp_metric]].dropna()
        else:
            # single-test path (rare for across-design compare); use first found
            f_deform = find_frame_csvs(root, design)
            if not f_deform:
                print(f"[skip] no deformation runs for {design}")
                continue
            X = load_metric_timeseries(f_deform[0], columns=["time_s", disp_metric])

        # --- prepare force average (or single) ---
        if use_average:
            force_avg_path = os.path.join(force_averages_dir, f"{design}__force_avg.csv")
            if not os.path.exists(force_avg_path) and compute_if_missing:
                f_force = find_force_csvs(root, design, include_averages=False)
                if f_force:
                    # _ = average_force_design_timeseries(
                    #     f_force, [force_metric], out_csv=force_avg_path, time_step=time_step, normalise_start=True
                    # )
                    print("[warn] average_force_design_timeseries not implemented; skipping force average computation")
            if not os.path.exists(force_avg_path):
                print(f"[skip] no force average for {design}")
                continue
            Y = pd.read_csv(force_avg_path)
            if force_metric not in Y.columns or "time_s" not in Y.columns:
                print(f"[skip] {design}: force metric '{force_metric}' not in {force_avg_path}")
                continue
            Y = Y[["time_s", force_metric]].dropna()
        else:
            f_force = find_force_csvs(root, design, include_averages=False)
            if not f_force:
                print(f"[skip] no force runs for {design}")
                continue
            Y = load_force_timeseries(f_force[0], columns=["time_s", force_metric])

        # align to overlapping time & (optionally) resample
        try:
            df_aligned = _resample_to_overlap(X, disp_metric, Y, force_metric, time_step=time_step)
        except Exception as e:
            print(f"[skip] {design}: cannot align series ({e})")
            continue

        # smooth if requested
        if smooth:
            if isinstance(smooth, bool):
                smooth_kwargs = {"method": "ema", "alpha": 0.25}
            else:
                smooth_kwargs = dict(smooth)
            df_aligned = smooth_dataframe(df_aligned, [disp_metric, force_metric], **smooth_kwargs)

        # plot
        if style == "scatter":
            plt.scatter(df_aligned[disp_metric], df_aligned[force_metric], s=15, alpha=0.9, label=design)
        else:
            plt.plot(df_aligned[disp_metric], df_aligned[force_metric], lw=2.0, alpha=0.95, label=design)

        used[design] = df_aligned

    plt.grid(True, alpha=0.3)
    plt.xlabel(xlabel or disp_metric)
    plt.ylabel(ylabel or force_metric)
    plt.title(title or f"{force_metric} vs {disp_metric} — across designs")
    if legend:
        plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
    return used
