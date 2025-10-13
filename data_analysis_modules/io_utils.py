
import os
import re
from glob import glob
from pathlib import Path
from typing import Union, Dict, List, Iterable, Sequence, Optional

import numpy as np
import pandas as pd


FRAMES_SUFFIX = "__frames.csv"


def find_frame_csvs(
    root: Union[str, os.PathLike],
    design: Union[str, Sequence[str], None] = None,
    frames_suffix: str = "__frames.csv",
) -> List[str]:
    """
    Recursively find per-frame CSVs under `root`.

    If `design` is provided, only return files for that design (or any of a list of designs).
    It matches both:
        <design>_test<N>__frames.csv
    and:
        <design>__frames.csv
    (case-insensitive on the file stem, but not on the suffix).

    Args:
        root: directory to search.
        design: a single design name (str), multiple names (list/tuple), or None for all.
        frames_suffix: file suffix to match (default: "__frames.csv").

    Returns:
        Sorted list of absolute paths (strings).
    """
    root = str(root)
    pattern = os.path.join(root, "**", f"*{frames_suffix}")
    files = sorted(set(glob(pattern, recursive=True)))

    if design is None:
        return files

    # normalize to a set of lowercased target names
    if isinstance(design, str):
        targets = {design.lower()}
    else:
        targets = {str(d).lower() for d in design}

    def _stem(path: str) -> str:
        return os.path.splitext(os.path.basename(path))[0]

    filtered: List[str] = []
    for fp in files:
        stem = _stem(fp)
        stem_low = stem.lower()

        # Accept "<design>__frames" OR "<design>_test<N>__frames"
        for d in targets:
            if stem_low == f"{d}{frames_suffix[:-4]}":  # strip ".csv" → "__frames"
                filtered.append(fp)
                break
            if stem_low.startswith(f"{d}_test") and stem_low.endswith(frames_suffix[:-4]):
                filtered.append(fp)
                break

    return sorted(set(filtered))



def parse_design_and_test_from_filename(path: str) -> tuple[str, str]:
    """
    From a frames csv path like '.../bending0_test1__frames.csv', return:
        ('bending0', 'test1')
    If pattern not found, returns (stem_without_suffix, 'test0').
    """
    stem = Path(path).name
    if stem.endswith(FRAMES_SUFFIX):
        stem = stem[:-len(FRAMES_SUFFIX)]
    # Split on last '_test' occurrence to be robust to extra underscores in design names
    m = re.search(r"(.*)_test(\d+)$", stem)
    if m:
        design = m.group(1)
        test = f"test{m.group(2)}"
        return design, test
    else:
        # fallback
        return stem, "test0"


def group_by_design(files: Iterable[str]) -> Dict[str, list[str]]:
    """
    Group a list of frames CSV paths by design name.
    Returns mapping: design -> [list of paths]
    """
    by_design: dict[str, list[str]] = {}
    for f in files:
        design, _ = parse_design_and_test_from_filename(f)
        by_design.setdefault(design, []).append(f)
    # stable sort paths within each design
    for k in by_design:
        by_design[k] = sorted(by_design[k])
    return dict(sorted(by_design.items(), key=lambda kv: kv[0]))


def list_designs(root: str, frames_glob: str = "**/*__frames.csv") -> List[str]:
    """
    Scan `root` (recursively) for processed per-frame CSVs like:
        <design>_test<number>__frames.csv
    and return a sorted list of unique <design> IDs.

    Examples it handles:
      - bending0_test1__frames.csv   → "bending0"
      - LINEAR1_test12__frames.csv   → "LINEAR1"
      - weird_name_with_underscores_test3__frames.csv → "weird_name_with_underscores"

    If a file doesn't match the exact pattern, it falls back to stripping the
    '__frames.csv' suffix and trimming the last '_test...' suffix if present.

    Args:
        root: Directory to search.
        frames_glob: Glob pattern relative to root (recursive by default).

    Returns:
        Sorted list of design identifiers (strings).
    """
    pattern = os.path.join(root, frames_glob)
    files = glob(pattern, recursive=True)

    designs = set()
    rx = re.compile(r"(.+?)_test\d+__frames\.csv$", re.IGNORECASE)

    for fp in files:
        base = os.path.basename(fp)
        m = rx.match(base)
        if m:
            designs.add(m.group(1))
            continue

        # Fallbacks for slightly different names
        if base.endswith("__frames.csv"):
            stem = base[: -len("__frames.csv")]
        else:
            stem, _ = os.path.splitext(base)

        # If there is a trailing '_test...' chunk, strip it
        low = stem.lower()
        i = low.rfind("_test")
        if i != -1:
            stem = stem[:i]

        if stem:
            designs.add(stem)

    return sorted(designs)


def load_metric_timeseries(csv_path: str, columns: list[str] | None = None, fps_fallback: float = 30.0) -> pd.DataFrame:
    """
    Load a per-frame CSV and return only ['time_s'] + requested metrics.
    - Strips whitespace from column names.
    - Ensures 'time_s' exists (falls back to frame/fps).
    - Coerces numeric columns to numeric.
    - Sorts by time and drops rows with NaN time.
    """
    df = pd.read_csv(csv_path)

    # 1) normalize column names
    df.columns = [str(c).strip() for c in df.columns]

    # 2) ensure time_s exists
    if "time_s" not in df.columns:
        if "frame" in df.columns:
            frame = pd.to_numeric(df["frame"], errors="coerce")
            df["time_s"] = frame / float(fps_fallback)
        else:
            raise ValueError(f"'time_s' not in {csv_path} and cannot infer from 'frame'")

    # 3) choose columns (de-dup and always include time_s exactly once)
    wanted = ["time_s"]
    if columns is None:
        # everything
        wanted = df.columns.tolist()
    else:
        for c in columns:
            c = str(c).strip()
            if c == "time_s":
                continue
            if c in df.columns and c not in wanted:
                wanted.append(c)

    out = df.loc[:, wanted].copy()

    # 4) coerce numerics
    out["time_s"] = pd.to_numeric(out["time_s"], errors="coerce")
    for c in wanted:
        if c != "time_s":
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # 5) sort & clean
    out = out.sort_values("time_s").dropna(subset=["time_s"]).reset_index(drop=True)
    return out



def unify_time_grid(dfs: list[pd.DataFrame], time_col: str = "time_s", step: float | None = None) -> tuple[np.ndarray, list[pd.DataFrame]]:
    """
    Given multiple DataFrames with a 'time_s' column, resample all to a common time grid:
      - start at max of the starts
      - end at min of the ends
      - step: provided or derived as min median-dt across inputs (capped to [1e-3, 0.1]).
    Returns (t_grid, resampled_dfs). Resampling is linear.

    This version coerces and cleans the time column to numeric scalars, sorts rows by time if needed,
    and uses the resulting numpy arrays for computing starts/ends/medians and interpolation to avoid
    "cannot convert the series to <class 'float'>" errors.
    """
    if not dfs:
        return np.array([]), []

    # Prepare (time_array, dataframe) pairs with cleaned numeric times
    pairs: list[tuple[np.ndarray, pd.DataFrame]] = []
    for df in dfs:
        if time_col not in df.columns:
            raise ValueError(f"Time column '{time_col}' not found in DataFrame")
        s = df[time_col]

        # Try coercing to numeric floats; fall back to element