# data_analysis_modules/smoothing.py
from __future__ import annotations
from typing import Sequence, Dict, Any, Optional
import pandas as pd

def _sg_smooth(s: pd.Series, window: int, polyorder: int) -> Optional[pd.Series]:
    """Internal: Savitzky–Golay with safe fallbacks."""
    try:
        from scipy.signal import savgol_filter
    except Exception:
        return None

    window = int(max(3, window))
    if window % 2 == 0:
        window += 1
    if polyorder >= window:
        polyorder = max(1, window - 1)

    try:
        arr = pd.to_numeric(s, errors="coerce").astype("float64").to_numpy()
        out = savgol_filter(arr, window_length=window, polyorder=polyorder, mode="interp")
        return pd.Series(out, index=s.index)
    except Exception:
        return None

def smooth_series(
    s: pd.Series,
    *,
    method: str = "moving",     # "moving" | "ema" | "savgol"
    window: int = 5,            # for moving/savgol
    alpha: float = 0.25,        # for ema
    polyorder: int = 2,
    center: bool = True,
) -> pd.Series:
    """
    Smooth a numeric series.
    - moving: centered moving average (pandas rolling)
    - ema:    exponential moving average (ewm)
    - savgol: Savitzky–Golay (requires SciPy); falls back to moving-average if unavailable
    """
    s = pd.to_numeric(pd.Series(s), errors="coerce")

    if method in ("moving", "ma"):
        w = max(1, int(window))
        return s.rolling(w, min_periods=1, center=center).mean()

    if method in ("ema", "exp"):
        a = float(min(max(alpha, 1e-4), 0.9999))
        return s.ewm(alpha=a, adjust=False).mean()

    if method in ("savgol", "sgolay"):
        out = _sg_smooth(s, window=window, polyorder=polyorder)
        return out if out is not None else s.rolling(max(1, int(window)), min_periods=1, center=center).mean()

    # unknown method → no-op
    return s

def smooth_dataframe(
    df: pd.DataFrame,
    cols: Sequence[str],
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Return a copy of df with selected columns smoothed via smooth_series(**kwargs).
    Nonexistent columns are ignored.
    """
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = smooth_series(out[c], **kwargs)
    return out
