# data_analysis_modules/smoothing.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Sequence

def _ema(series: pd.Series, alpha: float = 0.25) -> pd.Series:
    return series.ewm(alpha=float(alpha), adjust=False).mean()

def _moving(series: pd.Series, window: int = 9) -> pd.Series:
    window = max(1, int(window))
    return series.rolling(window=window, min_periods=max(1, window//2)).mean()

def _savgol(series: pd.Series, window: int = 11, polyorder: int = 2) -> pd.Series:
    from scipy.signal import savgol_filter
    window = int(window) if int(window) % 2 == 1 else int(window) + 1
    window = max(3, window)
    polyorder = max(1, int(polyorder))
    if window <= polyorder:
        window = polyorder + 2 + (polyorder % 2 == 0)
    arr = series.to_numpy(dtype=float)
    return pd.Series(savgol_filter(arr, window_length=window, polyorder=polyorder), index=series.index)

def smooth_dataframe(df: pd.DataFrame, cols: Sequence[str], method: str = "ema", **kwargs) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        if method == "ema":
            alpha = float(kwargs.get("alpha", 0.25))
            out[c] = _ema(out[c], alpha=alpha)
        elif method == "moving":
            window = int(kwargs.get("window", 9))
            out[c] = _moving(out[c], window=window)
        elif method == "savgol":
            window = int(kwargs.get("window", 11))
            polyorder = int(kwargs.get("polyorder", 2))
            out[c] = _savgol(out[c], window=window, polyorder=polyorder)
    return out