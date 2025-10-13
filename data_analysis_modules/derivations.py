# data_analysis_modules/derivations.py
from __future__ import annotations
from typing import Mapping, Callable, Union, Optional, Any
import numpy as np
import pandas as pd

# A derived spec maps "new_column_name" -> either:
#   - a string expression (uses pandas.eval), e.g. "lin_norm / rad_norm"
#   - a callable: lambda df: df["lin_norm"] / df["rad_norm"]
DerivedSpec = Mapping[str, Union[str, Callable[[pd.DataFrame], pd.Series]]]

def add_derived_columns(
    df: pd.DataFrame,
    derived: DerivedSpec,
    *,
    replace_inf_with: Optional[float] = np.nan,
    fillna: Optional[float] = None,
    engine: str = "python",          # keep it simple; 'numexpr' also works if installed
) -> pd.DataFrame:
    """
    Return a copy of `df` with extra columns defined by `derived`.

    Example:
      add_derived_columns(df, {
          "bulge_ratio": "lin_norm / rad_norm",
          "bend_efficiency": lambda d: d["horiz_disp_mm"] / (d["lin_mm"] + 1e-9),
      })

    Parameters
    ----------
    df : DataFrame
    derived : mapping
        name -> expression string OR callable(df)->Series/arraylike
    replace_inf_with : float or None
        Replace +/-inf produced by divisions, etc. Set None to skip.
    fillna : float or None
        Fill NaNs in the newly-created columns. Set None to skip.
    engine : str
        pandas.eval engine, usually "python" (safe & simple).

    Notes
    -----
    - Expressions can use any existing column name directly, plus 'np' for numpy.
    - We coerce the resulting new columns to numeric where possible.
    """
    out = df.copy()

    # Expose columns & numpy to expressions
    local_env: dict[str, Any] = {c: out[c] for c in out.columns}
    global_env: dict[str, Any] = {"np": np, "numpy": np}

    for new_col, spec in derived.items():
        if callable(spec):
            s = spec(out)
        elif isinstance(spec, str):
            # Use pandas.eval for vectorized evaluation
            s = pd.eval(spec, engine=engine, local_dict=local_env, global_dict=global_env)
        else:
            raise TypeError(f"Derived spec for '{new_col}' must be str or callable, got {type(spec)}")

        # Convert result to a numeric Series aligned to index
        s = pd.to_numeric(pd.Series(s, index=out.index), errors="coerce")

        if replace_inf_with is not None:
            s = s.replace([np.inf, -np.inf], replace_inf_with)
        if fillna is not None:
            s = s.fillna(fillna)

        out[new_col] = s

        # Keep the new column available to later expressions in this loop
        local_env[new_col] = out[new_col]

    return out
