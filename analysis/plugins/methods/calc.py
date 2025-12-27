"""Row-wise calculation methods for derived columns."""

import numpy as np
import pandas as pd

from . import register_calc_method


@register_calc_method("Sum - 求和")
def calc_sum(df_subset: pd.DataFrame) -> pd.Series:
    """Return row-wise sum across selected columns."""
    return df_subset.sum(axis=1, min_count=1)


@register_calc_method("Mean - 平均值")
def calc_mean(df_subset: pd.DataFrame) -> pd.Series:
    """Return row-wise mean across selected columns."""
    return df_subset.mean(axis=1)


@register_calc_method("Max - 最大值")
def calc_max(df_subset: pd.DataFrame) -> pd.Series:
    """Return row-wise max across selected columns."""
    return df_subset.max(axis=1)


@register_calc_method("Min - 最小值")
def calc_min(df_subset: pd.DataFrame) -> pd.Series:
    """Return row-wise min across selected columns."""
    return df_subset.min(axis=1)


@register_calc_method("Change - 较基线变化 (Col1 - Col2)")
def calc_cfb(df_subset: pd.DataFrame) -> pd.Series:
    """Return change from baseline (col1 - col2)."""
    if df_subset.shape[1] < 2:
        return pd.Series(np.nan, index=df_subset.index)
    return df_subset.iloc[:, 0] - df_subset.iloc[:, 1]


@register_calc_method("% Change/bl - 较基线变化率")
def calc_pct_change(df_subset: pd.DataFrame) -> pd.Series:
    """Return percent change relative to baseline."""
    if df_subset.shape[1] < 2:
        return pd.Series(np.nan, index=df_subset.index)

    baseline = df_subset.iloc[:, 0]
    current = df_subset.iloc[:, 1]

    with np.errstate(divide="ignore", invalid="ignore"):
        result = (current - baseline) / baseline * 100.0

    return result.replace([np.inf, -np.inf], np.nan)


@register_calc_method("Ratio - 比值 (Col1 / Col2)")
def calc_ratio(df_subset: pd.DataFrame) -> pd.Series:
    """Return ratio between the first two columns."""
    if df_subset.shape[1] < 2:
        return pd.Series(np.nan, index=df_subset.index)

    numerator = df_subset.iloc[:, 0]
    denominator = df_subset.iloc[:, 1]

    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator

    return result.replace([np.inf, -np.inf], np.nan)


@register_calc_method("Log Ratio - 对数比值 (ln(Col1/Col2))")
def calc_log_ratio(df_subset: pd.DataFrame) -> pd.Series:
    """Return log ratio log(col1/col2) for positive values only."""
    if df_subset.shape[1] < 2:
        return pd.Series(np.nan, index=df_subset.index)

    num = pd.to_numeric(df_subset.iloc[:, 0], errors="coerce")
    denom = pd.to_numeric(df_subset.iloc[:, 1], errors="coerce")
    valid = (num > 0) & (denom > 0)
    result = pd.Series(np.nan, index=df_subset.index, dtype="float64")

    with np.errstate(divide="ignore", invalid="ignore"):
        result[valid] = np.log(num[valid]) - np.log(denom[valid])

    return result
