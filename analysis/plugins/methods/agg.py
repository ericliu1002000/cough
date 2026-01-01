"""Aggregation methods and descriptive statistics for pivot tables."""

import numpy as np
import pandas as pd

from . import register_agg_method


def sas_quantile(data: pd.Series, q: float) -> float:
    """
    Compute quantiles with SAS PCTLDEF=2 (R type=2) behavior.

    This uses the SAS-aligned rule set to match FDA-style reporting.
    """
    clean_data = pd.to_numeric(data, errors="coerce").dropna().sort_values()
    if clean_data.empty:
        return np.nan

    vals = clean_data.values
    n = len(vals)
    target = n * q

    if np.isclose(target, np.round(target)):
        idx = int(np.round(target))
        if idx == 0:
            return vals[0]
        if idx == n:
            return vals[-1]
        return (vals[idx - 1] + vals[idx]) / 2.0

    idx = int(np.ceil(target))
    return vals[idx - 1]

@register_agg_method("Count - 计数")
def agg_count(series: pd.Series) -> int:
    """Return non-missing count for a series."""
    return series.count()

@register_agg_method("N")
def agg_n(series: pd.Series) -> int:
    """Return non-missing count for a series."""
    return pd.to_numeric(series, errors="coerce").count()


@register_agg_method("Missing - 缺失值数")
def agg_missing(series: pd.Series) -> int:
    """Return missing count for a series."""
    return pd.to_numeric(series, errors="coerce").isna().sum()


@register_agg_method("Mean - 平均值")
def agg_mean_atomic(series: pd.Series) -> float:
    """Return arithmetic mean for a series."""
    return pd.to_numeric(series, errors="coerce").mean()


def compute_trimmed_mean(series: pd.Series, keep_ratio: float) -> float:
    """Return trimmed mean keeping the middle proportion of data."""
    if keep_ratio is None:
        return np.nan
    try:
        keep_ratio = float(keep_ratio)
    except (TypeError, ValueError):
        return np.nan
    if keep_ratio <= 0:
        return np.nan
    keep_ratio = min(keep_ratio, 1.0)
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return np.nan
    values = np.sort(s.to_numpy(dtype=float))
    n = len(values)
    trim_ratio = max((1.0 - keep_ratio) / 2.0, 0.0)
    trim_n = int(np.floor(n * trim_ratio))
    trim_n = min(trim_n, (n - 1) // 2)
    trimmed = values[trim_n : n - trim_n]
    if trimmed.size == 0:
        return np.nan
    return float(trimmed.mean())


@register_agg_method("SD - 标准差")
def agg_sd_atomic(series: pd.Series) -> float:
    """Return sample standard deviation using ddof=1."""
    return pd.to_numeric(series, errors="coerce").std(ddof=1)


@register_agg_method("SEM - 标准误")
def agg_se_atomic(series: pd.Series) -> float:
    """Return standard error of mean for a series."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return np.nan
    return s.std(ddof=1) / np.sqrt(len(s))


@register_agg_method("Variance-方差")
def agg_variance_atomic(series: pd.Series) -> float:
    """Return sample variance using ddof=1."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return np.nan
    return s.var(ddof=1)


@register_agg_method("Median - 中位数")
def agg_median_atomic(series: pd.Series) -> float:
    """Return median using SAS-aligned quantiles."""
    return sas_quantile(series, 0.5)


@register_agg_method("Q1 - 25%分位数")
def agg_q1_atomic(series: pd.Series) -> float:
    """Return first quartile using SAS-aligned quantiles."""
    return sas_quantile(series, 0.25)


@register_agg_method("Q3 - 75%分位数")
def agg_q3_atomic(series: pd.Series) -> float:
    """Return third quartile using SAS-aligned quantiles."""
    return sas_quantile(series, 0.75)


@register_agg_method("Min - 最小值")
def agg_min_atomic(series: pd.Series) -> float:
    """Return minimum numeric value for a series."""
    return pd.to_numeric(series, errors="coerce").min()


@register_agg_method("Max - 最大值")
def agg_max_atomic(series: pd.Series) -> float:
    """Return maximum numeric value for a series."""
    return pd.to_numeric(series, errors="coerce").max()


@register_agg_method("GeoMean - 几何均值")
def agg_geo_mean(series: pd.Series) -> float:
    """Return geometric mean for positive values."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    s = s[s > 0]
    if s.empty:
        return np.nan
    return np.exp(np.log(s).mean())




@register_agg_method("GSD - 几何标准差")
def agg_gsd(series: pd.Series) -> float:
    """Return geometric standard deviation using log scale."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    s = s[s > 0]
    if s.empty:
        return np.nan
    log_vals = np.log(s)
    sd_log = log_vals.std(ddof=1)
    return float(np.exp(sd_log))


@register_agg_method("CV% - 变异系数")
def agg_cv_percent(series: pd.Series) -> float:
    """Return coefficient of variation percentage."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return np.nan
    mean_val = s.mean()
    if mean_val == 0:
        return np.nan
    std_val = s.std(ddof=1)
    return (std_val / mean_val) * 100.0


@register_agg_method("GCV% - 几何变异系数")
def agg_gcv_percent(series: pd.Series) -> float:
    """Return geometric coefficient of variation percentage."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    s = s[s > 0]
    if s.empty:
        return np.nan
    log_vals = np.log(s)
    sd_log = log_vals.std(ddof=1)
    var_log = sd_log**2
    return np.sqrt(np.exp(var_log) - 1.0) * 100.0


@register_agg_method(" n (Missing) - 例数(缺失)")
def agg_fmt_n_missing(series: pd.Series) -> str:
    """Return formatted count and missing text (e.g., 47(0))."""
    s = pd.to_numeric(series, errors="coerce")
    n = s.count()
    miss = s.isna().sum()
    return f"{n}({miss})"


@register_agg_method(" Mean (SD) - 均值(标准差)")
def agg_fmt_mean_sd(series: pd.Series) -> str:
    """Return formatted mean and SD (e.g., 10.859(3.139))."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return "NaN"
    mean_val = s.mean()
    sd = s.std(ddof=1)
    return f"{mean_val:.3f}({sd:.3f})"


@register_agg_method(" Mean (SE) - 均值(标准误)")
def agg_fmt_mean_se(series: pd.Series) -> str:
    """Return formatted mean and SE (e.g., 10.859(0.458))."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return "NaN"
    mean_val = s.mean()
    se = s.std(ddof=1) / np.sqrt(len(s))
    return f"{mean_val:.3f}({se:.3f})"





@register_agg_method(" Min, Max - 范围")
def agg_fmt_min_max(series: pd.Series) -> str:
    """Return formatted min/max range (e.g., 3.536, 16.714)."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return "NaN"
    return f"{s.min():.3f}, {s.max():.3f}"


@register_agg_method("Median (Q1, Q3) - 中位数(四分位)")
def agg_fmt_median_q1q3(series: pd.Series) -> str:
    """Return formatted median with quartiles (e.g., 11.143(8.179, 12.857))."""
    try:
        med = sas_quantile(series, 0.5)
        q1 = sas_quantile(series, 0.25)
        q3 = sas_quantile(series, 0.75)
        if pd.isna(med):
            return "NaN"
        return f"{med:.3f}({q1:.3f}, {q3:.3f})"
    except Exception:
        return "Error"


@register_agg_method("全量指标统计")
def agg_fmt_all(series: pd.Series) -> str:
    """Return a multi-line summary of all supported statistics."""
    try:
        return (
            f"N值：{agg_n(series)}\n"
            f"空值-未填写：{agg_missing(series):.3f}\n"
            f"最大-最小值：{agg_fmt_min_max(series)}\n"
            f"中位数 Median(Q1,Q3)：{agg_fmt_median_q1q3(series)}\n"
            f"MEAN平均值：{agg_mean_atomic(series):.3f}\n"
            f"SEM标准误：{agg_se_atomic(series):.3f}\n"
            f"SD标准差：{agg_sd_atomic(series):.3f}\n"
            f"GM几何平均数：{agg_geo_mean(series):.3f}\n"
            f"CV 变异系数：{agg_cv_percent(series):.3f}\n"
            f"GCV几何变异系数：{agg_gcv_percent(series):.3f}\n"
            f"方差Variance：{agg_variance_atomic(series):.3f}\n"
            f"GSD{agg_gsd(series):.3f}"
        )
    except Exception as exc:
        print("agg_fmt_all failed:", repr(exc))
        return f"Error: {type(exc).__name__}: {exc}"
