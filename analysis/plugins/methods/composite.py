"""Composite statistical helpers for summary-only calculations."""

import numpy as np
from scipy import stats


def calculate_anova_f_test(df, group_col, value_col):
    """Return one-way ANOVA F and p values for grouped data."""
    clean_df = df.dropna(subset=[group_col, value_col])

    groups = []
    all_group_names = clean_df[group_col].unique()

    if len(all_group_names) < 2:
        return None, None

    for group_name in all_group_names:
        group_data = clean_df[clean_df[group_col] == group_name][value_col].values
        groups.append(group_data)

    try:
        f_stat, p_val = stats.f_oneway(*groups)
        return f_stat, p_val
    except Exception:
        return None, None


# verified

def calculate_t_test_from_summary(
    mean_trt: float,
    mean_placebo: float,
    sd_trt: float,
    sd_placebo: float,
    n_trt: int,
    n_placebo: int,
    model: str = "student",
):
    """Return two-sample t-test results from summary statistics."""
    if any(
        v is None
        for v in [mean_trt, mean_placebo, sd_trt, sd_placebo, n_trt, n_placebo]
    ):
        return np.nan, np.nan
    if n_trt <= 1 or n_placebo <= 1 or sd_trt < 0 or sd_placebo < 0:
        return np.nan, np.nan

    model_str = (model or "student").lower()
    equal_var = model_str in ("student", "students", "pooled", "equal_var")

    try:
        t_stat, p_val = stats.ttest_ind_from_stats(
            mean1=mean_trt,
            std1=sd_trt,
            nobs1=n_trt,
            mean2=mean_placebo,
            std2=sd_placebo,
            nobs2=n_placebo,
            equal_var=equal_var,
        )
        return t_stat, p_val
    except Exception:
        return np.nan, np.nan



def calculate_proportion_p_value(
    p1: float,
    p2: float,
    n1: int,
    n2: int,
    method: str = "chisq",
    correct: bool = True,
) -> float:
    """Return a two-proportion p value using chi-square or Fisher's test."""
    if n1 is None or n2 is None or n1 <= 0 or n2 <= 0:
        return np.nan
    if p1 is None or p2 is None:
        return np.nan

    x1 = p1 * n1
    x2 = p2 * n2

    x1 = float(np.clip(x1, 0.0, n1))
    x2 = float(np.clip(x2, 0.0, n2))

    method = (method or "chisq").lower()

    if method == "fisher":
        x1_i = int(round(x1))
        x2_i = int(round(x2))
        table = np.array(
            [
                [x1_i, max(0, n1 - x1_i)],
                [x2_i, max(0, n2 - x2_i)],
            ]
        )
        try:
            _, p_val = stats.fisher_exact(table, alternative="two-sided")
            return p_val
        except Exception:
            return np.nan

    total_n = n1 + n2
    total_x = x1 + x2

    if total_n <= 0:
        return np.nan

    p_pool = total_x / total_n

    if p_pool <= 0.0 or p_pool >= 1.0:
        return np.nan

    def _one_term(x, n, p, use_correct: bool):
        """Return one chi-square term with optional continuity correction."""
        diff = x - n * p
        if not use_correct:
            return diff**2 / (n * p * (1.0 - p))
        adj = max(0.0, abs(diff) - 0.5)
        diff_corr = np.sign(diff) * adj
        return diff_corr**2 / (n * p * (1.0 - p))

    try:
        chi2_stat = _one_term(x1, n1, p_pool, use_correct=correct) + _one_term(
            x2, n2, p_pool, use_correct=correct
        )
        p_val = stats.chi2.sf(chi2_stat, df=1)
        return p_val
    except Exception:
        return np.nan
