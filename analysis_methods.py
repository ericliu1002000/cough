import pandas as pd
import numpy as np
from typing import List, Callable, Dict, Any

# ==========================================
# 1. 注册表 (Registry)
# ==========================================
CALC_METHODS: Dict[str, Callable] = {}
AGG_METHODS: Dict[str, Any] = {}

def register_calc_method(name: str):
    """装饰器：注册行级计算方法 (用于二段配置的衍生变量)"""
    def decorator(func):
        CALC_METHODS[name] = func
        return func
    return decorator

def register_agg_method(name: str):
    """装饰器：注册聚合方法 (用于透视表 aggfunc)"""
    def decorator(func):
        AGG_METHODS[name] = func
        return func
    return decorator

# ==========================================
# 2. 核心统计算法库 (FDA/SAS Alignment)
# ==========================================

def sas_quantile(data: pd.Series, q: float) -> float:
    """
    【核心算法】实现 SAS PCTLDEF=2 (等同于 R type=2) 分位数算法。
    
    背景：
    Python pandas/numpy 默认使用线性插值 (linear interpolation)，
    而临床试验提交给 FDA 的结果通常要求与 SAS 保持一致。
    SAS 默认 PCTLDEF=5，但在描述性统计中，Type 2 (Average) 经常被用作标准。
    
    算法逻辑：
    1. 对非空数据排序: x_1 <= x_2 <= ... <= x_n
    2. 计算位置 np = n * q
    3. 如果 np 是整数 j: 结果 = (x_j + x_{j+1}) / 2 （取第 j 和 j+1 项的平均）
    4. 如果 np 不是整数: 结果 = x_{ceil(np)} （向上取整对应的值）
    """
    clean_data = pd.to_numeric(data, errors='coerce').dropna().sort_values()
    if clean_data.empty:
        return np.nan
    
    vals = clean_data.values
    n = len(vals)
    target = n * q
    
    # 判断是否极为接近整数（处理浮点数精度问题）
    if np.isclose(target, np.round(target)):
        idx = int(np.round(target))
        # 边界情况处理
        if idx == 0: return vals[0]
        if idx == n: return vals[-1]
        
        # 整数位置：取前后平均 (Python 索引从 0 开始，所以是 idx-1 和 idx)
        return (vals[idx-1] + vals[idx]) / 2.0
    else:
        # 非整数位置：向上取整
        idx = int(np.ceil(target))
        return vals[idx-1]

# ==========================================
# 3. 聚合插件 (Aggregation) - 用于透视表
#    包含：原子函数 (Atomic) 和 复合函数 (Composite)
# ==========================================

# --- 3.1 原子统计量 (Atomic Statistics) ---

@register_agg_method("n - 例数")
def agg_n(series: pd.Series) -> int:
    """有效样本量 (Non-missing count)"""
    return pd.to_numeric(series, errors='coerce').count()

@register_agg_method("Missing - 缺失值数")
def agg_missing(series: pd.Series) -> int:
    """缺失值数量 (NA count)"""
    return pd.to_numeric(series, errors='coerce').isna().sum()

@register_agg_method("Mean - 平均值")
def agg_mean_atomic(series: pd.Series) -> float:
    """算术平均值"""
    return pd.to_numeric(series, errors='coerce').mean()

@register_agg_method("SD - 标准差")
def agg_sd_atomic(series: pd.Series) -> float:
    """样本标准差 (Sample Standard Deviation, Divisor = n-1)"""
    # ddof=1 对应样本标准差，符合 FDA/SAS 标准 (Excel STDEV.S)
    return pd.to_numeric(series, errors='coerce').std(ddof=1)

@register_agg_method("SEM - 标准误")
def agg_se_atomic(series: pd.Series) -> float:
    """标准误 (Standard Error of Mean = SD / sqrt(n))"""
    s = pd.to_numeric(series, errors='coerce').dropna()
    if s.empty: return np.nan
    return s.std(ddof=1) / np.sqrt(len(s))

@register_agg_method("Variance-方差")
def agg_variance_atomic(series: pd.Series) -> float:
    """样本方差 (Sample Variance, Divisor = n-1)，与 R var(x) 对齐"""
    s = pd.to_numeric(series, errors='coerce').dropna()
    if s.empty: return np.nan
    # ddof=1 => 除以 (n-1)，对应 R 的 var(x)
    return s.var(ddof=1)

@register_agg_method("Median - 中位数")
def agg_median_atomic(series: pd.Series) -> float:
    """中位数 (SAS PCTLDEF=2)"""
    return sas_quantile(series, 0.5)

@register_agg_method("Q1 - 25%分位数")
def agg_q1_atomic(series: pd.Series) -> float:
    """第一四分位数"""
    return sas_quantile(series, 0.25)

@register_agg_method("Q3 - 75%分位数")
def agg_q3_atomic(series: pd.Series) -> float:
    """第三四分位数"""
    return sas_quantile(series, 0.75)


@register_agg_method("Min - 最小值")
def agg_min_atomic(series: pd.Series) -> float:
    return pd.to_numeric(series, errors='coerce').min()

@register_agg_method("Max - 最大值")
def agg_max_atomic(series: pd.Series) -> float:
    return pd.to_numeric(series, errors='coerce').max()



@register_agg_method("GeoMean - 几何均值")
def agg_geo_mean(series: pd.Series) -> float:
    """几何平均数 (用于对数正态分布数据，如 PK 浓度)"""
    s = pd.to_numeric(series, errors='coerce').dropna()
    s = s[s > 0] # 几何均值要求正数
    if s.empty: return np.nan
    return np.exp(np.log(s).mean())

@register_agg_method("GSD - 几何标准差")
def agg_gsd(series: pd.Series) -> float:
    """
    几何标准差 (Geometric Standard Deviation, 与 R 常用做法对齐)

    R 中常见写法：
        x <- x[x > 0]
        gsd <- exp(sd(log(x), na.rm = TRUE))
    这里等价实现：
        - 过滤非正值；
        - 在 log 尺度上计算样本标准差 sd_log (ddof=1)；
        - 返回 GSD = exp(sd_log)。
    """
    s = pd.to_numeric(series, errors='coerce').dropna()
    s = s[s > 0]
    if s.empty:
        return np.nan
    log_vals = np.log(s)
    sd_log = log_vals.std(ddof=1)
    return float(np.exp(sd_log))

@register_agg_method("CV% - 变异系数")
def agg_cv_percent(series: pd.Series) -> float:
    """变异系数百分比 (SD / Mean * 100)"""
    s = pd.to_numeric(series, errors='coerce').dropna()
    if s.empty: return np.nan
    mean_val = s.mean()
    if mean_val == 0: return np.nan
    std_val = s.std(ddof=1)
    return (std_val / mean_val) * 100.0

@register_agg_method("GCV% - 几何变异系数")
def agg_gcv_percent(series: pd.Series) -> float:
    """
    几何变异系数百分比 (Geometric Coefficient of Variation, 与 R 常用做法对齐)
    
    典型 R 实现（log 正态假设）：
        x <- x[x > 0]
        s <- sd(log(x), na.rm = TRUE)
        gcv <- sqrt(exp(s^2) - 1) * 100
    这里完全等价实现：
        - 先对正数取 log，计算样本标准差 s (ddof=1)
        - 按公式 GCV% = sqrt(exp(s^2) - 1) * 100
    """
    s = pd.to_numeric(series, errors='coerce').dropna()
    s = s[s > 0]  # GCV 仅对正值有意义
    if s.empty:
        return np.nan
    log_vals = np.log(s)
    # 样本标准差，与 R 中 sd(log(x)) 一致 (n-1 作分母)
    sd_log = log_vals.std(ddof=1)
    var_log = sd_log ** 2
    return np.sqrt(np.exp(var_log) - 1.0) * 100.0

# --- 3.2 复合统计量 (Composite Statistics for Table 1) ---

@register_agg_method("Format: n (Missing) - 例数(缺失)")
def agg_fmt_n_missing(series: pd.Series) -> str:
    """格式: 47(0)"""
    s = pd.to_numeric(series, errors='coerce')
    n = s.count()
    miss = s.isna().sum()
    return f"{n}({miss})"

@register_agg_method("Format: Mean (SD) - 均值(标准差)")
def agg_fmt_mean_sd(series: pd.Series) -> str:
    """格式: 10.859(3.139)"""
    s = pd.to_numeric(series, errors='coerce').dropna()
    if s.empty: return "NaN"
    m = s.mean()
    sd = s.std(ddof=1)
    return f"{m:.3f}({sd:.3f})"

@register_agg_method("Format: Mean (SE) - 均值(标准误)")
def agg_fmt_mean_se(series: pd.Series) -> str:
    """格式: 10.859(0.458)"""
    s = pd.to_numeric(series, errors='coerce').dropna()
    if s.empty: return "NaN"
    m = s.mean()
    se = s.std(ddof=1) / np.sqrt(len(s))
    return f"{m:.3f}({se:.3f})"

@register_agg_method("Format: Min, Max - 范围")
def agg_fmt_min_max(series: pd.Series) -> str:
    """格式: 3.536, 16.714"""
    s = pd.to_numeric(series, errors='coerce').dropna()
    if s.empty: return "NaN"
    return f"{s.min():.3f}, {s.max():.3f}"

# --- 3.2 组合统计量，返回数字  ---

@register_agg_method("组合统计：: Median (Q1, Q3) - 中位数(四分位)")
def agg_fmt_median_q1q3(series: pd.Series) -> str:
    """格式: 11.143(8.179, 12.857)"""
    try:
        med = sas_quantile(series, 0.5)
        q1 = sas_quantile(series, 0.25)
        q3 = sas_quantile(series, 0.75)
        if pd.isna(med): return "NaN"
        return f"{med:.3f}({q1:.3f}, {q3:.3f})"
    except:
        return "Error"
    
@register_agg_method("组合统计：全量指标统计")
def agg_fmt_all(series: pd.Series) -> str:
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
    except Exception as e:
        print("agg_fmt_all failed:", repr(e))
        return f"Error: {type(e).__name__}: {e}"

# ==========================================
# 4. 行级计算插件 (Row-wise Transformation)
#    用于二段配置：变量之间运算
# ==========================================

@register_calc_method("Sum - 求和")
def calc_sum(df_subset: pd.DataFrame) -> pd.Series:
    """行内求和：Q1 + Q2 + Q3"""
    # min_count=1 确保如果全为 NaN 则结果为 NaN，而不是 0
    return df_subset.sum(axis=1, min_count=1)

@register_calc_method("Mean - 平均值")
def calc_mean(df_subset: pd.DataFrame) -> pd.Series:
    """行内平均：(Q1 + Q2) / 2"""
    return df_subset.mean(axis=1)

@register_calc_method("Max - 最大值")
def calc_max(df_subset: pd.DataFrame) -> pd.Series:
    return df_subset.max(axis=1)

@register_calc_method("Min - 最小值")
def calc_min(df_subset: pd.DataFrame) -> pd.Series:
    return df_subset.min(axis=1)

@register_calc_method("Change - 较基线变化 (Col2 - Col1)")
def calc_cfb(df_subset: pd.DataFrame) -> pd.Series:
    """
    计算相对于基线的绝对变化。
    约定：用户必须按顺序选择 [基线变量, 访视变量]
    公式：Visit - Baseline
    """
    if df_subset.shape[1] < 2:
        return pd.Series(np.nan, index=df_subset.index)
    return df_subset.iloc[:, 1] - df_subset.iloc[:, 0]

@register_calc_method("% Change - 较基线变化率")
def calc_pct_change(df_subset: pd.DataFrame) -> pd.Series:
    """
    计算相对于基线的百分比变化。
    公式：((Visit - Baseline) / Baseline) * 100
    """
    if df_subset.shape[1] < 2:
        return pd.Series(np.nan, index=df_subset.index)
    
    baseline = df_subset.iloc[:, 0]
    current = df_subset.iloc[:, 1]
    
    # 处理除零错误和无效值
    with np.errstate(divide='ignore', invalid='ignore'):
        res = (current - baseline) / baseline * 100.0
        
    # 将无穷大替换为 NaN
    return res.replace([np.inf, -np.inf], np.nan)

@register_calc_method("Ratio - 比值 (Col1 / Col2)")
def calc_ratio(df_subset: pd.DataFrame) -> pd.Series:
    """
    计算两列的比值。
    公式：Numerator / Denominator
    """
    if df_subset.shape[1] < 2:
        return pd.Series(np.nan, index=df_subset.index)
    
    num = df_subset.iloc[:, 0]
    denom = df_subset.iloc[:, 1]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        res = num / denom
    return res.replace([np.inf, -np.inf], np.nan)


from scipy import stats

def calculate_anova_f_test(df, group_col, value_col):
    """
    计算单因素方差分析 (One-Way ANOVA)   F值  P值
    输入:
        df: 包含所有组数据的 DataFrame
        group_col: 分组列 (如 'ARM')
        value_col: 数值列 (如 'LCQ_Baseline')
    输出:
        F-value, P-value
    """
    # 1. 数据清洗
    clean_df = df.dropna(subset=[group_col, value_col])
    
    # 2. 按组提取数据列表 [[1,2,3], [4,5,6], ...]
    groups = []
    all_group_names = clean_df[group_col].unique()
    
    if len(all_group_names) < 2:
        return None, None # 只有一组无法比较
        
    for g_name in all_group_names:
        group_data = clean_df[clean_df[group_col] == g_name][value_col].values
        groups.append(group_data)
        
    # 3. 调用 scipy 计算
    try:
        f_stat, p_val = stats.f_oneway(*groups)
        return f_stat, p_val
    except Exception:
        return None, None

#已验证
def calculate_t_test_from_summary(mean_trt: float,
                                  mean_placebo: float,
                                  sd_trt: float,
                                  sd_placebo: float,
                                  n_trt: int,
                                  n_placebo: int,
                                  model: str = "student"):
    """
    基于汇总统计量的两样本 t 检验 (连续变量)，对应 R 中：
        - Student t-test:   t.test(..., var.equal = TRUE)
        - Welch  t-test:    t.test(..., var.equal = FALSE)

    参数:
        mean_trt, mean_placebo : 两组均值
        sd_trt, sd_placebo     : 两组标准差
        n_trt, n_placebo       : 两组样本量
        model                  : "student" (等方差) 或 "welch" (不等方差)

    返回:
        t_stat, p_value (双侧检验)
    """
    # 基础检查
    if any(v is None for v in [mean_trt, mean_placebo, sd_trt, sd_placebo, n_trt, n_placebo]):
        return np.nan, np.nan
    if n_trt <= 1 or n_placebo <= 1 or sd_trt < 0 or sd_placebo < 0:
        return np.nan, np.nan

    model_str = (model or "student").lower()
    equal_var = model_str in ("student", "students", "pooled", "equal_var")

    try:
        t_stat, p_val = stats.ttest_ind_from_stats(
            mean1=mean_trt, std1=sd_trt, nobs1=n_trt,
            mean2=mean_placebo, std2=sd_placebo, nobs2=n_placebo,
            equal_var=equal_var,
        )
        return t_stat, p_val
    except Exception:
        return np.nan, np.nan



def calculate_proportion_p_value(p1: float,
                                 p2: float,
                                 n1: int,
                                 n2: int,
                                 method: str = "chisq",
                                 correct: bool = True) -> float:
    """
    计算两比例的 P 值，使之与 R 常用做法对齐。

    - method = "chisq"  对应 R 的 prop.test(x, n, correct = TRUE/FALSE)
    - method = "fisher" 对应 R 的 fisher.test(matrix(c(x1, n1-x1, x2, n2-x2), 2))

    参数:
        p1, p2 : 两组的比例 (如 0.12, 0.25)
        n1, n2 : 两组样本量
        method : "chisq" 或 "fisher"
        correct: 是否做连续性校正 (仅对 chisq 生效)，对应 R 的 correct 参数

    返回:
        p_value (float)，若输入无效则返回 np.nan
    """
    # 基础检查
    if n1 is None or n2 is None or n1 <= 0 or n2 <= 0:
        return np.nan
    if p1 is None or p2 is None:
        return np.nan

    # 将比例还原为“事件数”，允许是浮点，但 fisher 里会取整
    x1 = p1 * n1
    x2 = p2 * n2

    # 裁剪到合理范围，避免极端数值导致数值问题
    x1 = float(np.clip(x1, 0.0, n1))
    x2 = float(np.clip(x2, 0.0, n2))

    method = (method or "chisq").lower()

    if method == "fisher":
        # Fisher 精确检验需要整数计数
        x1_i = int(round(x1))
        x2_i = int(round(x2))
        table = np.array([
            [x1_i, max(0, n1 - x1_i)],
            [x2_i, max(0, n2 - x2_i)]
        ])
        try:
            _, p_val = stats.fisher_exact(table, alternative="two-sided")
            return p_val
        except Exception:
            return np.nan

    # 默认: R prop.test 的卡方检验（带连续性校正）
    # 参考 R 源码：pchisq(stat, df = 1, lower.tail = FALSE)
    total_n = n1 + n2
    total_x = x1 + x2

    if total_n <= 0:
        return np.nan

    p_pool = total_x / total_n

    # 当 p_pool 为 0 或 1 时，分母为 0，R 中会给出 NaN/Inf，这里统一返回 NaN
    if p_pool <= 0.0 or p_pool >= 1.0:
        return np.nan

    # 连续性校正：(|x - n*p| - 0.5)+ * sign(x - n*p)
    def _one_term(x, n, p, use_correct: bool):
        diff = x - n * p
        if not use_correct:
            return diff ** 2 / (n * p * (1.0 - p))
        adj = max(0.0, abs(diff) - 0.5)
        diff_corr = np.sign(diff) * adj
        return diff_corr ** 2 / (n * p * (1.0 - p))

    try:
        chi2_stat = _one_term(x1, n1, p_pool, use_correct=correct) + \
                    _one_term(x2, n2, p_pool, use_correct=correct)
        # R: pchisq(chi2_stat, df = 1, lower.tail = FALSE)
        p_val = stats.chi2.sf(chi2_stat, df=1)
        return p_val
    except Exception:
        return np.nan
