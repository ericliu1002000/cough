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

@register_agg_method("CV% - 变异系数")
def agg_cv_percent(series: pd.Series) -> float:
    """变异系数百分比 (SD / Mean * 100)"""
    s = pd.to_numeric(series, errors='coerce').dropna()
    if s.empty: return np.nan
    mean_val = s.mean()
    if mean_val == 0: return np.nan
    std_val = s.std(ddof=1)
    return (std_val / mean_val) * 100.0

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
            f"平均值：{agg_mean_atomic(series):.3f}\n"
            f"标准误SEM：{agg_se_atomic(series):.3f}\n"
            f"标准差SD：{agg_sd_atomic(series):.3f}\n"
            f"几何平均数：{agg_geo_mean(series):.3f}\n"
            f"CV 变异系数：{agg_cv_percent(series):.3f}\n"
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