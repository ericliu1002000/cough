"""Core analysis pipeline helpers used by dashboard pages."""

from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from scipy import stats

from analysis.plugins.methods import CALC_METHODS
from analysis.settings.config import get_engine
from analysis.settings.logging import log_error, log_exception
from analysis.repositories.metadata_repo import load_table_metadata
from analysis.repositories.sql_builder import build_sql


def run_analysis(config: Dict[str, Any]) -> tuple[str, pd.DataFrame]:
    """Build SQL and fetch raw data for the dashboard."""
    meta_data = load_table_metadata()

    selected_tables = config.get("selected_tables", [])
    table_columns_map = config.get("table_columns_map", {})
    filters = config.get("filters", {})
    subject_blocklist = config.get("subject_blocklist", "")

    sql = build_sql(
        selected_tables=selected_tables,
        table_columns_map=table_columns_map,
        filters=filters,
        subject_blocklist=subject_blocklist,
        meta_data=meta_data,
    )

    if not sql:
        st.error("配置错误：无法生成有效 SQL。请检查选表或筛选条件。")
        log_error(
            "analysis_service.run_analysis empty_sql",
            {"selected_tables": ",".join(selected_tables)},
        )
        return "", pd.DataFrame()

    engine = get_engine()
    with st.spinner("正在查询数据库..."):
        try:
            with engine.connect().execution_options(timeout=60) as conn:
                df = pd.read_sql(sql, conn)
        except Exception:
            log_exception("analysis_service.run_analysis query_failed")
            raise

    return sql, df


def apply_baseline_mapping(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Broadcast baseline values to each subject's rows.
    """
    if not config or not isinstance(config, dict):
        return df

    subj_col = config.get("subj_col")
    visit_col = config.get("visit_col")
    baseline_val = config.get("baseline_val")
    target_cols = config.get("target_cols", [])

    if not (subj_col and visit_col and baseline_val and target_cols):
        return df

    available_targets = [c for c in target_cols if c in df.columns]
    if not available_targets:
        return df
    if subj_col not in df.columns or visit_col not in df.columns:
        return df

    bl_mask = df[visit_col].astype(str) == str(baseline_val)
    bl_df = df.loc[bl_mask, [subj_col] + available_targets].copy()

    rename_map = {col: f"{col}_BL" for col in available_targets}
    bl_df = bl_df.rename(columns=rename_map)

    bl_df = bl_df.drop_duplicates(subset=[subj_col])

    merged_df = pd.merge(df, bl_df, on=subj_col, how="left")

    return merged_df


def apply_calculations(df: pd.DataFrame, rules: List[Dict]) -> pd.DataFrame:
    """
    Apply calculation rules with silent failure for two-pass workflows.
    """
    df_calc = df.copy()

    for rule in rules:
        try:
            name = rule["name"]
            cols = rule["cols"]
            method_name = rule["method"]

            valid_cols = [c for c in cols if c in df_calc.columns]

            if len(valid_cols) < len(cols):
                continue

            subset = df_calc[valid_cols].apply(pd.to_numeric, errors="coerce")

            if method_name in CALC_METHODS:
                calc_func = CALC_METHODS[method_name]
                df_calc[name] = calc_func(subset)

        except Exception:
            pass

    return df_calc


def calculate_anova_table(
    df: pd.DataFrame, index_col: str, group_col: str, value_col: str
) -> pd.DataFrame:
    """
    Compute one-way ANOVA grouped by pivot dimensions.
    """
    results = []

    clean_df = df.dropna(subset=[value_col, group_col])
    clean_df[value_col] = pd.to_numeric(clean_df[value_col], errors="coerce")

    row_levels = clean_df[index_col].unique()

    for level in row_levels:
        sub_df = clean_df[clean_df[index_col] == level]

        groups_data = []
        groups = sub_df[group_col].unique()

        if len(groups) < 2:
            results.append(
                {"Layer": level, "F-value": None, "P-value": None, "Note": "组数不足(<2)"}
            )
            continue

        for g in groups:
            vals = sub_df[sub_df[group_col] == g][value_col].dropna().values
            if len(vals) > 1:
                groups_data.append(vals)

        if len(groups_data) >= 2:
            try:
                f_stat, p_val = stats.f_oneway(*groups_data)
                results.append(
                    {
                        "Layer": level,
                        "F-value": float(f_stat),
                        "P-value": float(p_val),
                        "Note": "",
                    }
                )
            except Exception:
                results.append(
                    {
                        "Layer": level,
                        "F-value": None,
                        "P-value": None,
                        "Note": "计算失败",
                    }
                )
        else:
            results.append(
                {"Layer": level, "F-value": None, "P-value": None, "Note": "有效数据不足"}
            )

    return pd.DataFrame(results)
