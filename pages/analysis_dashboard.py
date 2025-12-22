import html
import json
from urllib.parse import urlencode
from typing import Any, Dict, List

import pandas as pd
import copy
import streamlit as st
from streamlit import config as st_config
from scipy import stats  # 用于计算 ANOVA

from settings import get_engine
from utils import (
    fetch_all_setups,
    fetch_setup_config,
    load_table_metadata,
    build_sql,
    save_calculation_config
)

# 引入插件系统
from analysis_methods import CALC_METHODS, AGG_METHODS
# 引入独立的图表组件
from charts.classic import draw_spaghetti_chart, build_spaghetti_fig, render_spaghetti_fig
from charts.uniform import (
    build_uniform_spaghetti_fig,
    compute_uniform_axes,
    render_uniform_spaghetti_fig,
)
from exports.charts import build_charts_export_html
from exports.common import df_to_csv_bytes
from exports.pivot import nested_pivot_to_excel_bytes
from views.pivot_classic import render_pivot_classic
from views.pivot_nested import render_pivot_nested

st.set_page_config(page_title="分析仪表盘", layout="wide")
st.title("📊 分析仪表盘")
st.markdown(
    """
    <style>
    section.main > div.block-container {
        max-width: 100%;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# 核心逻辑层 (Core Logic)
# ==========================================

def run_analysis(config: Dict[str, Any]) -> tuple[str, pd.DataFrame]:
    """ETL层：生成SQL并获取原始数据"""
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
        return "", pd.DataFrame()

    engine = get_engine()
    with st.spinner("正在查询数据库..."):
        # 设置超时防止卡死
        with engine.connect().execution_options(timeout=60) as conn:
            df = pd.read_sql(sql, conn)
            
    return sql, df


def apply_baseline_mapping(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    [BDS 引擎] 基线变量映射
    将纵向数据 (Long Format) 中的基线行数值，横向广播到该受试者的每一行。
    """
    if not config or not isinstance(config, dict):
        return df
    
    subj_col = config.get("subj_col")
    visit_col = config.get("visit_col")
    baseline_val = config.get("baseline_val")
    target_cols = config.get("target_cols", [])

    if not (subj_col and visit_col and baseline_val and target_cols):
        return df
    
    # 容错：确保所需列存在
    available_targets = [c for c in target_cols if c in df.columns]
    if not available_targets:
        return df
    if subj_col not in df.columns or visit_col not in df.columns:
        return df

    # 1. 提取基线子集
    # 筛选出 Visit == Baseline 的行
    bl_mask = df[visit_col].astype(str) == str(baseline_val)
    bl_df = df.loc[bl_mask, [subj_col] + available_targets].copy()
    
    # 2. 重命名生成 _BL 后缀
    rename_map = {col: f"{col}_BL" for col in available_targets}
    bl_df = bl_df.rename(columns=rename_map)
    
    # 3. 去重 (确保每个受试者只有一行基线)
    bl_df = bl_df.drop_duplicates(subset=[subj_col])

    # 4. 合并回主表 (Left Join)
    merged_df = pd.merge(df, bl_df, on=subj_col, how="left")
    
    return merged_df


def apply_calculations(df: pd.DataFrame, rules: List[Dict]) -> pd.DataFrame:
    """
    [计算引擎] 执行计算规则
    支持静默失败，以便支持两段式计算 (Two-Pass Calculation)。
    """
    df_calc = df.copy()
    
    for rule in rules:
        try:
            name = rule['name']
            cols = rule['cols']
            method_name = rule['method']
            
            # 验证列是否存在
            valid_cols = [c for c in cols if c in df_calc.columns]
            
            # 如果所需列不全（比如缺了基线列），在 Pass 1 阶段跳过
            if len(valid_cols) < len(cols):
                continue

            # 强制转数值
            subset = df_calc[valid_cols].apply(pd.to_numeric, errors='coerce')

            # 调用插件
            if method_name in CALC_METHODS:
                calc_func = CALC_METHODS[method_name]
                df_calc[name] = calc_func(subset)
                
        except Exception:
            # 静默失败，允许 Pass 2 重试
            pass
            
    return df_calc


def calculate_anova_table(df: pd.DataFrame, index_col: str, group_col: str, value_col: str) -> pd.DataFrame:
    """
    [统计引擎] 计算组间差异 (One-Way ANOVA)
    自动基于透视表的维度进行计算。
    """
    results = []
    
    # 确保数值有效
    clean_df = df.dropna(subset=[value_col, group_col])
    clean_df[value_col] = pd.to_numeric(clean_df[value_col], errors='coerce')
    
    # 1. 遍历行维度 (如: Day 14, Day 28)
    row_levels = clean_df[index_col].unique()
    
    for level in row_levels:
        # 取出这一层的数据
        sub_df = clean_df[clean_df[index_col] == level]
        
        # 2. 按组提取数据
        groups_data = []
        groups = sub_df[group_col].unique()
        
        if len(groups) < 2:
            results.append({
                "Layer": level, "F-value": None, "P-value": None, "Note": "组数不足(<2)"
            })
            continue
            
        # 提取每一组的数值列表
        for g in groups:
            vals = sub_df[sub_df[group_col] == g][value_col].dropna().values
            if len(vals) > 1: 
                groups_data.append(vals)
        
        # 3. 计算 F/P
        if len(groups_data) >= 2:
            try:
                f_stat, p_val = stats.f_oneway(*groups_data)
                results.append({
                    "Layer": level,
                    "F-value": f_stat,
                    "P-value": p_val,
                    "Note": "Significant" if p_val < 0.05 else ""
                })
            except Exception:
                results.append({"Layer": level, "F-value": None, "P-value": None, "Note": "Calc Error"})
        else:
            results.append({"Layer": level, "F-value": None, "P-value": None, "Note": "数据不足"})
            
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        # 格式化显示
        res_df["F-value"] = res_df["F-value"].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "-")
        res_df["P-value"] = res_df["P-value"].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "-")
        # 排序
        try:
            res_df = res_df.sort_values("Layer")
        except:
            pass
        
    return res_df


# ==========================================
# UI 表现层 (Main)
# ==========================================

def main() -> None:
    # --- 1. 侧边栏 ---
    with st.sidebar:
        st.header("🧩 分析集配置")
        setups = fetch_all_setups()

        if not setups:
            st.info("暂无配置。请先去主页创建。")
            return

        option_labels = [f"{row['setup_name']}" for row in setups]
        selected_label = st.selectbox("选择配置", options=option_labels)
        selected_row = next(r for r in setups if f"{r['setup_name']}" == selected_label)
        
        if selected_row.get("description"):
            st.info(f"📝 **备注**: {selected_row['description']}")

    # --- 1.1 状态管理与初始化 ---
    if "current_setup_name" not in st.session_state:
        st.session_state["current_setup_name"] = selected_row["setup_name"]
        need_reload = True
    else:
        need_reload = st.session_state["current_setup_name"] != selected_row["setup_name"]

    if need_reload:
        st.session_state["current_setup_name"] = selected_row["setup_name"]
        
        cfg_pack = fetch_setup_config(selected_row["setup_name"]) or {}
        calc_cfg = cfg_pack.get("calculation") or {}
        if isinstance(calc_cfg, list): calc_cfg = {"calc_rules": calc_cfg}
        
        st.session_state["calc_rules"] = calc_cfg.get("calc_rules", [])
        st.session_state["calc_note"] = calc_cfg.get("note", "")
        st.session_state["exclusions"] = calc_cfg.get("exclusions", [])
        st.session_state["pivot_config"] = calc_cfg.get("pivot", {})
        st.session_state["baseline_config"] = calc_cfg.get("baseline", {}) 

        p_cfg = st.session_state["pivot_config"]
        # 兼容历史数据：早期版本可能使用简单字符串 'mean' 作为聚合函数名，
        # 现在统一为聚合函数名称列表。
        raw_agg = p_cfg.get("agg", ["Mean - 平均值"])
        if isinstance(raw_agg, str):
            if raw_agg == "mean":
                raw_agg = "Mean - 平均值"
            raw_aggs = [raw_agg]
        else:
            raw_aggs = list(raw_agg)

        st.session_state["pivot_index"] = p_cfg.get("index", [])
        st.session_state["pivot_columns"] = p_cfg.get("columns", [])
        st.session_state["pivot_values"] = p_cfg.get("values", [])
        st.session_state["pivot_aggs"] = raw_aggs
        st.session_state["pivot_view_mode"] = p_cfg.get("view", "classic")
        row_order_cfg = p_cfg.get("row_order", {})
        if not isinstance(row_order_cfg, dict):
            row_order_cfg = {}
        st.session_state["pivot_row_order_field"] = row_order_cfg.get("field")
        st.session_state["pivot_row_order_values"] = list(
            row_order_cfg.get("values", [])
        )

        st.session_state.pop("raw_df", None)
        st.session_state.pop("current_sql", None)
        st.session_state.pop("selected_subject_id", None)

    # --- 2. 加载源数据 ---
    if st.button("🚀 加载源数据", type="primary"):
        full_cfg = fetch_setup_config(selected_row["setup_name"])
        if full_cfg and full_cfg.get("extraction"):
            sql, df_res = run_analysis(full_cfg["extraction"])
            if not df_res.empty:
                st.session_state["raw_df"] = df_res
                st.session_state["current_sql"] = sql
                st.success(f"加载成功！共 {len(df_res)} 行。")
            else:
                st.warning("查询结果为空。")

    # --- 3. 数据处理流水线 ---
    if "raw_df" in st.session_state:
        raw_df = st.session_state["raw_df"]
        
        # -------------------------------------------------------
        # 【Step 2】原始 SQL + 原始数据清单预览
        # -------------------------------------------------------
        with st.expander("查看原始 SQL"):
            st.code(st.session_state.get("current_sql", ""), language="sql")

        # 原始数据预览：展示完整数据清单（几百行级别）
        with st.expander("📄 原始数据预览（查询结果）", expanded=False):
            st.dataframe(raw_df, width="stretch")
            st.download_button(
                "📥 下载原始数据",
                df_to_csv_bytes(raw_df, index=False),
                "raw_data.csv",
            )

        st.divider()

        # -------------------------------------------------------
        # 【Pass 1: 预计算】
        # 先算一遍衍生变量 (如 Total)，为了让基线配置能选到它们
        # -------------------------------------------------------
        df_pass1 = apply_calculations(raw_df, st.session_state["calc_rules"])
        all_cols_pass1 = list(df_pass1.columns)

        # ==========================================
        # [Step A] 基线变量映射 (BDS Engine)
        # ==========================================
        st.subheader("🧬 基线变量映射 (BDS)")
        st.caption("在此定义基线，系统会自动生成 `_BL` 后缀变量。")
        
        bl_cfg = st.session_state.get("baseline_config", {})
        
        with st.expander("⚙️ 配置基线逻辑", expanded=not bool(bl_cfg)):
            c1, c2, c3 = st.columns(3)
            # 智能猜测
            def_subj_idx = next((i for i, c in enumerate(all_cols_pass1) if "SUBJ" in c.upper()), 0)
            def_visit_idx = next((i for i, c in enumerate(all_cols_pass1) if "VISIT" in c.upper() or "AVISIT" in c.upper()), 0)

            with c1:
                subj_col = st.selectbox("受试者 ID 列", all_cols_pass1, index=def_subj_idx, key="bl_subj_ui")
            with c2:
                visit_col = st.selectbox("访视/时间点列", all_cols_pass1, index=def_visit_idx, key="bl_visit_ui")
            
            # 动态获取访视列表
            if visit_col and visit_col in df_pass1.columns:
                unique_visits = sorted(df_pass1[visit_col].dropna().astype(str).unique().tolist())
            else:
                unique_visits = []
                
            with c3:
                try:
                    saved_bl_val = bl_cfg.get("baseline_val")
                    bl_idx = unique_visits.index(saved_bl_val) if saved_bl_val in unique_visits else 0
                except:
                    bl_idx = 0
                baseline_val = st.selectbox("哪一个访视是基线?", unique_visits, index=bl_idx, key="bl_val_ui")
            
            target_cols = st.multiselect(
                "选择数值变量 (生成 _BL 列)", 
                options=all_cols_pass1,
                default=[c for c in bl_cfg.get("target_cols", []) if c in all_cols_pass1],
                key="bl_targets_ui"
            )
            
            if st.button("✅ 应用基线配置"):
                st.session_state["baseline_config"] = {
                    "subj_col": subj_col, "visit_col": visit_col,
                    "baseline_val": baseline_val, "target_cols": target_cols
                }
                st.rerun()

        if st.session_state.get("baseline_config"):
            targets = st.session_state["baseline_config"].get("target_cols", [])
            if targets:
                st.info(f"已生成变量: {', '.join([t+'_BL' for t in targets])}")

        st.divider()

        # ==========================================
        # [Step B] 衍生变量计算
        # ==========================================
        st.subheader("🧮 衍生变量计算")
        
        # 模拟基线映射以获取列名
        df_preview_bl = apply_baseline_mapping(df_pass1, st.session_state.get("baseline_config", {}))
        current_cols = list(df_preview_bl.columns) + [r['name'] for r in st.session_state["calc_rules"]]
        
        with st.expander("➕ 添加新计算规则", expanded=True):
            c1, c2, c3, c4 = st.columns([2, 3, 2, 1])
            with c1: 
                new_name = st.text_input("新变量名", placeholder="例: Change_Score")
            with c2: 
                targets_sel = st.multiselect("参与计算的列", options=current_cols)
            with c3: 
                method = st.selectbox("计算方式", options=list(CALC_METHODS.keys()))
            with c4:
                st.write(""); st.write("")
                if st.button("添加"):
                    if new_name and targets_sel:
                        st.session_state["calc_rules"].append({
                            "name": new_name, "cols": targets_sel, "method": method
                        })
                        st.rerun()

        if st.session_state["calc_rules"]:
            for i, rule in enumerate(st.session_state["calc_rules"]):
                c1, c2 = st.columns([8, 1])
                c1.markdown(f"**Step {i+1}:** `{rule['name']}` = **{rule['method']}** ({', '.join(rule['cols'])})")
                if c2.button("🗑️", key=f"del_rule_{i}"):
                    st.session_state["calc_rules"].pop(i)
                    st.rerun()

        # ==========================================
        # [Step C] 数据剔除
        # ==========================================
        st.divider()
        st.markdown("##### 🗑️ 数据剔除规则")
        
        with st.expander("配置剔除条件"):
            ec1, ec2 = st.columns([2, 3])
            cur_excl = st.session_state.get("exclusions", [])
            def_field = cur_excl[0]["field"] if cur_excl else (current_cols[0] if current_cols else None)
            def_vals = cur_excl[0]["values"] if cur_excl else []
            
            with ec1:
                try: f_idx = current_cols.index(def_field) if def_field in current_cols else 0
                except: f_idx = 0
                excl_field = st.selectbox("字段名", current_cols, index=f_idx, key="ex_f")
            
            with ec2:
                if excl_field and excl_field in df_preview_bl.columns:
                    u_vals = df_preview_bl[excl_field].astype(str).unique().tolist()[:200]
                    excl_values = st.multiselect("剔除值 (Not In)", u_vals, default=def_vals, key="ex_v")
                else:
                    excl_values = []

            if excl_values:
                st.session_state["exclusions"] = [{"field": excl_field, "values": excl_values}]
            else:
                st.session_state["exclusions"] = []
                
        if st.session_state.get("exclusions"):
            r = st.session_state["exclusions"][0]
            st.info(f"当前剔除: `{r['field']}` NOT IN {r['values']}")

        # ==========================================
        # [Step D] 备注 & 保存配置
        # ==========================================
        st.markdown("##### 📝 备注")
        st.text_area("分析备注", key="calc_note", height=80)

        st.divider()
        if st.button("💾 保存所有配置"):
            payload = {
                "baseline": st.session_state.get("baseline_config", {}),
                "calc_rules": st.session_state.get("calc_rules", []),
                "note": st.session_state.get("calc_note", ""),
                "exclusions": st.session_state.get("exclusions", []),
                "pivot": {
                    "index": st.session_state.get("pivot_index", []),
                    "columns": st.session_state.get("pivot_columns", []),
                    "values": st.session_state.get("pivot_values", []),
                    "agg": st.session_state.get("pivot_aggs", ["Mean - 平均值"]),
                    "view": st.session_state.get("pivot_view_mode", "classic"),
                    "row_order": {
                        "field": st.session_state.get("pivot_row_order_field"),
                        "values": st.session_state.get(
                            "pivot_row_order_values", []
                        ),
                    },
                },
            }
            save_calculation_config(selected_row["setup_name"], payload)
            st.success("配置已保存！")

        # =======================================================
        # 【最终执行流水线】Pass 1 -> BDS -> Filter -> Pass 2
        # =======================================================
        final_df = raw_df.copy()
        # 1. Pass 1 计算
        final_df = apply_calculations(final_df, st.session_state["calc_rules"])
        # 2. 基线映射
        final_df = apply_baseline_mapping(final_df, st.session_state.get("baseline_config", {}))
        # 3. 剔除
        if st.session_state.get("exclusions"):
            for rule in st.session_state["exclusions"]:
                f, vals = rule.get("field"), rule.get("values")
                if f and f in final_df.columns and vals:
                    final_df = final_df[~final_df[f].astype(str).isin([str(v) for v in vals])]
        # 4. Pass 2 计算 (Change 规则生效)
        final_df = apply_calculations(final_df, st.session_state["calc_rules"])

        # ==========================================
        # [Step E] 透视分析 & 统计检验 & 绘图
        # ==========================================
        st.divider()
        st.subheader("📊 透视分析 & 统计检验")

        # 数据预览
        with st.expander("📄 最终数据预览"):
            st.dataframe(final_df.head(100), width="stretch")
            st.download_button(
                "📥 下载最终数据",
                df_to_csv_bytes(final_df, index=False),
                "final_data.csv",
            )

        all_final_cols = list(final_df.columns)

        def normalize_pivot_selection(key: str) -> None:
            cur = st.session_state.get(key, [])
            if isinstance(cur, str):
                cur_list = [cur]
            elif cur is None:
                cur_list = []
            elif isinstance(cur, (list, tuple, set)):
                cur_list = list(cur)
            else:
                cur_list = [cur]
            st.session_state[key] = [c for c in cur_list if c in all_final_cols]

        normalize_pivot_selection("pivot_index")
        normalize_pivot_selection("pivot_columns")
        normalize_pivot_selection("pivot_values")

        def sync_pivot_row_order(
            field: str, available_values: list[str]
        ) -> list[str]:
            stored_field = st.session_state.get("pivot_row_order_field")
            stored_values = st.session_state.get("pivot_row_order_values", [])
            if not isinstance(stored_values, list):
                stored_values = list(stored_values)

            if stored_field != field:
                st.session_state["pivot_row_order_field"] = field
                st.session_state["pivot_row_order_values"] = list(available_values)
                return st.session_state["pivot_row_order_values"]

            new_order = [v for v in stored_values if v in available_values]
            new_order.extend([v for v in available_values if v not in new_order])
            st.session_state["pivot_row_order_values"] = new_order
            return new_order
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            idx = st.multiselect("行维度 (如 Visit)", all_final_cols, key="pivot_index")
        with c2:
            col = st.multiselect("列维度 (如 Group)", all_final_cols, key="pivot_columns")
        with c3:
            val = st.multiselect("值字段 (如 Score)", all_final_cols, key="pivot_values")
        with c4:
            agg_options = list(AGG_METHODS.keys())
            default_aggs = [
                a for a in st.session_state.get("pivot_aggs", ["Mean - 平均值"])
                if a in agg_options
            ]
            if not default_aggs:
                default_aggs = ["Mean - 平均值"]
            aggs = st.multiselect(
                "聚合函数（可多选）",
                agg_options,
                default=default_aggs,
                key="pivot_aggs",
            )

        view_labels = {"classic": "经典透视表", "nested": "嵌套透视表"}
        view_options = list(view_labels.values())
        current_view = st.session_state.get("pivot_view_mode", "classic")
        current_label = view_labels.get(current_view, view_options[0])
        try:
            view_index = view_options.index(current_label)
        except ValueError:
            view_index = 0
        view_choice = st.radio(
            "透视表视图",
            view_options,
            index=view_index,
            horizontal=True,
        )
        selected_view = next(
            key for key, label in view_labels.items() if label == view_choice
        )
        st.session_state["pivot_view_mode"] = selected_view

        row_order_values = None
        if idx:
            first_field = idx[0]
            if first_field in final_df.columns:
                available_values = (
                    final_df[first_field]
                    .dropna()
                    .astype(str)
                    .drop_duplicates()
                    .tolist()
                )
            else:
                available_values = []
            row_order_values = sync_pivot_row_order(
                first_field, available_values
            )

            with st.expander(f"行维度顺序（{first_field}）", expanded=False):
                if not row_order_values:
                    st.caption("暂无可排序的值。")
                else:
                    selected_value = st.selectbox(
                        "选择要移动的值",
                        row_order_values,
                        key="pivot_row_order_selected",
                    )
                    move_up, move_down = st.columns(2)
                    if move_up.button("上移", key="pivot_row_order_up"):
                        new_order = list(row_order_values)
                        idx_pos = new_order.index(selected_value)
                        if idx_pos > 0:
                            new_order[idx_pos - 1], new_order[idx_pos] = (
                                new_order[idx_pos],
                                new_order[idx_pos - 1],
                            )
                            st.session_state["pivot_row_order_values"] = new_order
                            row_order_values = new_order
                    if move_down.button("下移", key="pivot_row_order_down"):
                        new_order = list(row_order_values)
                        idx_pos = new_order.index(selected_value)
                        if idx_pos < len(new_order) - 1:
                            new_order[idx_pos + 1], new_order[idx_pos] = (
                                new_order[idx_pos],
                                new_order[idx_pos + 1],
                            )
                            st.session_state["pivot_row_order_values"] = new_order
                            row_order_values = new_order
                    st.caption("当前顺序：" + " → ".join(row_order_values))

        if idx and col and val and aggs:
            # 1. 透视表
            try:
                view_mode = st.session_state.get("pivot_view_mode", "classic")
                if view_mode == "nested":
                    nested_data = render_pivot_nested(
                        final_df,
                        index_cols=idx,
                        column_cols=col,
                        value_cols=val,
                        agg_names=aggs,
                        row_order=row_order_values,
                    )
                    st.download_button(
                        "📥 下载嵌套透视表（Excel）",
                        nested_pivot_to_excel_bytes(nested_data),
                        "pivot_table_nested.xlsx",
                    )
                else:
                    pivot = render_pivot_classic(
                        final_df,
                        index_cols=idx,
                        column_cols=col,
                        value_cols=val,
                        agg_names=aggs,
                        row_order=row_order_values,
                    )
                    st.download_button(
                        "📥 下载透视表",
                        df_to_csv_bytes(pivot, index=True),
                        "pivot_table_multi_agg.csv",
                    )
            except Exception as e:
                st.error(f"透视失败: {e}")

            # 2. [自动化] 组间差异检验 (ANOVA)
            # 自动使用透视表的配置：Index=分层, Col=分组, Val=数值
            if len(idx) == 1 and len(col) == 1 and len(val) == 1:
                st.markdown("#### 📉 组间差异检验 (One-Way ANOVA)")
                st.caption(f"自动计算：按 **{idx[0]}** 分层，比较不同 **{col[0]}** 组别之间的 **{val[0]}** 差异。")
                
                anova_df = calculate_anova_table(
                    final_df, 
                    index_col=idx[0], 
                    group_col=col[0], 
                    value_col=val[0]
                )
                st.dataframe(anova_df, width="stretch")

            # 3. 绘图（支持多行维度 / 多列维度，按迪卡尔积生成单元格）
            if val:
                if len(val) > 1:
                    st.info("当前图表仅支持单一值字段绘图，请在“值字段”中只选择一个。")
                else:
                    st.markdown("---")
                    st.subheader("📈 单元格分布图")

                    # 预留一个位置用于显示“已生成 X 个图表（时间）”的提示
                    charts_info_placeholder = st.empty()

                    # 收集当前页面实际绘制的所有图表，用于 HTML 导出
                    all_figs: list[dict[str, Any]] = []

                    # 计算行维度和列维度的所有组合键（多维）
                    row_key_cols = idx
                    col_key_cols = col

                    if row_key_cols:
                        row_keys_df = (
                            final_df[row_key_cols]
                            .dropna()
                            .astype(str)
                            .drop_duplicates()
                        )
                        row_keys = row_keys_df.to_dict(orient="records")
                    else:
                        row_keys = [{}]

                    if col_key_cols:
                        col_keys_df = (
                            final_df[col_key_cols]
                            .dropna()
                            .astype(str)
                            .drop_duplicates()
                        )
                        col_keys = col_keys_df.to_dict(orient="records")
                    else:
                        col_keys = [{}]

                    total_charts = len(row_keys) * len(col_keys)
                    if total_charts == 0:
                        st.info("当前透视配置下没有可用于绘图的单元格。")
                    else:
                        max_charts = 120
                        if total_charts > max_charts:
                            st.warning(
                                f"⚠️ 图表数量较多（{total_charts} 个）。"
                                f" 默认仅展示前 {max_charts} 个，可勾选下方选项加载全部。"
                            )
                            render_all = st.checkbox(
                                f"加载全部 {total_charts} 个图表（可能较慢）",
                                key="charts_render_all",
                            )
                            limit = total_charts if render_all else max_charts
                        else:
                            limit = total_charts

                        count = 0
                        def_id_idx = next(
                            (i for i, c in enumerate(all_final_cols) if "SUBJ" in c.upper()),
                            0,
                        )
                        subj_col = st.selectbox(
                            "ID 列 (用于绘图)", all_final_cols, index=def_id_idx
                        )
                        value_col = val[0]
                        chart_type = st.radio(
                            "图表类型",
                            ["经典", "统一坐标"],
                            horizontal=True,
                            key="chart_type_mode",
                        )

                        use_uniform_chart = chart_type == "统一坐标"
                        uniform_x_range = None
                        uniform_y_max = None
                        if use_uniform_chart:
                            st.markdown(
                                """
                                <style>
                                div[data-testid="stPlotlyChart"] > div {
                                    width: 100% !important;
                                    aspect-ratio: 1 / 1;
                                }
                                div[data-testid="stPlotlyChart"] .js-plotly-plot,
                                div[data-testid="stPlotlyChart"] .plot-container,
                                div[data-testid="stPlotlyChart"] .svg-container {
                                    width: 100% !important;
                                    height: 100% !important;
                                }
                                </style>
                                """,
                                unsafe_allow_html=True,
                            )
                            uniform_x_range, uniform_y_max = compute_uniform_axes(
                                final_df, row_key_cols, col_key_cols, value_col
                            )
                            if uniform_y_max <= 0:
                                uniform_x_range = None
                                uniform_y_max = None

                        agg_names_for_plot = aggs[:2]
                        agg_funcs_for_plot = [
                            AGG_METHODS.get(name) for name in agg_names_for_plot
                        ]
                        # 绘图使用的聚合函数：取多选聚合函数中的第一个作为参考线
                        primary_agg_name = aggs[0] if aggs else "Mean - 平均值"
                        actual_func_for_plot = AGG_METHODS.get(primary_agg_name, "mean")

                    # 为每个行组合分配一个固定颜色，使同一行组合下不同列维度的图表颜色一致
                    color_palette = [
                        "#1f77b4",
                        "#ff7f0e",
                        "#2ca02c",
                        "#d62728",
                        "#9467bd",
                        "#8c564b",
                        "#e377c2",
                        "#7f7f7f",
                        "#bcbd22",
                        "#17becf",
                    ]

                    max_cols_per_row = 3

                    def render_cell_chart(
                        row_key: dict,
                        col_key: dict,
                        row_idx: int,
                        col_idx: int,
                        chart_color: str,
                    ) -> None:
                        nonlocal count

                        cell = final_df
                        for col_name, v in row_key.items():
                            cell = cell[cell[col_name].astype(str) == v]
                        for col_name, v in col_key.items():
                            cell = cell[cell[col_name].astype(str) == v]

                        if cell.empty:
                            return

                        title_parts = [
                            f"{k}={row_key[k]}" for k in row_key_cols if k in row_key
                        ] + [
                            f"{k}={col_key[k]}" for k in col_key_cols if k in col_key
                        ]
                        title = "<br>".join(title_parts) if title_parts else "(All)"
                        title_html = "<br>".join(
                            [html.escape(p) for p in title_parts]
                        ) if title_parts else "(All)"
                        internal_title = ""
                        key_suffix = f"r{row_idx}_c{col_idx}"

                        if use_uniform_chart:
                            fig = build_uniform_spaghetti_fig(
                                df=cell,
                                subj_col=subj_col,
                                value_col=value_col,
                                title=internal_title,
                                x_range=uniform_x_range,
                                y_max_count=uniform_y_max,
                                agg_funcs=agg_funcs_for_plot,
                                agg_names=agg_names_for_plot,
                                marker_color=chart_color,
                            )
                        else:
                            fig = build_spaghetti_fig(
                                df=cell,
                                subj_col=subj_col,
                                value_col=value_col,
                                title=internal_title,
                                agg_func=actual_func_for_plot,
                                agg_name=primary_agg_name,
                                marker_color=chart_color,
                            )
                        if fig is None:
                            return

                        st.markdown(
                            (
                                "<div style='text-align:center;"
                                "font-weight:600;font-size:16px;"
                                "line-height:1.2;margin-bottom:8px;'>"
                                f"{title_html}</div>"
                            ),
                            unsafe_allow_html=True,
                        )

                        # -------------------------------------------------------
                        # 🚀 关键点 2: 深拷贝隔离 (Deep Copy Isolation)
                        # -------------------------------------------------------
                        # 在 render 之前，先克隆一份“干净”的 Figure 用于导出。
                        # 这样无论 st.plotly_chart 对 fig 做了什么(如注入JS回调)，
                        # 导出用的 fig_for_export 永远是纯净的。
                        fig_for_export = copy.deepcopy(fig)

                        legend_items = []
                        meta = getattr(fig.layout, "meta", None)
                        if isinstance(meta, dict):
                            legend_items = meta.get("legend_items", [])

                        if use_uniform_chart:
                            render_uniform_spaghetti_fig(
                                fig, key=f"c_{key_suffix}"
                            )
                            if legend_items:
                                legend_lines = []
                                for item in legend_items:
                                    dash_style = (
                                        "dashed"
                                        if item.get("dash") == "dash"
                                        else "solid"
                                    )
                                    label_text = html.escape(
                                        str(item.get("label", "Agg"))
                                    )
                                    value_text = item.get("value")
                                    try:
                                        value_fmt = f"{float(value_text):.2f}"
                                    except Exception:
                                        value_fmt = "-"
                                    legend_lines.append(
                                        "<div style='display:flex;"
                                        "justify-content:center;align-items:center;"
                                        "gap:8px;font-size:12px;color:#c00;"
                                        "line-height:1.2;margin-top:2px;'>"
                                        f"<span style='display:inline-block;"
                                        f"width:32px;border-top:3px {dash_style} #c00;'></span>"
                                        f"<span>{label_text}: {value_fmt}</span>"
                                        "</div>"
                                    )
                                st.markdown(
                                    (
                                        "<div style='margin-top:4px;'>"
                                        + "".join(legend_lines)
                                        + "</div>"
                                    ),
                                    unsafe_allow_html=True,
                                )
                        else:
                            render_spaghetti_fig(fig, key=f"c_{key_suffix}")

                        all_figs.append(
                            {
                                "title": title,
                                "title_html": title_html,
                                "fig": fig_for_export,
                                "legend_items": legend_items,
                                "chart_type": (
                                    "uniform" if use_uniform_chart else "classic"
                                ),
                            }
                        )
                        count += 1

                    stop_render = False
                    for i, rk in enumerate(row_keys):
                        if stop_render:
                            break
                        group_color = color_palette[i % len(color_palette)]

                        if use_uniform_chart:
                            for chunk_start in range(0, len(col_keys), max_cols_per_row):
                                if stop_render:
                                    break
                                chunk = col_keys[
                                    chunk_start : chunk_start + max_cols_per_row
                                ]
                                cols = st.columns(max_cols_per_row)
                                for col_pos, ck in enumerate(chunk):
                                    if count >= limit:
                                        stop_render = True
                                        break
                                    j = chunk_start + col_pos
                                    with cols[col_pos]:
                                        render_cell_chart(
                                            rk, ck, i, j, group_color
                                        )
                        else:
                            for j, ck in enumerate(col_keys):
                                if count >= limit:
                                    stop_render = True
                                    break
                                render_cell_chart(rk, ck, i, j, group_color)

                    # 在图表区域顶部给出生成数量和时间提示
                    from datetime import datetime

                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    charts_info_placeholder.caption(
                        f"已为您生成 {count} 个图表（{ts})"
                    )

                    # 4. 一键导出当前所有图表为 HTML
                    if count > 0 and all_figs:
                        if st.button("📥 下载所有图表 (HTML)", key="btn_export_charts"):
                            html_blocks: list[str] = []

                            # 【DEBUG START】 打印第一张图的 X 轴数据，看看是数值还是下标
                            if all_figs:
                                first_fig = all_figs[0]["fig"]
                                # 尝试获取 X 轴数据（通常在 data[0].x）
                                try:
                                    x_sample = first_fig.data[0].x
                                    print(f"--- [DEBUG] Export Check ---")
                                    print(f"First Chart Title: {all_figs[0]['title']}")
                                    print(f"X Data Type: {type(x_sample)}")
                                    # 打印前 10 个值
                                    print(f"X Data Sample: {list(x_sample)[:10] if hasattr(x_sample, '__iter__') else x_sample}")
                                    print(f"----------------------------")
                                except Exception as e:
                                    print(f"--- [DEBUG] Error reading x data: {e} ---")
                            # 【DEBUG END】

                            full_html = build_charts_export_html(all_figs)

                            st.download_button(
                                "⬇️ 保存为 HTML 文件",
                                data=full_html.encode("utf-8"),
                                file_name="all_charts.html",
                                mime="text/html",
                                key="btn_export_charts_download",
                            )

                    # 5. 点击散点后展示选中受试者的完整明细
                    selected_id = st.session_state.get("selected_subject_id")
                    if selected_id is not None:
                        st.markdown("---")
                        st.subheader(f"📄 受试者明细：{selected_id}")

                        if subj_col in final_df.columns:
                            subj_df = final_df[
                                final_df[subj_col].astype(str) == str(selected_id)
                            ]
                            if subj_df.empty:
                                st.info("当前数据集中未找到该受试者的记录。")
                            else:
                                st.dataframe(subj_df, width="stretch")
                        else:
                            st.info(
                                f"当前数据中不存在受试者列 `{subj_col}`，无法展示明细。"
                            )

                        # 提供跳转到受试者档案页面的入口
                        def build_subject_profile_url(subject_id: Any) -> str:
                            base_path = st_config.get_option("server.baseUrlPath") or ""
                            base_prefix = f"/{base_path.strip('/')}" if base_path else ""
                            query = urlencode({"subject_id": str(subject_id)})
                            return f"{base_prefix}/subject_profile?{query}"
                        
                        st.link_button(
                            "🔍 在新标签页打开受试者档案",
                            build_subject_profile_url(selected_id),
                        )

if __name__ == "__main__":
    main()
