"""Streamlit analysis dashboard page."""

import copy
import html
import math
from typing import Any

import pandas as pd
import streamlit as st

from analysis.auth.session import require_login
from analysis.plugins.methods import CALC_METHODS, AGG_METHODS
from analysis.plugins.methods.agg import compute_trimmed_mean
from analysis.plugins.charts.boxplot import (
    build_boxplot_matrix_fig,
    compute_boxplot_range,
    render_boxplot_fig,
)
from analysis.plugins.charts.lineplot import build_pivot_line_fig, render_line_fig
from analysis.plugins.charts.uniform import (
    build_uniform_spaghetti_fig,
    compute_uniform_axes,
    render_uniform_spaghetti_fig,
    resolve_uniform_control_group,
)
from analysis.plugins.charts.uniform_min_max import (
    build_uniform_spaghetti_fig as build_uniform_min_max_spaghetti_fig,
    compute_uniform_axes as compute_uniform_min_max_axes,
    render_uniform_spaghetti_fig as render_uniform_min_max_spaghetti_fig,
)
from analysis.plugins.charts.uniform_log import (
    build_uniform_spaghetti_fig as build_uniform_log_spaghetti_fig,
    compute_uniform_axes as compute_uniform_log_axes,
    render_uniform_spaghetti_fig as render_uniform_log_spaghetti_fig,
)
from analysis.exports.charts import build_charts_export_html
from analysis.exports.common import df_to_csv_bytes
from analysis.exports.pivot import nested_pivot_to_excel_bytes
from setup_catalog.services.analysis_list_setups import (
    fetch_all_setups,
    fetch_setup_config,
    save_calculation_config,
)
from analysis.services.analysis_service import (
    apply_baseline_mapping,
    apply_calculations,
    calculate_anova_table,
    run_analysis,
)
from analysis.services.calculation_graph import (
    build_dependency_rows,
    build_graphviz_dot,
    run_calculation_graph,
)
from analysis.services.calculation_config import (
    apply_calculation_config,
    build_calculation_payload,
    cascade_delete_targets,
)
from analysis.settings.logging import log_access
from analysis.state.dashboard import reset_dashboard_state
from analysis.views.pivot_nested import render_pivot_nested
from analysis.views.components.page_utils import (
    build_page_url,
    hide_login_sidebar_entry,
    render_sidebar_navigation,
)

page_title = st.session_state.get("page_title") or "Analysis Dashboard"
DEFAULT_PIVOT_AGGS = ["Mean - 平均值"]
st.set_page_config(page_title=page_title, layout="wide")
hide_login_sidebar_entry()
st.title(f"📊 {page_title}")
st.markdown(
    """
    <style>
    section.main > div.block-container {
        max-width: 100%;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    @media (max-width: 800px) {
        #pivot-dim-row-marker + div[data-testid="stHorizontalBlock"],
        #pivot-metric-row-marker + div[data-testid="stHorizontalBlock"] {
            flex-direction: column;
        }
        #pivot-dim-row-marker + div[data-testid="stHorizontalBlock"] > div,
        #pivot-metric-row-marker + div[data-testid="stHorizontalBlock"] > div {
            width: 100% !important;
            flex: 1 1 100% !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# UI 表现层 (Main)
# ==========================================

def main() -> None:
    """Render the main analysis dashboard page."""
    require_login()
    log_access("analysis_dashboard")
    # --- 1. 侧边栏 ---
    with st.sidebar:
        render_sidebar_navigation(active_page="analysis_dashboard")
        st.header("🧩 分析集配置")
        setups = fetch_all_setups()

        if not setups:
            st.info("暂无配置。请先去主页创建。")
            return

        option_labels = [f"{row['setup_name']}" for row in setups]
        query_setup_param = st.query_params.get("setup_name")
        if isinstance(query_setup_param, list):
            query_setup_param = query_setup_param[0] if query_setup_param else None
        query_setup = query_setup_param
        if not query_setup:
            query_setup = st.session_state.pop("jump_setup", None)
        default_index = 0
        if query_setup in option_labels:
            default_index = option_labels.index(query_setup)
        selected_label = st.selectbox(
            "选择配置",
            options=option_labels,
            index=default_index,
        )
        selected_row = next(r for r in setups if f"{r['setup_name']}" == selected_label)
        
        if selected_row.get("description"):
            st.info(f"📝 **备注**: {selected_row['description']}")

        st.markdown("##### 📝 备注")
        default_note = st.session_state.get("calc_note", "")
        st.text_area(
            "分析备注",
            value=default_note,
            key="calc_note_input",
            height=80,
        )

        if st.button("💾 保存所有配置", key="save_all_config"):
            payload = build_calculation_payload(
                st.session_state,
                default_agg=DEFAULT_PIVOT_AGGS,
            )
            save_calculation_config(selected_row["setup_name"], payload)
            st.success("配置已保存！")

    # --- 1.1 状态管理与初始化 ---
    st.session_state["current_setup_name"] = selected_row["setup_name"]

    # --- 2. 加载源数据 ---
    load_clicked = st.button("🚀 加载源数据", type="primary")
    auto_load = False
    if query_setup_param and query_setup_param == selected_row["setup_name"]:
        last_auto = st.session_state.get("auto_loaded_setup_name")
        if last_auto != query_setup_param:
            auto_load = True
            st.session_state["auto_loaded_setup_name"] = query_setup_param

    if load_clicked or auto_load:
        full_cfg = fetch_setup_config(selected_row["setup_name"]) or {}
        calc_cfg = full_cfg.get("calculation") or {}

        # 重置 UI 缓存，确保完全使用数据库配置
        reset_dashboard_state()

        # 覆盖缓存为数据库配置
        apply_calculation_config(
            st.session_state,
            calc_cfg,
            default_agg=DEFAULT_PIVOT_AGGS,
        )

        st.session_state.pop("raw_df", None)
        st.session_state.pop("current_sql", None)
        st.session_state.pop("selected_subject_id", None)

        if full_cfg.get("extraction"):
            sql, df_res = run_analysis(full_cfg["extraction"])
            if not df_res.empty:
                st.session_state["raw_df"] = df_res
                st.session_state["current_sql"] = sql
                st.success(f"加载成功！共 {len(df_res)} 行。")
            else:
                st.warning("查询结果为空。")

        new_title = selected_row["setup_name"]
        if st.session_state.get("page_title") != new_title:
            st.session_state["page_title"] = new_title
            st.rerun()

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
                old_cfg = st.session_state.get("baseline_config", {}) or {}
                old_targets = set(old_cfg.get("target_cols", []) or [])
                new_targets = set(target_cols)
                removed_targets = sorted(old_targets - new_targets)

                calc_payload = build_calculation_payload(
                    st.session_state,
                    default_agg=DEFAULT_PIVOT_AGGS,
                )
                if removed_targets:
                    outputs = [f"{t}_BL" for t in removed_targets]
                    updated_cfg, removed = cascade_delete_targets(
                        calc_payload, outputs
                    )
                else:
                    updated_cfg = calc_payload
                    removed = {"outputs": []}

                updated_cfg["baseline"] = {
                    "subj_col": subj_col,
                    "visit_col": visit_col,
                    "baseline_val": baseline_val,
                    "target_cols": target_cols,
                }
                apply_calculation_config(
                    st.session_state,
                    updated_cfg,
                    default_agg=DEFAULT_PIVOT_AGGS,
                )
                if removed.get("outputs"):
                    st.info(
                        "已级联删除依赖变量: "
                        + ", ".join(removed["outputs"])
                    )
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

        def build_default_agg_name(
            group_by: list[str], col: str | None, func: Any | None
        ) -> str:
            """Build a default output name for aggregation rules."""
            if not col or not func:
                return ""
            func_name = str(func).strip().replace(" ", "_")
            group_part = "_".join([str(v) for v in group_by if v]) or "all"
            return f"{func_name}_{col}_by_{group_part}"

        def collect_agg_outputs(rules: list[dict[str, Any]]) -> list[str]:
            """Collect aggregation output names for column selection."""
            outputs: list[str] = []
            for rule in rules:
                if not isinstance(rule, dict):
                    continue
                group_by = rule.get("group_by", [])
                if not isinstance(group_by, list):
                    group_by = [group_by] if group_by else []
                metrics = rule.get("metrics", [])
                if not isinstance(metrics, list):
                    continue
                for metric in metrics:
                    if not isinstance(metric, dict):
                        continue
                    name = metric.get("name")
                    if not name:
                        name = build_default_agg_name(
                            group_by,
                            metric.get("col"),
                            metric.get("fn") or metric.get("func"),
                        )
                    if name:
                        outputs.append(name)
            return outputs

        # 模拟基线映射以获取列名
        df_preview_bl = apply_baseline_mapping(df_pass1, st.session_state.get("baseline_config", {}))
        agg_outputs = collect_agg_outputs(st.session_state.get("aggregations", []))
        agg_input_cols = list(df_preview_bl.columns)
        current_cols = list(df_preview_bl.columns)
        current_cols.extend([r.get("name") for r in st.session_state["calc_rules"] if r.get("name")])
        current_cols.extend(agg_outputs)
        current_cols = list(dict.fromkeys([c for c in current_cols if c]))
        
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
                    calc_payload = build_calculation_payload(
                        st.session_state,
                        default_agg=DEFAULT_PIVOT_AGGS,
                    )
                    node_id = f"derive_{i + 1}"
                    updated_cfg, removed = cascade_delete_targets(
                        calc_payload, [node_id]
                    )
                    apply_calculation_config(
                        st.session_state,
                        updated_cfg,
                        default_agg=DEFAULT_PIVOT_AGGS,
                    )
                    if removed.get("outputs"):
                        st.info(
                            "已级联删除依赖变量: "
                            + ", ".join(removed["outputs"])
                        )
                    st.rerun()

        # ==========================================
        # [Step C] 数据剔除
        # ==========================================
        st.divider()
        st.markdown("##### 🗑️ 数据剔除规则")
        st.caption("多条规则之间为 AND 关系。")

        def normalize_exclusion_op(op: Any) -> str:
            """Normalize exclusion operators for UI."""
            raw = str(op or "not_in").strip().lower()
            raw = raw.replace(" ", "_")
            mapping = {
                "notin": "not_in",
                "not-in": "not_in",
                "=": "eq",
                "==": "eq",
                "!=": "ne",
                "<>": "ne",
                ">": "gt",
                ">=": "ge",
                "≥": "ge",
                "<": "lt",
                "<=": "le",
                "≤": "le",
                "like": "like",
                "contains": "like",
                "is_null": "is_null",
                "null": "is_null",
                "empty": "is_null",
                "isnull": "is_null",
                "为空": "is_null",
                "is_not_null": "is_not_null",
                "not_null": "is_not_null",
                "notnull": "is_not_null",
                "not_empty": "is_not_null",
                "不为空": "is_not_null",
            }
            return mapping.get(raw, raw)

        op_choices = [
            ("=", "eq"),
            ("!=", "ne"),
            ("in", "in"),
            ("not in", "not_in"),
            (">", "gt"),
            (">=", "ge"),
            ("<", "lt"),
            ("<=", "le"),
            ("like", "like"),
            ("为空", "is_null"),
            ("不为空", "is_not_null"),
        ]
        op_by_label = {label: op for label, op in op_choices}
        label_by_op = {op: label for label, op in op_choices}

        exclusions = st.session_state.get("exclusions", [])
        if not isinstance(exclusions, list):
            exclusions = []

        with st.expander("配置剔除条件", expanded=True):
            new_rules: list[dict[str, Any]] = []
            delete_idx: int | None = None

            if not exclusions:
                st.caption("暂无剔除条件。")

            for i, rule in enumerate(exclusions):
                rule = dict(rule or {})
                rule_op = normalize_exclusion_op(rule.get("op", "not_in"))
                rule_values = rule.get("values", [])
                if not isinstance(rule_values, list):
                    rule_values = [rule_values] if rule_values is not None else []

                c1, c2, c3, c4 = st.columns([2, 2, 5, 1])

                field_options = current_cols if current_cols else ["(无可用字段)"]
                field_disabled = not current_cols
                field = rule.get("field")
                try:
                    field_idx = field_options.index(field) if field in field_options else 0
                except ValueError:
                    field_idx = 0

                with c1:
                    selected_field = st.selectbox(
                        "字段名",
                        field_options,
                        index=field_idx,
                        key=f"ex_field_{i}",
                        disabled=field_disabled,
                    )
                if field_disabled:
                    selected_field = None

                label = label_by_op.get(rule_op, "not in")
                try:
                    op_idx = [lbl for lbl, _ in op_choices].index(label)
                except ValueError:
                    op_idx = 1

                with c2:
                    selected_label = st.selectbox(
                        "条件",
                        [lbl for lbl, _ in op_choices],
                        index=op_idx,
                        key=f"ex_op_{i}",
                    )
                selected_op = op_by_label.get(selected_label, "not_in")

                values: list[Any] = []
                with c3:
                    if selected_op in {"in", "not_in"}:
                        if selected_field and selected_field in df_preview_bl.columns:
                            u_vals = (
                                df_preview_bl[selected_field]
                                .astype(str)
                                .unique()
                                .tolist()[:200]
                            )
                        else:
                            u_vals = []
                        default_vals = [
                            str(v)
                            for v in rule_values
                            if str(v) in u_vals
                        ]
                        values = st.multiselect(
                            "值",
                            u_vals,
                            default=default_vals,
                            key=f"ex_vals_{i}",
                        )
                    elif selected_op in {"gt", "ge", "lt", "le", "eq", "ne"}:
                        default_val = str(rule_values[0]) if rule_values else ""
                        val = st.text_input(
                            "值",
                            value=default_val,
                            placeholder="例如 10",
                            key=f"ex_value_{i}",
                        )
                        values = [val] if val != "" else []
                    elif selected_op == "like":
                        default_val = str(rule_values[0]) if rule_values else ""
                        val = st.text_input(
                            "模式",
                            value=default_val,
                            placeholder="例如 %abc%",
                            key=f"ex_value_{i}",
                        )
                        values = [val] if val != "" else []
                    else:
                        st.caption("无需填写值")
                        values = []

                with c4:
                    if st.button("🗑️", key=f"ex_del_{i}"):
                        delete_idx = i

                new_rules.append(
                    {
                        "field": selected_field,
                        "op": selected_op,
                        "values": values,
                    }
                )

            if st.button("➕ 添加条件", key="ex_add_rule"):
                exclusions.append(
                    {
                        "field": current_cols[0] if current_cols else None,
                        "op": "not_in",
                        "values": [],
                    }
                )
                st.session_state["exclusions"] = exclusions
                st.rerun()

            if delete_idx is not None:
                st.session_state["exclusions"] = [
                    rule for idx, rule in enumerate(new_rules) if idx != delete_idx
                ]
                st.rerun()

            st.session_state["exclusions"] = new_rules

        if st.session_state.get("exclusions"):
            summaries = []
            for rule in st.session_state["exclusions"]:
                field = rule.get("field")
                op = normalize_exclusion_op(rule.get("op", "not_in"))
                values = rule.get("values", [])
                if not field:
                    continue
                label = label_by_op.get(op, op)
                if op in {"is_null", "is_not_null"}:
                    summaries.append(f"`{field}` {label}")
                else:
                    summaries.append(f"`{field}` {label} {values}")
            if summaries:
                st.info("当前剔除: " + " AND ".join(summaries))

        # ==========================================
        # [Step D] 聚合变量（广播）
        # ==========================================
        st.divider()
        st.subheader("🧮 聚合变量（广播）")
        st.caption("按分组列计算统计值，并将结果广播回原始数据。")

        agg_func_choices = [(name, name) for name in AGG_METHODS.keys()]
        agg_func_labels = [label for label, _ in agg_func_choices]
        agg_func_by_label = {label: func for label, func in agg_func_choices}
        agg_label_by_func = {func: label for label, func in agg_func_choices}

        aggregations = st.session_state.get("aggregations", [])
        if not isinstance(aggregations, list):
            aggregations = []

        with st.expander("配置聚合规则", expanded=True):
            new_rules: list[dict[str, Any]] = []
            delete_idx: int | None = None
            global_label = "全表"

            if not aggregations:
                st.caption("暂无聚合规则。")

            for i, rule in enumerate(aggregations):
                rule = dict(rule or {})
                group_by = rule.get("group_by", [])
                if not isinstance(group_by, list):
                    group_by = [group_by] if group_by else []

                metrics = rule.get("metrics", [])
                metric = metrics[0] if metrics and isinstance(metrics[0], dict) else {}
                metric_col = metric.get("col")
                metric_func = metric.get("fn") or metric.get("func") or "mean"
                metric_name = metric.get("name") or ""

                c1, c2, c3, c4, c5 = st.columns([3, 2, 2, 3, 1])

                with c1:
                    group_options = [global_label] + list(agg_input_cols)
                    default_group = (
                        [global_label]
                        if not group_by
                        else [c for c in group_by if c in agg_input_cols]
                    )
                    group_sel = st.multiselect(
                        "分组列",
                        options=group_options,
                        default=default_group,
                        key=f"agg_group_{i}",
                    )
                    if global_label in group_sel:
                        group_sel = []
                    else:
                        group_sel = [c for c in group_sel if c in agg_input_cols]

                field_options = agg_input_cols if agg_input_cols else ["(无可用字段)"]
                field_disabled = not agg_input_cols
                if metric_col not in field_options:
                    metric_col = field_options[0] if agg_input_cols else None
                try:
                    col_idx = field_options.index(metric_col) if metric_col in field_options else 0
                except ValueError:
                    col_idx = 0

                with c2:
                    metric_col_sel = st.selectbox(
                        "聚合列",
                        options=field_options,
                        index=col_idx,
                        key=f"agg_col_{i}",
                        disabled=field_disabled,
                    )
                if field_disabled:
                    metric_col_sel = None

                func_label = agg_label_by_func.get(metric_func, None)
                if func_label not in agg_func_labels:
                    func_label = agg_func_labels[0] if agg_func_labels else ""
                func_idx = agg_func_labels.index(func_label) if func_label in agg_func_labels else 0

                with c3:
                    func_label_sel = st.selectbox(
                        "函数",
                        options=agg_func_labels,
                        index=func_idx,
                        key=f"agg_func_{i}",
                    )
                metric_func_sel = agg_func_by_label.get(func_label_sel, "mean")

                default_name = build_default_agg_name(
                    group_sel,
                    metric_col_sel,
                    metric_func_sel,
                )
                with c4:
                    name_input = st.text_input(
                        "输出列名",
                        value=metric_name,
                        placeholder=default_name,
                        key=f"agg_name_{i}",
                    )
                final_name = name_input.strip() or default_name

                with c5:
                    if st.button("🗑️", key=f"agg_del_{i}"):
                        delete_idx = i

                new_rules.append(
                    {
                        "group_by": group_sel,
                        "metrics": [
                            {
                                "col": metric_col_sel,
                                "fn": metric_func_sel,
                                "name": final_name,
                            }
                        ],
                        "broadcast": True,
                    }
                )

            if st.button("➕ 添加聚合规则", key="agg_add_rule"):
                aggregations.append(
                    {
                        "group_by": [],
                        "metrics": [
                            {
                                "col": agg_input_cols[0] if agg_input_cols else None,
                                "fn": agg_func_choices[0][1],
                                "name": "",
                            }
                        ],
                        "broadcast": True,
                    }
                )
                st.session_state["aggregations"] = aggregations
                st.rerun()

            if delete_idx is not None:
                st.session_state["aggregations"] = [
                    rule for idx, rule in enumerate(new_rules) if idx != delete_idx
                ]
                st.rerun()

            st.session_state["aggregations"] = new_rules

        if st.session_state.get("aggregations"):
            summaries = []
            for rule in st.session_state["aggregations"]:
                if not isinstance(rule, dict):
                    continue
                group_by = rule.get("group_by") or []
                metrics = rule.get("metrics") or []
                if not metrics:
                    continue
                metric = metrics[0] if isinstance(metrics[0], dict) else {}
                col = metric.get("col")
                func = metric.get("fn") or metric.get("func")
                name = metric.get("name") or build_default_agg_name(group_by, col, func)
                if col and func:
                    group_label = ", ".join(group_by) if group_by else global_label
                    summaries.append(f"`{name}` = {func}({col}) BY {group_label}")
            if summaries:
                st.info("当前聚合: " + " | ".join(summaries))

        # =======================================================
        # 【最终执行流水线】基于 DAG 的统一执行
        # =======================================================
        calc_payload = build_calculation_payload(
            st.session_state,
            default_agg=DEFAULT_PIVOT_AGGS,
        )

        with st.expander("🧭 DAG 依赖图", expanded=False):
            graph = calc_payload.get("graph", {})
            dot = build_graphviz_dot(graph)
            chart_col, _ = st.columns([1, 2])
            try:
                chart_col.graphviz_chart(dot, use_container_width=True)
            except Exception as exc:
                st.warning(f"Graphviz render failed: {exc}")
                st.code(dot, language="dot")

            rows = build_dependency_rows(graph)
            if rows:
                st.markdown("##### 依赖列表")
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            else:
                st.info("暂无 DAG 节点。")

        final_df = run_calculation_graph(raw_df, calc_payload.get("graph", {}))

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
            """Normalize pivot selection session state to a valid column list."""
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
            """Sync row order values with the latest available options."""
            if not available_values:
                return []

            row_orders = st.session_state.get("pivot_row_orders", {})
            if not isinstance(row_orders, dict):
                row_orders = {}

            stored_values = row_orders.get(field)
            if not stored_values:
                row_orders[field] = list(available_values)
                st.session_state["pivot_row_orders"] = row_orders
                return row_orders[field]

            if not isinstance(stored_values, list):
                stored_values = list(stored_values)

            cleaned = [v for v in stored_values if v in available_values]
            missing = [v for v in available_values if v not in cleaned]
            if missing or len(cleaned) != len(stored_values):
                cleaned.extend(missing)
                row_orders[field] = cleaned
                st.session_state["pivot_row_orders"] = row_orders
            return cleaned

        def sync_pivot_col_order(
            field: str, available_values: list[str]
        ) -> list[str]:
            """Sync column order values with the latest available options."""
            col_order_map = st.session_state.get("pivot_col_order", {})
            if not isinstance(col_order_map, dict):
                col_order_map = {}

            if not available_values:
                return []

            stored_values = col_order_map.get(field)
            if not stored_values:
                col_order_map[field] = list(available_values)
                st.session_state["pivot_col_order"] = col_order_map
                return col_order_map[field]

            if not isinstance(stored_values, list):
                stored_values = list(stored_values)

            cleaned = [v for v in stored_values if v in available_values]
            missing = [v for v in available_values if v not in cleaned]
            if missing or len(cleaned) != len(stored_values):
                cleaned.extend(missing)
                col_order_map[field] = cleaned
                st.session_state["pivot_col_order"] = col_order_map
            return cleaned

        def order_key_frame(
            keys_df: pd.DataFrame,
            key_cols: list[str],
            order_map: dict[str, list[str]],
        ) -> pd.DataFrame:
            """Apply ordering rules to a key DataFrame, preserving stable order."""
            if not key_cols or not order_map or keys_df.empty:
                return keys_df

            ordered = keys_df.copy()
            order_cols = []
            for col_name in key_cols:
                order_list = order_map.get(col_name)
                if not order_list:
                    continue
                if not isinstance(order_list, (list, tuple, set)):
                    continue
                order_values = [str(v) for v in order_list]
                order_index = {
                    value: idx for idx, value in enumerate(order_values)
                }
                ordered_col = ordered[col_name].map(
                    lambda v: order_index.get(str(v), len(order_index))
                )
                order_col = f"_order_{col_name}"
                ordered[order_col] = ordered_col
                order_cols.append(order_col)

            if not order_cols:
                return keys_df

            ordered = ordered.sort_values(order_cols, kind="stable")
            return ordered.drop(columns=order_cols)
        
        st.markdown("<div id='pivot-dim-row-marker'></div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            idx = st.multiselect("行维度 (如 Visit)", all_final_cols, key="pivot_index")
        with c2:
            col = st.multiselect("列维度 (如 Group)", all_final_cols, key="pivot_columns")

        st.markdown("<div id='pivot-metric-row-marker'></div>", unsafe_allow_html=True)
        c3, c4 = st.columns(2)
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
                "统计量（可多选）",
                agg_options,
                default=default_aggs,
                key="pivot_aggs",
            )
            agg_axis_labels = {"按行": "row", "按列": "col"}
            current_axis = st.session_state.get("pivot_agg_axis", "row")
            default_label = "按列" if current_axis == "col" else "按行"
            axis_label_col, axis_radio_col = st.columns([1, 3])
            with axis_label_col:
                st.markdown("统计量布局")
            with axis_radio_col:
                selected_label = st.radio(
                    "统计量布局",
                    list(agg_axis_labels.keys()),
                    index=list(agg_axis_labels.keys()).index(default_label),
                    horizontal=True,
                    key="pivot_agg_axis_ui",
                    label_visibility="collapsed",
                )
            st.session_state["pivot_agg_axis"] = agg_axis_labels[selected_label]

        row_orders_map = st.session_state.get("pivot_row_orders", {})
        if not isinstance(row_orders_map, dict):
            row_orders_map = {}
        if idx:
            row_orders_map = {
                k: v for k, v in row_orders_map.items() if k in idx
            }
            st.session_state["pivot_row_orders"] = row_orders_map
        else:
            row_orders_map = {}
            st.session_state["pivot_row_orders"] = row_orders_map

        row_order_values_map: dict[str, list[str]] = {}
        if idx:
            for field in idx:
                if field in final_df.columns:
                    available_values = (
                        final_df[field]
                        .dropna()
                        .astype(str)
                        .drop_duplicates()
                        .tolist()
                    )
                else:
                    available_values = []
                row_order_values_map[field] = sync_pivot_row_order(
                    field, available_values
                )

        col_order_map = st.session_state.get("pivot_col_order", {})
        if not isinstance(col_order_map, dict):
            col_order_map = {}
        if col:
            col_order_map = {
                k: v for k, v in col_order_map.items() if k in col
            }
            st.session_state["pivot_col_order"] = col_order_map

        order_left, order_right = st.columns(2)
        with order_left:
            if not idx:
                st.caption("请选择行维度以排序。")
            else:
                for field in idx:
                    with st.expander(
                        f"行维度顺序（{field}）", expanded=False
                    ):
                        values = row_order_values_map.get(field, [])
                        if not values:
                            st.caption("暂无可排序的值。")
                            continue
                        selected_value = st.selectbox(
                            "选择要移动的值",
                            values,
                            key=f"pivot_row_order_selected_{field}",
                        )
                        move_up, move_down = st.columns(2)
                        if move_up.button(
                            "上移", key=f"pivot_row_order_up_{field}"
                        ):
                            new_order = list(values)
                            idx_pos = new_order.index(selected_value)
                            if idx_pos > 0:
                                new_order[idx_pos - 1], new_order[idx_pos] = (
                                    new_order[idx_pos],
                                    new_order[idx_pos - 1],
                                )
                                row_orders = st.session_state.get(
                                    "pivot_row_orders", {}
                                )
                                if not isinstance(row_orders, dict):
                                    row_orders = {}
                                row_orders[field] = new_order
                                st.session_state["pivot_row_orders"] = row_orders
                                row_order_values_map[field] = new_order
                                st.rerun()
                        if move_down.button(
                            "下移", key=f"pivot_row_order_down_{field}"
                        ):
                            new_order = list(values)
                            idx_pos = new_order.index(selected_value)
                            if idx_pos < len(new_order) - 1:
                                new_order[idx_pos + 1], new_order[idx_pos] = (
                                    new_order[idx_pos],
                                    new_order[idx_pos + 1],
                                )
                                row_orders = st.session_state.get(
                                    "pivot_row_orders", {}
                                )
                                if not isinstance(row_orders, dict):
                                    row_orders = {}
                                row_orders[field] = new_order
                                st.session_state["pivot_row_orders"] = row_orders
                                row_order_values_map[field] = new_order
                                st.rerun()
                        st.caption("当前顺序：" + " → ".join(values))

        with order_right:
            if not col:
                st.caption("请选择列维度以排序。")
            else:
                with st.expander("列维度顺序", expanded=False):
                    for col_idx, col_field in enumerate(col):
                        if col_field in final_df.columns:
                            col_values = (
                                final_df[col_field]
                                .dropna()
                                .astype(str)
                                .drop_duplicates()
                                .tolist()
                            )
                        else:
                            col_values = []
                        col_order_values = sync_pivot_col_order(
                            col_field, col_values
                        )
                        st.markdown(f"**{col_field}**")
                        if not col_order_values:
                            st.caption("暂无可排序的值。")
                            continue
                        col_key = col_field
                        selected_col_value = st.selectbox(
                            "选择要移动的值",
                            col_order_values,
                            key=f"pivot_col_order_selected_{col_key}",
                        )
                        move_up, move_down = st.columns(2)
                        if move_up.button(
                            "上移", key=f"pivot_col_order_up_{col_key}"
                        ):
                            new_order = list(col_order_values)
                            idx_pos = new_order.index(selected_col_value)
                            if idx_pos > 0:
                                new_order[idx_pos - 1], new_order[idx_pos] = (
                                    new_order[idx_pos],
                                    new_order[idx_pos - 1],
                                )
                                latest_map = st.session_state.get(
                                    "pivot_col_order", {}
                                )
                                if not isinstance(latest_map, dict):
                                    latest_map = {}
                                latest_map[col_field] = new_order
                                st.session_state["pivot_col_order"] = (
                                    latest_map
                                )
                                col_order_values = new_order
                                st.rerun()
                        if move_down.button(
                            "下移", key=f"pivot_col_order_down_{col_key}"
                        ):
                            new_order = list(col_order_values)
                            idx_pos = new_order.index(selected_col_value)
                            if idx_pos < len(new_order) - 1:
                                new_order[idx_pos + 1], new_order[idx_pos] = (
                                    new_order[idx_pos],
                                    new_order[idx_pos + 1],
                                )
                                latest_map = st.session_state.get(
                                    "pivot_col_order", {}
                                )
                                if not isinstance(latest_map, dict):
                                    latest_map = {}
                                latest_map[col_field] = new_order
                                st.session_state["pivot_col_order"] = (
                                    latest_map
                                )
                                col_order_values = new_order
                                st.rerun()
                        st.caption(
                            "当前顺序：" + " → ".join(col_order_values)
                        )

        if idx and col and val and aggs:
            # 1. 透视表
            try:
                nested_data = render_pivot_nested(
                    final_df,
                    index_cols=idx,
                    column_cols=col,
                    value_cols=val,
                    agg_names=aggs,
                    row_orders=row_orders_map,
                    col_orders=st.session_state.get("pivot_col_order", {}),
                    agg_axis=st.session_state.get("pivot_agg_axis", "row"),
                )
                st.download_button(
                    "📥 下载嵌套透视表（Excel）",
                    nested_pivot_to_excel_bytes(
                        nested_data,
                        agg_axis=st.session_state.get("pivot_agg_axis", "row"),
                    ),
                    "pivot_table_nested.xlsx",
                )
                if len(val) != 1:
                    st.info("折线图仅支持单一值字段。")
                elif not col:
                    st.info("折线图需要至少一个列维度。")
                else:
                    st.markdown("#### 📈 折线图")
                    value_col = val[0]
                    row_cols = idx
                    col_orders = st.session_state.get(
                        "pivot_col_order", {}
                    )
                    row_orders = row_orders_map
                    mean_y_ranges: dict[str, list[float]] = {}
                    line_aggs = [
                        "Mean - 平均值",
                        "Median - 中位数",
                    ]
                    line_aggs = [a for a in line_aggs if a in AGG_METHODS]
                    error_mode = None
                    keep_percent = 90
                    if "Mean - 平均值" in line_aggs:
                        control_cols = st.columns([3, 1])
                        with control_cols[0]:
                            error_mode = st.radio(
                                "均值误差条",
                                ["无", "SE", "SD"],
                                horizontal=True,
                                key="line_error_mode",
                                index=0,
                            )
                        with control_cols[1]:
                            keep_percent = st.number_input(
                                "Trimmed Mean保留区间(%)",
                                min_value=0,
                                max_value=100,
                                value=int(
                                    st.session_state.get(
                                        "line_trim_keep_percent", 90
                                    )
                                ),
                                step=1,
                                key="line_trim_keep_percent",
                            )
                    else:
                        keep_percent = st.number_input(
                            "Trimmed Mean保留区间(%)",
                            min_value=0,
                            max_value=100,
                            value=int(
                                st.session_state.get(
                                    "line_trim_keep_percent", 90
                                )
                            ),
                            step=1,
                            key="line_trim_keep_percent",
                        )
                    trim_pct = max(0.0, (100 - float(keep_percent)) / 2.0)
                    

                    items_by_col: dict[str, dict[str, dict[str, Any]]] = {
                        col_field: {} for col_field in col
                    }
                    export_by_col: dict[str, dict[str, dict[str, Any]]] = {
                        col_field: {} for col_field in col
                    }
                    for agg_name in line_aggs:
                        for col_field in col:
                            is_mean = agg_name == "Mean - 平均值"
                            resolved_error = None
                            if is_mean and error_mode and error_mode != "无":
                                resolved_error = error_mode
                            fig = build_pivot_line_fig(
                                df=final_df,
                                value_col=value_col,
                                row_key_cols=row_cols,
                                col_field=col_field,
                                agg_name=agg_name,
                                row_orders=row_orders,
                                col_orders=col_orders,
                                error_mode=resolved_error,
                                show_counts=is_mean,
                                y_range_pad_ratio=0.1,
                            )
                            if fig is None:
                                continue
                            if is_mean:
                                mean_range = fig.layout.yaxis.range
                                if mean_range:
                                    mean_y_ranges[col_field] = list(
                                        mean_range
                                    )
                            title = f"{col_field} | {agg_name}"
                            items_by_col[col_field][agg_name] = {
                                "title": title,
                                "fig": fig,
                            }
                            export_by_col[col_field][agg_name] = {
                                "title": title,
                                "title_html": html.escape(title),
                                "fig": copy.deepcopy(fig),
                                "legend_items": [],
                                "chart_type": "line",
                            }
                    keep_ratio = max(float(keep_percent) / 100.0, 0.0)
                    trimmed_agg_name = f"trimmed mean ({keep_percent}%)"
                    trimmed_func = lambda s, keep_ratio=keep_ratio: (
                        compute_trimmed_mean(s, keep_ratio)
                    )
                    for col_field in col:
                        fig = build_pivot_line_fig(
                            df=final_df,
                            value_col=value_col,
                            row_key_cols=row_cols,
                            col_field=col_field,
                            agg_name=trimmed_agg_name,
                            row_orders=row_orders,
                            col_orders=col_orders,
                            agg_func=trimmed_func,
                            y_range_pad_ratio=0.1,
                        )
                        if fig is None:
                            continue
                        mean_range = mean_y_ranges.get(col_field)
                        if mean_range:
                            fig.update_yaxes(range=list(mean_range))
                        title = f"{col_field} | trimmed mean ({keep_percent}%)"
                        items_by_col[col_field]["__trimmed__"] = {
                            "title": title,
                            "fig": fig,
                        }
                        export_by_col[col_field]["__trimmed__"] = {
                            "title": title,
                            "title_html": html.escape(title),
                            "fig": copy.deepcopy(fig),
                            "legend_items": [],
                            "chart_type": "line",
                        }
                    ordered_keys = [
                        "Mean - 平均值",
                        "__trimmed__",
                        "Median - 中位数",
                    ]
                    ordered_items = []
                    ordered_export_items = []
                    for col_field in col:
                        for key in ordered_keys:
                            item = items_by_col.get(col_field, {}).get(key)
                            if item:
                                ordered_items.append(item)
                            export_item = export_by_col.get(
                                col_field, {}
                            ).get(key)
                            if export_item:
                                ordered_export_items.append(export_item)
                    if not ordered_items:
                        st.info("暂无可绘制的折线图数据。")
                    else:
                        max_cols = 3
                        for start in range(0, len(ordered_items), max_cols):
                            row_items = ordered_items[
                                start : start + max_cols
                            ]
                            cols = st.columns(max_cols)
                            for col_idx in range(max_cols):
                                if col_idx >= len(row_items):
                                    continue
                                item = row_items[col_idx]
                                with cols[col_idx]:
                                    st.markdown(
                                        f"**{item['title']}**"
                                    )
                                    render_line_fig(
                                        item["fig"],
                                        key=(
                                            "pivot_line_"
                                            f"{start + col_idx}"
                                        ),
                                    )

                    line_export_items = ordered_export_items
                    if line_export_items:
                        if st.button(
                            "📥 下载折线图 (HTML)",
                            key="btn_export_line_charts",
                        ):
                            full_html = build_charts_export_html(
                                line_export_items
                            )
                            st.download_button(
                                "⬇️ 保存折线图 HTML",
                                data=full_html.encode("utf-8"),
                                file_name="pivot_line_charts.html",
                                mime="text/html",
                                key="btn_export_line_charts_download",
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

                    row_orders_for_chart = (
                        row_orders_map if isinstance(row_orders_map, dict) else {}
                    )
                    col_orders_for_chart = st.session_state.get(
                        "pivot_col_order", {}
                    )
                    if not isinstance(col_orders_for_chart, dict):
                        col_orders_for_chart = {}

                    if row_key_cols:
                        row_keys_df = (
                            final_df[row_key_cols]
                            .dropna()
                            .astype(str)
                            .drop_duplicates()
                        )
                        row_keys_df = order_key_frame(
                            row_keys_df, row_key_cols, row_orders_for_chart
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
                        col_keys_df = order_key_frame(
                            col_keys_df, col_key_cols, col_orders_for_chart
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
                            ["统一坐标", "差值统一坐标", "对数坐标", "箱线图"],
                            horizontal=True,
                            key="chart_type_mode",
                        )

                        use_uniform_chart = chart_type == "统一坐标"
                        use_uniform_min_max_chart = chart_type == "差值统一坐标"
                        use_uniform_log_chart = chart_type == "对数坐标"
                        use_boxplot_chart = chart_type == "箱线图"
                        uniform_x_range = None
                        uniform_y_max = None
                        uniform_min_max_x_range = None
                        uniform_min_max_y_max = None
                        uniform_log_x_range = None
                        uniform_log_y_max = None
                        boxplot_y_range = None
                        if (
                            use_uniform_chart
                            or use_uniform_min_max_chart
                            or use_uniform_log_chart
                        ):
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
                        if use_uniform_chart:
                            uniform_x_range, uniform_y_max = compute_uniform_axes(
                                final_df, row_key_cols, col_key_cols, value_col
                            )
                            if uniform_y_max <= 0:
                                uniform_x_range = None
                                uniform_y_max = None
                        if use_uniform_min_max_chart:
                            (
                                uniform_min_max_x_range,
                                uniform_min_max_y_max,
                            ) = compute_uniform_min_max_axes(
                                final_df, row_key_cols, col_key_cols, value_col
                            )
                            if uniform_min_max_y_max <= 0:
                                uniform_min_max_x_range = None
                                uniform_min_max_y_max = None
                        if use_uniform_log_chart:
                            (
                                uniform_log_x_range,
                                uniform_log_y_max,
                            ) = compute_uniform_log_axes(
                                final_df, row_key_cols, col_key_cols, value_col
                            )
                            if uniform_log_y_max <= 0:
                                uniform_log_x_range = None
                                uniform_log_y_max = None
                            raw_vals = final_df[value_col]
                            numeric_vals = pd.to_numeric(
                                raw_vals, errors="coerce"
                            )
                            invalid_mask = numeric_vals.isna() | (numeric_vals <= 0)
                            invalid_vals = raw_vals[invalid_mask]
                            if not invalid_vals.empty:
                                invalid_count = int(invalid_mask.sum())
                                unique_vals = [
                                    str(v)
                                    for v in invalid_vals.dropna().unique().tolist()
                                ]
                                max_display = 8
                                display_vals = unique_vals[:max_display]
                                more = len(unique_vals) - len(display_vals)
                                display_text = ", ".join(display_vals)
                                if not display_text:
                                    display_text = "空值"
                                if more > 0:
                                    display_text += f" 等 {more} 个不同值"
                                st.warning(
                                    f"对数坐标已过滤 {invalid_count} 条非正值/非数值："
                                    f"{display_text}"
                                )
                        if use_boxplot_chart:
                            boxplot_y_range = compute_boxplot_range(
                                final_df, value_col
                            )

                        control_group = None
                        if (
                            use_uniform_chart
                            or use_uniform_min_max_chart
                            or use_uniform_log_chart
                        ):
                            control_group = resolve_uniform_control_group(
                                col_key_cols,
                                col_keys,
                                st.session_state.get("uniform_control_group"),
                                key="uniform_control_group",
                            )

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

                    def build_row_key_sig(row_key: dict) -> str:
                        """Build a stable signature string for a row key."""
                        if not row_key_cols:
                            return "(All)"
                        return "\x1f".join(
                            [str(row_key.get(c, "")) for c in row_key_cols]
                        )

                    control_stats_by_row = {}
                    if (
                        use_uniform_chart
                        or use_uniform_min_max_chart
                        or use_uniform_log_chart
                    ) and control_group:
                        for rk in row_keys:
                            ctrl_df = final_df
                            for col_name, v in rk.items():
                                ctrl_df = ctrl_df[
                                    ctrl_df[col_name].astype(str) == v
                                ]
                            for col_name, v in control_group.items():
                                if col_name in ctrl_df.columns:
                                    ctrl_df = ctrl_df[
                                        ctrl_df[col_name].astype(str) == str(v)
                                    ]
                            vals = pd.to_numeric(
                                ctrl_df[value_col], errors="coerce"
                            ).dropna()
                            if vals.empty:
                                continue
                            if use_uniform_log_chart:
                                vals = vals[vals > 0]
                                if vals.empty:
                                    continue
                                log_vals = vals.apply(math.log)
                                control_stats_by_row[build_row_key_sig(rk)] = (
                                    float(math.exp(log_vals.mean())),
                                    float(math.exp(log_vals.median())),
                                )
                            else:
                                control_stats_by_row[build_row_key_sig(rk)] = (
                                    float(vals.mean()),
                                    float(vals.median()),
                                )

                    if use_boxplot_chart:
                        col_group_labels = []
                        for ck in col_keys:
                            if col_key_cols:
                                label = " / ".join(
                                    [
                                        html.escape(str(ck.get(c, "")))
                                        for c in col_key_cols
                                    ]
                                )
                            else:
                                label = "All"
                            col_group_labels.append(label)

                        visible_labels = col_group_labels
                        if len(col_group_labels) > 1:
                            visible_labels = st.multiselect(
                                "显示列组",
                                options=col_group_labels,
                                default=col_group_labels,
                                key="boxplot_visible_cols",
                            )
                            if not visible_labels:
                                st.info("请至少选择一个列组以显示箱线图。")
                                visible_labels = []

                        filtered_col_keys = [
                            ck
                            for ck, label in zip(col_keys, col_group_labels)
                            if label in visible_labels
                        ]

                        combo_keys = []
                        for rk in row_keys:
                            for ck in filtered_col_keys:
                                combo = {}
                                combo.update(rk)
                                combo.update(ck)
                                combo_keys.append(combo)
                        if limit and combo_keys:
                            combo_keys = combo_keys[:limit]

                        fig = build_boxplot_matrix_fig(
                            df=final_df,
                            subj_col=subj_col,
                            value_col=value_col,
                            row_key_cols=row_key_cols,
                            col_key_cols=col_key_cols,
                            row_keys=row_keys,
                            col_keys=filtered_col_keys,
                            combo_keys=combo_keys,
                            y_range=boxplot_y_range,
                            color_labels=col_group_labels,
                        )
                        if fig is not None:
                            fig_for_export = copy.deepcopy(fig)
                            render_boxplot_fig(fig, key="c_boxplot_all")
                            all_figs.append(
                                {
                                    "title": "",
                                    "title_html": "",
                                    "fig": fig_for_export,
                                    "legend_items": [],
                                    "chart_type": "boxplot",
                                }
                            )
                            count = 1
                    else:
                        max_cols_per_row = 3

                        def render_cell_chart(
                            row_key: dict,
                            col_key: dict,
                            row_idx: int,
                            col_idx: int,
                            chart_color: str,
                        ) -> None:
                            """Render a single cell chart within the pivot grid."""
                            nonlocal count

                            cell = final_df
                            for col_name, v in row_key.items():
                                cell = cell[cell[col_name].astype(str) == v]
                            for col_name, v in col_key.items():
                                cell = cell[cell[col_name].astype(str) == v]

                            if cell.empty:
                                return

                            title_parts = [
                                f"{k}={row_key[k]}"
                                for k in row_key_cols
                                if k in row_key
                            ] + [
                                f"{k}={col_key[k]}"
                                for k in col_key_cols
                                if k in col_key
                            ]
                            title = (
                                "<br>".join(title_parts)
                                if title_parts
                                else "(All)"
                            )
                            title_html = "<br>".join(
                                [html.escape(p) for p in title_parts]
                            ) if title_parts else "(All)"
                            internal_title = ""
                            key_suffix = f"r{row_idx}_c{col_idx}"

                            control_mean = None
                            control_median = None
                            if control_group:
                                stats = control_stats_by_row.get(
                                    build_row_key_sig(row_key)
                                )
                                if stats:
                                    control_mean, control_median = stats
                            if use_uniform_log_chart:
                                fig = build_uniform_log_spaghetti_fig(
                                    df=cell,
                                    subj_col=subj_col,
                                    value_col=value_col,
                                    title=internal_title,
                                    x_range=uniform_log_x_range,
                                    y_max_count=uniform_log_y_max,
                                    control_mean=control_mean,
                                    control_median=control_median,
                                    marker_color=chart_color,
                                )
                                render_chart = render_uniform_log_spaghetti_fig
                                chart_type_key = "uniform_log"
                            elif use_uniform_min_max_chart:
                                fig = build_uniform_min_max_spaghetti_fig(
                                    df=cell,
                                    subj_col=subj_col,
                                    value_col=value_col,
                                    title=internal_title,
                                    x_range=uniform_min_max_x_range,
                                    y_max_count=uniform_min_max_y_max,
                                    control_mean=control_mean,
                                    control_median=control_median,
                                    marker_color=chart_color,
                                )
                                render_chart = render_uniform_min_max_spaghetti_fig
                                chart_type_key = "uniform_min_max"
                            else:
                                fig = build_uniform_spaghetti_fig(
                                    df=cell,
                                    subj_col=subj_col,
                                    value_col=value_col,
                                    title=internal_title,
                                    x_range=uniform_x_range,
                                    y_max_count=uniform_y_max,
                                    control_mean=control_mean,
                                    control_median=control_median,
                                    marker_color=chart_color,
                                )
                                render_chart = render_uniform_spaghetti_fig
                                chart_type_key = "uniform"
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

                            render_chart(fig, key=f"c_{key_suffix}")
                            if legend_items:
                                legend_lines = []
                                for item in legend_items:
                                    dash_style = (
                                        "dashed"
                                        if item.get("dash") == "dash"
                                        else "solid"
                                    )
                                    line_color = item.get("color", "#c00")
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
                                        f"gap:8px;font-size:12px;color:{line_color};"
                                        "line-height:1.2;margin-top:2px;'>"
                                        f"<span style='display:inline-block;"
                                        f"width:32px;border-top:3px {dash_style} {line_color};'></span>"
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

                            all_figs.append(
                                {
                                    "title": title,
                                    "title_html": title_html,
                                    "fig": fig_for_export,
                                    "legend_items": legend_items,
                                    "chart_type": chart_type_key,
                                }
                            )
                            count += 1

                        stop_render = False
                        for i, rk in enumerate(row_keys):
                            if stop_render:
                                break
                            group_color = color_palette[i % len(color_palette)]

                            for chunk_start in range(
                                0, len(col_keys), max_cols_per_row
                            ):
                                if stop_render:
                                    break
                                chunk = col_keys[
                                    chunk_start : chunk_start
                                    + max_cols_per_row
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
                            """Build a link to the subject profile page."""
                            return build_page_url(
                                "subject_profile",
                                {"subject_id": str(subject_id)},
                            )
                        
                        st.link_button(
                            "🔍 在新标签页打开受试者档案",
                            build_subject_profile_url(selected_id),
                        )

if __name__ == "__main__":
    main()
