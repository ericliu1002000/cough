"""Streamlit data builder page."""

import os

import pandas as pd
import streamlit as st

from analysis.auth.session import require_login
from analysis.settings.config import get_engine
from analysis.settings.constants import OPERATORS, SUBJECT_ID_ALIASES
from analysis.repositories.metadata_repo import get_id_column, load_table_metadata
from analysis.repositories.setup_repo import (
    delete_setup_config,
    fetch_all_setups,
    fetch_setup_config,
    save_extraction_config,
)
from analysis.repositories.sql_builder import (
    build_sql,
    get_unique_values,
)
from analysis.state.data_builder import add_filter_row, init_filter_rows, remove_filter_row

# 从环境变量读取可选的最大表数量，默认为 5
MAX_TABLE_NUMBER = int(os.getenv("MAX_TABLE_NUMBER", "5"))

# ===========================
# 2. 界面布局 (Streamlit)
# ===========================

st.set_page_config(page_title="临床数据拼表器", layout="wide")
require_login()
st.title("🏥 临床试验数据拼表工具")

meta_data = load_table_metadata()
all_tables = list(meta_data.keys())

# --- Session State 初始化 ---
# filter_rows: 存储筛选条件的列表，每项是一个 dict
init_filter_rows()

# --- 侧边栏 ---
with st.sidebar:
    # 配置管理区
    st.header("🧩 分析集配置")

    setups = fetch_all_setups()
    setup_options = ["<新配置>"]
    setup_name_to_desc = {}
    for row in setups:
        name = row["setup_name"]
        desc = row.get("description") or ""
        label = f"{name} - {desc}" if desc else name
        setup_options.append(label)
        setup_name_to_desc[label] = name

    selected_setup_label = st.selectbox(
        "选择已有配置",
        options=setup_options,
        index=0,
    )

    # 加载配置按钮
    if selected_setup_label != "<新配置>":
        selected_setup_name = setup_name_to_desc[selected_setup_label]

        if st.button("✏️ 加载配置", key="btn_load_setup"):
            cfg_all = fetch_setup_config(selected_setup_name)
            if cfg_all is not None:
                extraction_cfg = cfg_all.get("extraction") or {}

                # 恢复选表
                if "selected_tables" in extraction_cfg:
                    st.session_state["selected_tables"] = extraction_cfg[
                        "selected_tables"
                    ]
                # 恢复每张表的列选择
                if "table_columns_map" in extraction_cfg:
                    for tbl, cols in extraction_cfg["table_columns_map"].items():
                        st.session_state[f"sel_col_{tbl}"] = cols
                # 恢复筛选条件
                conditions = extraction_cfg.get("filters", {}).get("conditions", [])
                st.session_state.filter_rows = [
                    {"id": i} for i in range(len(conditions))
                ]
                for i, cond in enumerate(conditions):
                    st.session_state[f"f_tbl_{i}"] = cond.get("table")
                    st.session_state[f"f_col_{i}"] = cond.get("col")
                    st.session_state[f"f_op_{i}"] = cond.get("op")
                    st.session_state[f"f_val_{i}"] = cond.get("val")
                # 恢复黑名单
                if "subject_blocklist" in extraction_cfg:
                    st.session_state["subject_blocklist"] = extraction_cfg[
                        "subject_blocklist"
                    ]

                # 恢复 Group By / 聚合配置（如果有）
                if "group_by" in extraction_cfg:
                    gb_list = extraction_cfg.get("group_by") or []
                    st.session_state["use_group_by"] = bool(gb_list)
                    st.session_state["gb_count"] = len(gb_list)
                    for i, gb in enumerate(gb_list):
                        st.session_state[f"gb_tbl_{i}"] = gb.get("table")
                        st.session_state[f"gb_col_{i}"] = gb.get("col")
                        if gb.get("alias") is not None:
                            st.session_state[f"gb_alias_{i}"] = gb.get("alias")

                if "aggregations" in extraction_cfg:
                    agg_list = extraction_cfg.get("aggregations") or []
                    st.session_state["agg_count"] = len(agg_list)
                    if agg_list:
                        st.session_state["use_group_by"] = True
                    for i, agg in enumerate(agg_list):
                        st.session_state[f"agg_tbl_{i}"] = agg.get("table")
                        st.session_state[f"agg_col_{i}"] = agg.get("col")
                        if agg.get("func") is not None:
                            # 简单反解 COUNT(DISTINCT ...) 为 COUNT_DISTINCT
                            func = agg.get("func")
                            if func.startswith("COUNT(DISTINCT"):
                                st.session_state[f"agg_func_{i}"] = "COUNT_DISTINCT"
                            else:
                                st.session_state[f"agg_func_{i}"] = func
                        if agg.get("alias") is not None:
                            st.session_state[f"agg_alias_{i}"] = agg.get("alias")

                st.success(f"已加载配置：{selected_setup_name}")
                st.rerun()

        # 删除配置按钮
        if st.button("🗑️ 删除配置", key="btn_delete_setup"):
            delete_setup_config(selected_setup_name)
            st.success(f"已删除配置：{selected_setup_name}")
            st.rerun()

    st.markdown("---")

    st.header("⚙️ 全局配置")
    st.info(f"🔗 智能 Join 逻辑已启用。\nKey: {', '.join(SUBJECT_ID_ALIASES)}")
    
    st.subheader("🚫 受试者黑名单 (Not In)")
    subject_blocklist = st.text_area(
        "输入要排除的 ID (一行一个):",
        height=100,
        key="subject_blocklist",
    )

# --- 主界面 ---
st.subheader("1. 选择要拼接的表 (按 Join 顺序)")
selected_tables = st.multiselect(
    f"请选择表 (最多 {MAX_TABLE_NUMBER} 张):",
    options=all_tables,
    default=None,
    key="selected_tables",
    help="第一个选中的表将作为主表 (Left Table)"
)

if not selected_tables:
    st.info("👈 请先选择至少一张表。")
    st.stop()

# 限制最大选表数
if len(selected_tables) > MAX_TABLE_NUMBER:
    st.error(f"❌ 最多只能选择 {MAX_TABLE_NUMBER} 张表，当前已选 {len(selected_tables)} 张。请删除部分表。")
    st.stop()

if len(selected_tables) == MAX_TABLE_NUMBER:
    st.warning(f"⚠️ 已达到最大选表数量限制 ({MAX_TABLE_NUMBER})。")

# 显示列选择器
table_columns_map = {} 
with st.expander("2. 选择展示列 (点击展开)", expanded=True):
    cols_ui = st.columns(3)
    for idx, table_name in enumerate(selected_tables):
        # 智能提示该表的 Key
        this_id = get_id_column(table_name, meta_data)
        key_hint = f"🔑 {this_id}" if this_id else "❓ 无ID"
        
        with cols_ui[idx % 3]:
            available_cols = meta_data.get(table_name, [])
            st.markdown(f"**{table_name}** <small style='color:gray'>({key_hint})</small>", unsafe_allow_html=True)
            col_key = f"sel_col_{table_name}"
            # 如果 Session State 中已有值（例如从已保存配置加载），则不再传 default，
            # 避免出现“同时设置默认值和 Session State”的警告。
            if col_key in st.session_state:
                selected_cols = st.multiselect(
                    f"选择 {table_name} 的字段",
                    options=available_cols,
                    key=col_key,
                    label_visibility="collapsed",
                )
            else:
                selected_cols = st.multiselect(
                    f"选择 {table_name} 的字段",
                    options=available_cols,
                    default=available_cols[:5] if available_cols else [],
                    key=col_key,
                    label_visibility="collapsed",
                )
            table_columns_map[table_name] = selected_cols

st.divider()

# ===========================
# 3. 可视化 WHERE 构建器
# ===========================
st.subheader("3. 筛选条件 (Where Builder)")
st.caption("构建 SQL WHERE 子句，条件之间通过 AND 连接。")

if st.button("➕ 添加筛选条件"):
    add_filter_row()

final_conditions = []

# 渲染筛选行
if st.session_state.filter_rows:
    for i, row in enumerate(st.session_state.filter_rows):
        with st.container():
            c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 3, 1])
            
            # 1. 表选择
            with c1:
                t_sel = st.selectbox("表", options=selected_tables, key=f"f_tbl_{i}", label_visibility="collapsed")
            
            # 2. 列选择 (基于表)
            with c2:
                cols = meta_data.get(t_sel, [])
                c_sel = st.selectbox("列", options=cols, key=f"f_col_{i}", label_visibility="collapsed")
            
            # 3. 操作符
            with c3:
                op_sel = st.selectbox("条件", options=list(OPERATORS.keys()), format_func=lambda x: OPERATORS[x], key=f"f_op_{i}", label_visibility="collapsed")
            
            # 4. 值输入 (根据操作符变化)
            with c4:
                val_key = f"f_val_{i}"
                
                # 特殊逻辑：如果是 IN / NOT IN，显示多选框，并尝试加载数据
                if op_sel in ["IN", "NOT IN"]:
                    # 使用 session_state 保存每一行已加载的候选值，保证多次交互后仍能回显
                    loaded_vals_key = f"loaded_vals_{i}"
                    loaded_vals = st.session_state.get(loaded_vals_key, [])

                    # 加载值的功能放在一个小的 expander 里以免占据太多空间
                    with st.expander("🔍 加载值", expanded=False):
                        if st.button("从数据库加载 Top 100", key=f"btn_load_{i}"):
                            loaded_vals = get_unique_values(t_sel, c_sel)
                            # 将加载结果持久化到 session_state，避免下次交互丢失
                            st.session_state[loaded_vals_key] = loaded_vals
                            # 简要提示加载结果
                            if loaded_vals:
                                st.success(f"已加载 {len(loaded_vals)} 个值，请在下方选择或输入。")

                    # 为了保证“已选择的值”在 options 中始终可见，
                    # 将当前选中的值与已加载的候选值合并去重后作为 options
                    current_selected = st.session_state.get(val_key, [])
                    # 转成字符串，保持与 loaded_vals 类型一致
                    current_selected = [str(v) for v in current_selected]
                    merged_options = sorted(set(current_selected) | set(loaded_vals))

                    val_input = st.multiselect(
                        "值", 
                        options=merged_options, 
                        key=val_key,
                        label_visibility="collapsed",
                        placeholder="输入值并回车，或选择..."
                    )
                
                elif op_sel in ["IS NULL", "IS NOT NULL"]:
                    val_input = None
                    st.write("---")
                
                else:
                    # 单值输入
                    val_input = st.text_input("值", key=val_key, label_visibility="collapsed")

            # 5. 删除
            with c5:
                if st.button("🗑️", key=f"btn_del_{i}"):
                    remove_filter_row(i)
                    st.rerun()

            # 收集有效条件
            if t_sel and c_sel and op_sel:
                if (op_sel in ["IS NULL", "IS NOT NULL"]) or val_input:
                    final_conditions.append({
                        "table": t_sel,
                        "col": c_sel,
                        "op": op_sel,
                        "val": val_input
                    })
else:
    st.info("暂无筛选条件。点击上方按钮添加。")

filters_config = {"conditions": final_conditions}

# ===========================
# 3.x Group By & 聚合配置
# ===========================
st.subheader("3.x 分组与聚合 (可选)")
use_group_by = st.checkbox("启用 Group By 聚合模式", value=False, key="use_group_by")

group_by_config = []
aggregations_config = []

if use_group_by:
    st.caption("在启用 Group By 后：SELECT 中的非分组字段必须通过聚合函数给出。")

    st.markdown("**分组字段 (GROUP BY)**")
    gb_rows = st.number_input("分组字段个数", min_value=0, max_value=10, value=0, step=1, key="gb_count")
    for i in range(int(gb_rows)):
        c1, c2, c3 = st.columns([2, 2, 2])
        with c1:
            tbl = st.selectbox(
                f"分组表 {i+1}",
                options=selected_tables,
                key=f"gb_tbl_{i}",
            )
        with c2:
            cols = meta_data.get(tbl, [])
            col = st.selectbox(
                f"分组列 {i+1}",
                options=cols,
                key=f"gb_col_{i}",
            )
        with c3:
            alias = st.text_input(
                "别名 (可选)",
                key=f"gb_alias_{i}",
                placeholder=f"{tbl}_{col}" if tbl and col else "",
            )
        if tbl and col:
            group_by_config.append({"table": tbl, "col": col, "alias": alias})

    st.markdown("**聚合字段 (Aggregations)**")
    agg_rows = st.number_input("聚合字段个数", min_value=0, max_value=20, value=0, step=1, key="agg_count")
    agg_func_options = ["COUNT", "COUNT_DISTINCT", "SUM", "AVG", "MIN", "MAX"]
    for i in range(int(agg_rows)):
        c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
        with c1:
            tbl = st.selectbox(
                f"聚合表 {i+1}",
                options=selected_tables,
                key=f"agg_tbl_{i}",
            )
        with c2:
            cols = meta_data.get(tbl, [])
            col = st.selectbox(
                f"聚合列 {i+1}",
                options=cols,
                key=f"agg_col_{i}",
            )
        with c3:
            func_raw = st.selectbox(
                "函数",
                options=agg_func_options,
                key=f"agg_func_{i}",
            )
        with c4:
            alias = st.text_input(
                "别名 (可选)",
                key=f"agg_alias_{i}",
                placeholder=f"{func_raw}_{tbl}_{col}" if tbl and col else "",
            )

        if tbl and col and func_raw:
            func_sql = "COUNT(DISTINCT" if func_raw == "COUNT_DISTINCT" else func_raw
            aggregations_config.append(
                {
                    "table": tbl,
                    "col": col,
                    "func": func_sql,
                    "alias": alias,
                }
            )

# --- 生成 ---
st.divider()

if st.button("🚀 生成 SQL 并预览数据", type="primary"):
    sql = build_sql(
        selected_tables,
        table_columns_map,
        filters_config,
        subject_blocklist,
        meta_data,
        group_by=group_by_config if use_group_by else None,
        aggregations=aggregations_config if use_group_by else None,
    )
    
    if sql:
        st.subheader("生成的 SQL:")
        st.code(sql, language="sql")
        
        try:
            with st.spinner("正在查询..."):
                engine = get_engine()
                # 加上 execution_options(timeout=30) 防止卡死
                with engine.connect().execution_options(timeout=60) as conn:
                    df_result = pd.read_sql(sql, conn)
            
            st.success(f"查询成功！预览前 {len(df_result)} 行 (已限制 Limit 1000)。")
            st.dataframe(df_result, width="stretch")
            
            # 只有当有数据时才显示下载
            if not df_result.empty:
                csv = df_result.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 下载结果为 CSV",
                    data=csv,
                    file_name="cohort_data.csv",
                    mime="text/csv",
                )
            
        except Exception as e:
            st.error(f"SQL 执行错误: {e}")
            st.warning("提示: 如果查询超时，请尝试减少选择的表数量或增加筛选条件。")
    else:
        st.error("无法生成 SQL，请检查配置。")

# ===========================
# 4. 保存分析集配置
# ===========================
st.divider()
st.subheader("4. 保存当前分析集配置")

with st.form("save_setup_form"):
    setup_name_input = st.text_input("配置名称 (setup_name)*", key="setup_name_input")
    description_input = st.text_input("备注说明 (可选)", key="description_input")
    submitted = st.form_submit_button("💾 保存 / 更新配置")

if submitted:
    name = (setup_name_input or "").strip()
    if not name:
        st.error("配置名称不能为空。")
    else:
        # 组装当前配置
        extraction_config = {
            "selected_tables": selected_tables,
            "table_columns_map": table_columns_map,
            "filters": filters_config,
            "subject_blocklist": subject_blocklist,
            "group_by": group_by_config if use_group_by else [],
            "aggregations": aggregations_config if use_group_by else [],
            "max_table_number": MAX_TABLE_NUMBER,
        }
        save_extraction_config(name, description_input or None, extraction_config)
        st.success(f"配置 `{name}` 已保存 / 更新。")
