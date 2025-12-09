import json
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st
# [移除] 不再需要这个第三方库，避免布局 bug
# from streamlit_plotly_events import plotly_events

from settings import get_engine
from utils import (
    fetch_all_setups,
    fetch_setup_config,
    load_table_metadata,
    build_sql,
)

# 设置页面基本信息
st.set_page_config(page_title="分析仪表盘", layout="wide")
st.title("📊 分析仪表盘")


def run_analysis(config: Dict[str, Any]) -> tuple[str, pd.DataFrame]:
    """
    根据配置运行一次查询，返回生成的 SQL 和结果 DataFrame。
    """
    meta_data = load_table_metadata()

    selected_tables = config.get("selected_tables", [])
    table_columns_map = config.get("table_columns_map", {})
    filters = config.get("filters", {})
    subject_blocklist = config.get("subject_blocklist", "")

    # 调用 utils 中的核心逻辑生成 SQL
    sql = build_sql(
        selected_tables=selected_tables,
        table_columns_map=table_columns_map,
        filters=filters,
        subject_blocklist=subject_blocklist,
        meta_data=meta_data,
    )

    if not sql:
        st.error("无法根据当前配置生成 SQL。请检查配置内容。")
        return "", pd.DataFrame()

    engine = get_engine()
    # 使用 spinner 提示用户正在查询
    with st.spinner("正在执行数据库查询..."):
        # 建议加上超时限制防止卡死，这里设为 60 秒
        with engine.connect().execution_options(timeout=60) as conn:
            df = pd.read_sql(sql, conn)
            
    return sql, df


def apply_calculations(df: pd.DataFrame, rules: List[Dict]) -> pd.DataFrame:
    """
    核心逻辑：按顺序应用计算规则（二段配置）。
    """
    # 创建副本，以免修改 session_state 中的原始数据
    df_calc = df.copy()
    
    for rule in rules:
        try:
            name = rule['name']
            cols = rule['cols']
            method = rule['method']
            
            # 1. 过滤掉不存在的列，防止报错
            valid_cols = [c for c in cols if c in df_calc.columns]
            
            if not valid_cols:
                continue

            # 2. 强制将参与计算的列转换为数字类型
            for col in valid_cols:
                if not pd.api.types.is_numeric_dtype(df_calc[col]):
                    df_calc[col] = pd.to_numeric(df_calc[col], errors='coerce')

            # 3. 根据选择的方法进行行级运算 (axis=1)
            if method == '求和 (Sum)':
                df_calc[name] = df_calc[valid_cols].sum(axis=1, min_count=1)
            elif method == '平均值 (Mean)':
                df_calc[name] = df_calc[valid_cols].mean(axis=1)
            elif method == '最大值 (Max)':
                df_calc[name] = df_calc[valid_cols].max(axis=1)
            elif method == '最小值 (Min)':
                df_calc[name] = df_calc[valid_cols].min(axis=1)
                
        except Exception as e:
            st.error(f"⚠️ 计算规则 `{rule['name']}` 执行失败: {e}")
            
    return df_calc


def draw_spaghetti_chart(
    df: pd.DataFrame,
    subj_col: str,
    value_col: str,
    title: str,
    key: str,
) -> None:
    """
    绘制单个“透视单元格”的散点图（原 spaghetti chart）：
    - 纵轴: 受试者 ID (subj_col)
    - 横轴: 数值字段 (value_col)
    - 统计量: 均值竖线标注。

    使用 Streamlit 原生 on_select 事件处理点击交互。
    """
    if df.empty:
        st.info("该组合下无数据。")
        return

    if subj_col not in df.columns or value_col not in df.columns:
        st.info("受试者 ID 或数值列在当前数据集中不存在。")
        return

    # 数据清洗
    tmp = df[[subj_col, value_col]].copy()
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.dropna(subset=[value_col])
    
    if tmp.empty:
        st.info("该组合下无有效数值数据。")
        return
    
    # 绘图：个体点
    fig = px.scatter(
        tmp,
        x=value_col,
        y=subj_col,
        opacity=0.6,
        hover_data={subj_col: True, value_col: True},
        # 【关键】将受试者ID放入 custom_data，以便在点击事件中精确获取
        custom_data=[subj_col],
    )

    # 显示数值标签
    fig.update_traces(text=tmp[value_col].round(2), textposition="middle right")

    # 统计量计算：添加均值线
    series = tmp[value_col]
    agg_value = series.mean()
    try:
        agg_x = float(agg_value)
        fig.add_vline(
            x=agg_x,
            line_width=3,
            line_dash="dash",
            line_color="red",
            annotation_text=f"mean: {agg_x:.2f}",
            annotation_position="top",
        )
    except Exception:
        pass

    fig.update_layout(
        title=title,
        xaxis_title=value_col,
        yaxis_title=subj_col,
        height=400, # 设定一个合理的高度
        margin=dict(l=20, r=20, t=40, b=20),
    )

    # =========================================================
    # 使用 Streamlit 原生交互：点击点后写入 session_state
    # =========================================================
    st.plotly_chart(
        fig,
        width="stretch",
        on_select="rerun",       # 点击/选择后触发一次 rerun
        selection_mode="points", # 支持点级选择
        key=key,
    )

    # rerun 后，从 session_state[key] 中读取选中点信息
    chart_state = st.session_state.get(key)
    if chart_state is not None:
        # 兼容对象属性或 dict 两种形式
        selection = getattr(chart_state, "selection", None)
        if selection is None and isinstance(chart_state, dict):
            selection = chart_state.get("selection")

        if selection and "points" in selection and selection["points"]:
            clicked_point = selection["points"][0]
            custom_data = clicked_point.get("customdata")
            if custom_data:
                selected_id = custom_data[0]
            else:
                # 降级方案：若 customdata 不存在，则取 y 值（本图中 y 轴即 ID）
                selected_id = clicked_point.get("y")

            if selected_id is not None:
                st.session_state["selected_subject_id"] = selected_id


def main() -> None:
    # ===========================
    # 1. 侧边栏：加载配置
    # ===========================
    with st.sidebar:
        st.header("🧩 选择分析集")
        setups = fetch_all_setups()

        if not setups:
            st.info("暂无配置。请先在主页配置并保存数据集。")
            return

        # 创建下拉菜单选项
        option_labels = [f"{row['setup_name']}" for row in setups]
        selected_label = st.selectbox("选择配置", options=option_labels)
        
        # 找到对应的 setup 对象
        selected_row = next(row for row in setups if row['setup_name'] == selected_label)
        
        if selected_row.get("description"):
            st.info(f"📝 **备注**: {selected_row['description']}")

    # ===========================
    # 2. 主区域：加载数据
    # ===========================
    # 只有点击按钮时才去数据库查询，避免每次刷新都查
    if st.button("🚀 加载源数据", type="primary"):
        # 获取完整的配置（含一段/二段）
        cfg_all = fetch_setup_config(selected_row["setup_name"])
        if cfg_all:
            extraction_cfg = cfg_all.get("extraction") or {}
            calculation_cfg = cfg_all.get("calculation") or []

            sql, df_result = run_analysis(extraction_cfg)
            if not df_result.empty:
                # 将原始数据存入 Session State
                st.session_state["raw_df"] = df_result
                st.session_state["current_sql"] = sql
                # 恢复二段配置（计算规则）
                st.session_state["calc_rules"] = calculation_cfg or []
                
                # 初始化计算规则列表（如果还没有的话）
                if "calc_rules" not in st.session_state:
                    st.session_state["calc_rules"] = [] 
                
                st.success(f"数据加载成功！共 {len(df_result)} 行。")
            else:
                st.warning("查询结果为空。")

    # ===========================
    # 3. 数据处理与展示流水线
    # ===========================
    if "raw_df" in st.session_state:
        raw_df = st.session_state["raw_df"]
        
        # 展示生成的 SQL (折叠)
        with st.expander("查看原始 SQL 语句"):
            st.code(st.session_state.get("current_sql", ""), language="sql")

        st.divider()
        
        # --- 二段配置：衍生变量计算 ---
        st.subheader("🧮 衍生变量计算 (二段配置)")
        st.caption("在此处定义计算规则，例如：量表总分 = Q1 + Q2 + ...")
        
        # 确保规则列表存在
        if "calc_rules" not in st.session_state:
            st.session_state["calc_rules"] = []

        # [A] 添加新规则的表单
        with st.expander("➕ 添加新计算规则", expanded=True):
            c1, c2, c3, c4 = st.columns([2, 3, 2, 1])
            
            # 关键：这里要让用户能选到“之前规则生成的新列”
            # 我们做一次模拟推演，获取所有潜在的列名
            current_cols = list(raw_df.columns) + [r['name'] for r in st.session_state["calc_rules"]]
            
            with c1:
                new_col_name = st.text_input("新变量名", placeholder="例如: LCQ_Total")
            with c2:
                target_cols = st.multiselect("参与计算的列", options=current_cols)
            with c3:
                calc_method = st.selectbox("计算方式", ["求和 (Sum)", "平均值 (Mean)", "最大值 (Max)", "最小值 (Min)"])
            with c4:
                st.write("") # 占位，让按钮对齐底部
                st.write("")
                if st.button("添加"):
                    if new_col_name and target_cols:
                        # 检查变量名是否重复
                        if new_col_name in current_cols:
                            st.error("变量名已存在，请换一个名字。")
                        else:
                            rule = {
                                "name": new_col_name,
                                "cols": target_cols,
                                "method": calc_method
                            }
                            st.session_state["calc_rules"].append(rule)
                            st.rerun() # 刷新页面以应用新规则
                    else:
                        st.error("请填写完整信息")

        # [B] 展示和管理已有的规则
        if st.session_state["calc_rules"]:
            st.markdown("##### 已应用的计算流程：")
            for i, rule in enumerate(st.session_state["calc_rules"]):
                col1, col2 = st.columns([8, 1])
                with col1:
                    # 格式化显示：变量 = Method(列1, 列2...)
                    cols_str = ", ".join(rule['cols'])
                    if len(cols_str) > 80: cols_str = cols_str[:80] + "..."
                    st.info(f"**Step {i+1}:** `{rule['name']}` = **{rule['method']}** ( {cols_str} )")
                with col2:
                    if st.button("🗑️", key=f"del_rule_{i}"):
                        st.session_state["calc_rules"].pop(i)
                        st.rerun()

        # [D] 保存计算规则到数据库（仅二段配置）
        if st.button("💾 保存计算规则"):
            from utils import save_calculation_config

            save_calculation_config(selected_row["setup_name"], st.session_state["calc_rules"])
            st.success("二段计算规则已保存。")

        # [C] 实时执行计算流水线
        # 这一步非常快，因为是在内存中操作 Pandas
        final_df = apply_calculations(raw_df, st.session_state["calc_rules"])

        # --- 结果展示区 ---
        st.divider()

        # 先展示数据预览
        st.subheader("📄 数据预览")
        st.write(
            f"原始列数: **{len(raw_df.columns)}** | 计算后列数: **{len(final_df.columns)}**"
        )
        st.dataframe(final_df, use_container_width=True)

        csv = final_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 下载最终数据 (CSV)",
            data=csv,
            file_name="analysis_final.csv",
            mime="text/csv",
        )

        # 紧接着展示透视分析区域
        st.divider()
        st.subheader("📊 透视分析 & 图表")

        # 使用包含新变量的 final_df 进行透视
        all_columns = list(final_df.columns)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            idx = st.multiselect("行维度 (Index)", options=all_columns)
        with c2:
            col = st.multiselect("列维度 (Columns)", options=all_columns)
        with c3:
            val = st.multiselect("值字段 (Values)", options=all_columns)
        with c4:
            agg = st.selectbox(
                "聚合函数", ["mean", "sum", "count", "min", "max", "std"]
            )

        if not (idx and col and val):
            st.info("👆 请先选择【行维度、列维度和值字段】之后，再进行透视和绘图。")
        else:
            # 透视表
            try:
                # 在透视前，确保值字段列为数值类型
                pivot_source = final_df.copy()
                for v in val:
                    pivot_source[v] = pd.to_numeric(pivot_source[v], errors="coerce")

                pivot = pd.pivot_table(
                    pivot_source,
                    index=idx or None,
                    columns=col or None,
                    values=val,
                    aggfunc=agg,
                )
                st.dataframe(pivot, use_container_width=True)

                pivot_csv = pivot.to_csv().encode("utf-8-sig")
                st.download_button(
                    label="📥 下载透视结果",
                    data=pivot_csv,
                    file_name="pivot_table.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"透视表生成失败: {e}")

            # 只有在行维度、列维度、值字段各选 1 个时，才绘制图表
            if len(idx) == 1 and len(col) == 1 and len(val) == 1:
                row_field = idx[0]
                col_field = col[0]
                value_field = val[0]

                st.markdown("----")
                st.subheader("📈 透视单元格分布图")

                # 选择受试者 ID 列
                id_candidates = ["SUBJID","USUBJID", "SUBJECTID", "ID"]
                default_id_idx = 0
                for token in id_candidates:
                    for i, c in enumerate(all_columns):
                        if token in c.upper():
                            default_id_idx = i
                            break
                    else:
                        continue
                    break

                subj_col = st.selectbox(
                    "受试者 ID 列",
                    options=all_columns,
                    index=default_id_idx,
                )

                # 行 / 列取值
                row_values = (
                    final_df[row_field].dropna().astype(str).drop_duplicates().tolist()
                )
                col_values = (
                    final_df[col_field].dropna().astype(str).drop_duplicates().tolist()
                )

                # 一行一个图表：遍历行维度和列维度的笛卡尔积
                for rv in row_values:
                    for cv in col_values:
                        
                        cell_df = final_df[
                            (final_df[row_field].astype(str) == rv)
                            & (final_df[col_field].astype(str) == cv)
                        ]
                        
                        title = f"{row_field}={rv} | {col_field}={cv}"
                        key = f"cell_{row_field}_{rv}_{col_field}_{cv}"
                        draw_spaghetti_chart(
                            cell_df,
                            subj_col=subj_col,
                            value_col=value_field,
                            title=title,
                            key=key,
                        )
                        
                # 若有选中的受试者，则在最底部展示其全程明细
                subj_id = st.session_state.get("selected_subject_id")
                if subj_id is not None:
                    st.markdown("----")
                    st.subheader(f"🔍 受试者 {subj_id} 全程明细")
                    detail_df = (
                        final_df[final_df[subj_col] == subj_id]
                        .sort_values(by=row_field)
                        .reset_index(drop=True)
                    )
                    st.dataframe(detail_df, use_container_width=True)

if __name__ == "__main__":
    main()
