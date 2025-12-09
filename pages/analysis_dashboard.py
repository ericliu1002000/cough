import json
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

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
    
    参数:
        df: 原始 DataFrame
        rules: 规则列表
        
    返回:
        处理后的新 DataFrame（包含新计算的列）
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

            # 2. 【关键修复】强制将参与计算的列转换为数字类型
            # errors='coerce' 意味着：如果遇到无法转换的值（如 "N/A" 或纯文本），这就变成 NaN (空值)，而不会报错卡死
            for col in valid_cols:
                # 检查一下是否已经是数字，如果不是才转，避免重复操作（虽然重复转也没事）
                if not pd.api.types.is_numeric_dtype(df_calc[col]):
                    df_calc[col] = pd.to_numeric(df_calc[col], errors='coerce')

            # 3. 根据选择的方法进行行级运算 (axis=1)
            # 此时 df_calc[valid_cols] 里的数据已经是数字或 NaN 了
            if method == '求和 (Sum)':
                # min_count=1 确保如果整行都是 NaN，结果也是 NaN 而不是 0
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
        # 获取完整的配置 JSON
        cfg = fetch_setup_config(selected_row["setup_name"])
        if cfg:
            sql, df_result = run_analysis(cfg)
            if not df_result.empty:
                # 将原始数据存入 Session State
                st.session_state["raw_df"] = df_result
                st.session_state["current_sql"] = sql
                
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

        # [C] 实时执行计算流水线
        # 这一步非常快，因为是在内存中操作 Pandas
        final_df = apply_calculations(raw_df, st.session_state["calc_rules"])

        # --- 结果展示区 ---
        st.divider()
        tab1, tab2 = st.tabs(["📄 数据预览", "📊 透视分析"])
        
        # Tab 1: 明细数据
        with tab1:
            st.write(f"原始列数: **{len(raw_df.columns)}** | 计算后列数: **{len(final_df.columns)}**")
            st.dataframe(final_df, use_container_width=True)
            
            csv = final_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 下载最终数据 (CSV)",
                data=csv,
                file_name="analysis_final.csv",
                mime="text/csv",
            )
            
        # Tab 2: 透视表
        with tab2:
            st.subheader("透视分析")
            
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
                agg = st.selectbox("聚合函数", ["mean", "sum", "count", "min", "max", "std"])
            
            if val:
                try:
                    # 生成透视表
                    pivot = pd.pivot_table(
                        final_df, 
                        index=idx or None, 
                        columns=col or None, 
                        values=val, 
                        aggfunc=agg
                    )
                    st.dataframe(pivot, use_container_width=True)
                    
                    # 下载透视结果
                    pivot_csv = pivot.to_csv().encode('utf-8-sig')
                    st.download_button(
                        label="📥 下载透视结果",
                        data=pivot_csv,
                        file_name="pivot_table.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"透视表生成失败: {e}")
            else:
                st.info("👆 请至少选择一个【值字段 (Values)】来生成透视表。")

if __name__ == "__main__":
    main()