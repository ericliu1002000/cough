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
    save_calculation_config
)

# 引入插件系统
from analysis_methods import CALC_METHODS, AGG_METHODS
# 引入独立的图表组件 (请确保 cough/charts.py 已创建)
from charts import draw_spaghetti_chart

st.set_page_config(page_title="分析仪表盘", layout="wide")
st.title("📊 分析仪表盘")


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
    
    功能：
    将纵向数据 (Long Format) 中的基线行数值，横向广播到该受试者的每一行。
    """
    if not config or not isinstance(config, dict):
        return df
    
    subj_col = config.get("subj_col")
    visit_col = config.get("visit_col")
    baseline_val = config.get("baseline_val")
    target_cols = config.get("target_cols", [])

    # 参数校验
    if not (subj_col and visit_col and baseline_val and target_cols):
        return df
    
    # 容错：确保所需列存在 (可能配置了但还没算出来)
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
            
            # 如果所需列不全（比如缺了基线列），在 Pass 1 阶段跳过，不报错
            if len(valid_cols) < len(cols):
                continue

            # 强制转数值
            subset = df_calc[valid_cols].apply(pd.to_numeric, errors='coerce')

            # 调用插件
            if method_name in CALC_METHODS:
                calc_func = CALC_METHODS[method_name]
                df_calc[name] = calc_func(subset)
            else:
                # 只有找不到方法时才警告
                st.warning(f"⚠️ 找不到计算方法: {method_name}")
                
        except Exception:
            # 静默失败，允许 Pass 2 重试
            pass
            
    return df_calc


# ==========================================
# UI 表现层 (Main)
# ==========================================

def main() -> None:
    # --- 1. 侧边栏：加载配置 ---
    with st.sidebar:
        st.header("🧩 分析集配置")
        setups = fetch_all_setups()

        if not setups:
            st.info("暂无配置。请先去主页创建。")
            return

        option_labels = [f"{row['setup_name']}" for row in setups]
        selected_label = st.selectbox("选择配置", options=option_labels)
        
        # 找到选中的配置对象
        selected_row = next(r for r in setups if f"{r['setup_name']}" == selected_label)
        
        if selected_row.get("description"):
            st.info(f"📝 **备注**: {selected_row['description']}")

    # --- 1.1 状态管理与初始化 ---
    # 检测配置是否切换，如果切换则重新加载二段配置
    if "current_setup_name" not in st.session_state:
        st.session_state["current_setup_name"] = selected_row["setup_name"]
        need_reload = True
    else:
        need_reload = st.session_state["current_setup_name"] != selected_row["setup_name"]

    if need_reload:
        st.session_state["current_setup_name"] = selected_row["setup_name"]
        
        # 从数据库加载完整配置
        cfg_pack = fetch_setup_config(selected_row["setup_name"]) or {}
        calc_cfg = cfg_pack.get("calculation") or {}
        
        # 兼容旧版本数据结构
        if isinstance(calc_cfg, list):
            calc_cfg = {"calc_rules": calc_cfg}
            
        # 初始化 Session State
        st.session_state["calc_rules"] = calc_cfg.get("calc_rules", [])
        st.session_state["calc_note"] = calc_cfg.get("note", "")
        st.session_state["exclusions"] = calc_cfg.get("exclusions", [])
        st.session_state["pivot_config"] = calc_cfg.get("pivot", {})
        st.session_state["baseline_config"] = calc_cfg.get("baseline", {}) # [新增] 基线配置

        # 同步 UI 控件状态
        p_cfg = st.session_state["pivot_config"]
        st.session_state["pivot_index"] = p_cfg.get("index", [])
        st.session_state["pivot_columns"] = p_cfg.get("columns", [])
        st.session_state["pivot_values"] = p_cfg.get("values", [])
        st.session_state["pivot_agg"] = p_cfg.get("agg", "Mean - 平均值")

        # 清空旧数据缓存
        st.session_state.pop("raw_df", None)
        st.session_state.pop("current_sql", None)
        st.session_state.pop("selected_subject_id", None)

    # --- 2. 加载源数据 (Extraction) ---
    if st.button("🚀 加载源数据", type="primary"):
        full_cfg = fetch_setup_config(selected_row["setup_name"])
        if full_cfg and full_cfg.get("extraction"):
            sql, df_res = run_analysis(full_cfg["extraction"])
            if not df_res.empty:
                st.session_state["raw_df"] = df_res
                st.session_state["current_sql"] = sql
                st.success(f"数据加载成功！共 {len(df_res)} 行。")
            else:
                st.warning("查询结果为空。")

    # --- 3. 数据处理流水线 (Pipeline) ---
    if "raw_df" in st.session_state:
        raw_df = st.session_state["raw_df"]
        
        # -------------------------------------------------------
        # 【Pass 1: 预计算】
        # 先算一遍衍生变量 (如 Total)，为了让基线配置能选到它们
        # -------------------------------------------------------
        df_pass1 = apply_calculations(raw_df, st.session_state["calc_rules"])
        
        # 此时 df_pass1 包含了 "总分" 列，但可能还没有 "Total_BL" 和 "Change"
        all_cols_pass1 = list(df_pass1.columns)

        with st.expander("查看原始 SQL"):
            st.code(st.session_state.get("current_sql", ""), language="sql")
        
        st.divider()

        # ==========================================
        # [Step A] 基线变量映射 (BDS Engine)
        # ==========================================
        st.subheader("🧬 基线变量映射 (BDS)")
        st.caption("在此定义基线（支持选择刚刚计算出的衍生变量），系统会自动生成 `_BL` 后缀变量。")
        
        # 读取当前基线配置
        bl_cfg = st.session_state.get("baseline_config", {})
        
        # UI 配置区
        with st.expander("⚙️ 配置基线逻辑", expanded=not bool(bl_cfg)):
            c1, c2, c3 = st.columns(3)
            
            # 智能猜测列名默认值
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
                # 尝试恢复已保存的基线值
                try:
                    saved_bl_val = bl_cfg.get("baseline_val")
                    bl_idx = unique_visits.index(saved_bl_val) if saved_bl_val in unique_visits else 0
                except:
                    bl_idx = 0
                baseline_val = st.selectbox("哪一个访视是基线?", unique_visits, index=bl_idx, key="bl_val_ui")
            
            # 【关键】这里的 options 使用 all_cols_pass1，包含了 Pass 1 算出来的变量
            target_cols = st.multiselect(
                "选择数值变量 (生成 _BL 列)", 
                options=all_cols_pass1,
                default=[c for c in bl_cfg.get("target_cols", []) if c in all_cols_pass1],
                key="bl_targets_ui"
            )
            
            if st.button("✅ 应用基线配置"):
                st.session_state["baseline_config"] = {
                    "subj_col": subj_col,
                    "visit_col": visit_col,
                    "baseline_val": baseline_val,
                    "target_cols": target_cols
                }
                st.rerun()

        # 提示用户已生成的变量
        if st.session_state.get("baseline_config"):
            targets = st.session_state["baseline_config"].get("target_cols", [])
            if targets:
                new_cols_str = ", ".join([f"`{c}_BL`" for c in targets])
                st.info(f"已生成基线变量：{new_cols_str}")

        st.divider()

        # ==========================================
        # [Step B] 衍生变量计算
        # ==========================================
        st.subheader("🧮 衍生变量计算")
        
        # -------------------------------------------------------
        # 【模拟基线映射】
        # 为了让“添加规则”UI 能选到 _BL 变量，我们需要先模拟跑一次映射
        # -------------------------------------------------------
        df_preview_bl = apply_baseline_mapping(df_pass1, st.session_state.get("baseline_config", {}))
        
        # 此时的可用列 = 原始 + Pass1变量 + 基线变量 + 已定义变量名
        current_cols = list(df_preview_bl.columns) + [r['name'] for r in st.session_state["calc_rules"]]
        
        with st.expander("➕ 添加新计算规则", expanded=True):
            c1, c2, c3, c4 = st.columns([2, 3, 2, 1])
            with c1: 
                new_name = st.text_input("新变量名", placeholder="例: Score_Change")
            with c2: 
                targets_sel = st.multiselect("参与计算的列", options=current_cols)
            with c3: 
                # 动态读取插件列表
                method = st.selectbox("计算方式", options=list(CALC_METHODS.keys()))
            with c4:
                st.write("")
                st.write("")
                if st.button("添加"):
                    if new_name and targets_sel:
                        st.session_state["calc_rules"].append({
                            "name": new_name, 
                            "cols": targets_sel, 
                            "method": method
                        })
                        st.rerun()
                    else:
                        st.error("请填写完整")

        # 展示已配置规则
        if st.session_state["calc_rules"]:
            for i, rule in enumerate(st.session_state["calc_rules"]):
                c1, c2 = st.columns([8, 1])
                c1.markdown(f"**Step {i+1}:** `{rule['name']}` = **{rule['method']}** ( {', '.join(rule['cols'])} )")
                if c2.button("🗑️", key=f"del_rule_{i}"):
                    st.session_state["calc_rules"].pop(i)
                    st.rerun()

        # ==========================================
        # [Step C] 数据剔除 (Filters)
        # ==========================================
        st.divider()
        st.markdown("##### 🗑️ 数据剔除规则")
        st.caption("剔除不需要的行（如筛选失败的受试者）。")

        with st.expander("配置剔除条件"):
            ec1, ec2 = st.columns([2, 3])
            
            # 读取当前默认值
            cur_excl = st.session_state.get("exclusions", [])
            def_field = cur_excl[0]["field"] if cur_excl else (current_cols[0] if current_cols else None)
            def_vals = cur_excl[0]["values"] if cur_excl else []
            
            with ec1:
                # 尝试找到默认字段的索引
                try: f_idx = current_cols.index(def_field) if def_field in current_cols else 0
                except: f_idx = 0
                excl_field = st.selectbox("字段名", current_cols, index=f_idx, key="ex_f")
            
            with ec2:
                # 获取唯一值供选择
                if excl_field and excl_field in df_preview_bl.columns:
                    u_vals = df_preview_bl[excl_field].astype(str).unique().tolist()[:200]
                    excl_values = st.multiselect("剔除值 (Not In)", u_vals, default=def_vals, key="ex_v")
                else:
                    excl_values = []

            # 自动保存剔除规则到 Session (简化版：只支持一条规则)
            if excl_values:
                st.session_state["exclusions"] = [{"field": excl_field, "values": excl_values}]
            else:
                st.session_state["exclusions"] = []
                
        if st.session_state.get("exclusions"):
            r = st.session_state["exclusions"][0]
            st.info(f"当前剔除: `{r['field']}` NOT IN {r['values']}")

        # ==========================================
        # [Step D] 备注与保存
        # ==========================================
        st.markdown("##### 📝 备注")
        st.text_area("分析备注", key="calc_note", height=80)

        st.divider()
        if st.button("💾 保存所有配置 (基线+计算+剔除+透视)"):
            payload = {
                "baseline": st.session_state.get("baseline_config", {}), # [保存] 基线配置
                "calc_rules": st.session_state["calc_rules"],
                "note": st.session_state.get("calc_note", ""),
                "exclusions": st.session_state.get("exclusions", []),
                "pivot": {
                    "index": st.session_state.get("pivot_index"),
                    "columns": st.session_state.get("pivot_columns"),
                    "values": st.session_state.get("pivot_values"),
                    "agg": st.session_state.get("pivot_agg")
                }
            }
            save_calculation_config(selected_row["setup_name"], payload)
            st.success("✅ 配置已全部保存！")

        # =======================================================
        # 【最终执行流水线 (The Sandwich Pipeline)】
        # 1. 原始数据 -> 2. Pass1计算 -> 3. 基线映射 -> 4. 剔除 -> 5. Pass2计算
        # =======================================================
        
        # Step 1: 原始数据
        final_df = raw_df.copy()
        
        # Step 2: Pass 1 计算 (算出 Total 等)
        # 此时关于 _BL 的计算会失败，但没关系，apply_calculations 会静默跳过
        final_df = apply_calculations(final_df, st.session_state["calc_rules"])
        
        # Step 3: 基线映射 (生成 _BL 变量)
        final_df = apply_baseline_mapping(final_df, st.session_state.get("baseline_config", {}))
        
        # Step 4: 剔除数据
        if st.session_state.get("exclusions"):
            for rule in st.session_state["exclusions"]:
                f, vals = rule.get("field"), rule.get("values")
                if f and f in final_df.columns and vals:
                    # 执行 NOT IN 过滤
                    final_df = final_df[~final_df[f].astype(str).isin([str(v) for v in vals])]
        
        # Step 5: Pass 2 计算 (算出 Change 等)
        # 此时 _BL 变量已存在，之前失败的计算规则现在可以成功执行了
        final_df = apply_calculations(final_df, st.session_state["calc_rules"])

        # ==========================================
        # [Step E] 透视分析 & 绘图
        # ==========================================
        st.divider()
        st.subheader("📊 透视分析")

        # 数据预览
        with st.expander("📄 最终数据预览"):
            st.dataframe(final_df.head(100), use_container_width=True)
            st.download_button("📥 下载最终数据", final_df.to_csv(index=False).encode("utf-8-sig"), "final_data.csv")

        all_final_cols = list(final_df.columns)
        
        # 透视控件
        c1, c2, c3, c4 = st.columns(4)
        with c1: 
            idx = st.multiselect("行维度", all_final_cols, key="pivot_index")
        with c2: 
            col = st.multiselect("列维度", all_final_cols, key="pivot_columns")
        with c3: 
            val = st.multiselect("值字段", all_final_cols, key="pivot_values")
        with c4: 
            agg_name = st.selectbox("聚合函数", list(AGG_METHODS.keys()), key="pivot_agg")

        if idx and col and val:
            try:
                # 准备数据 (再次确保数值化，防止透视报错)
                p_src = final_df.copy()
                for v in val:
                    p_src[v] = pd.to_numeric(p_src[v], errors='coerce')
                
                # 获取函数对象
                actual_func = AGG_METHODS.get(agg_name, "mean")
                
                # 生成透视表
                pivot = pd.pivot_table(
                    p_src, index=idx, columns=col, values=val, 
                    aggfunc=actual_func
                )
                st.dataframe(pivot, use_container_width=True)
                
                # 下载
                st.download_button("📥 下载透视结果", pivot.to_csv().encode("utf-8-sig"), "pivot_table.csv")

            except Exception as e:
                st.error(f"透视表生成失败: {e}")

            # ==========================
            # 绘图区域 (调用 charts.py)
            # ==========================
            # 只有在维度确定时才绘图
            if len(idx) == 1 and len(col) == 1 and len(val) == 1:
                st.markdown("---")
                st.subheader("📈 单元格分布图")
                
                row_field = idx[0]
                col_field = col[0]
                val_field = val[0]
                
                # 智能选择 ID 列
                def_id_idx = next((i for i, c in enumerate(all_final_cols) if "SUBJ" in c.upper()), 0)
                subj_col = st.selectbox("受试者 ID 列 (用于绘图)", all_final_cols, index=def_id_idx)

                # 遍历绘制小图
                row_vals = final_df[row_field].dropna().astype(str).drop_duplicates().tolist()
                col_vals = final_df[col_field].dropna().astype(str).drop_duplicates().tolist()
                
                # 限制绘图数量，防止浏览器卡死
                total_charts = len(row_vals) * len(col_vals)
                if total_charts > 20:
                    st.warning(f"⚠️ 图表数量过多 ({total_charts})，仅展示前 20 个。")
                
                count = 0
                for rv in row_vals:
                    for cv in col_vals:
                        if count >= 20: break
                        
                        # 提取单元格数据
                        cell_df = final_df[
                            (final_df[row_field].astype(str) == rv) & 
                            (final_df[col_field].astype(str) == cv)
                        ]
                        
                        title = f"{row_field}={rv} | {col_field}={cv}"
                        key = f"chart_{rv}_{cv}"
                        
                        # [重构] 调用外部组件
                        draw_spaghetti_chart(
                            df=cell_df,
                            subj_col=subj_col,
                            value_col=val_field,
                            title=title,
                            key=key,
                            agg_func=actual_func,
                            agg_name=agg_name
                        )
                        count += 1

if __name__ == "__main__":
    main()