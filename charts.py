import pandas as pd
import plotly.express as px
import streamlit as st
from typing import Any, Dict

def draw_spaghetti_chart(
    df: pd.DataFrame,
    subj_col: str,
    value_col: str,
    title: str,
    key: str,
    agg_func: Any = None,
    agg_name: str = "Mean"
) -> None:
    """
    绘制单个“透视单元格”的散点图 (Spaghetti Plot Component)。
    
    功能：
    1. 绘制个体散点。
    2. 如果 agg_func 返回的是数值，绘制红色的统计参考线。
    3. 处理 Streamlit 原生点击交互。
    """
    if df.empty:
        st.info("该组合下无数据。")
        return

    # 数据清洗：确保数值列有效
    tmp = df[[subj_col, value_col]].copy()
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.dropna(subset=[value_col])
    
    if tmp.empty:
        st.info("该组合下无有效数值数据。")
        return

    # Y 轴排序：不管 subj_col 是什么列，都按列内容从小到大排序
    try:
        # 优先尝试按数值排序
        tmp["_y_sort_key"] = pd.to_numeric(tmp[subj_col], errors="coerce")
        if tmp["_y_sort_key"].isna().all():
            raise ValueError
        tmp = tmp.sort_values(by="_y_sort_key")
    except Exception:
        # 回退为按字符串字典序排序
        tmp = tmp.sort_values(by=subj_col, key=lambda s: s.astype(str))
    
    # --- 绘图逻辑 ---
    fig = px.scatter(
        tmp,
        x=value_col,
        y=subj_col,
        opacity=0.6,
        hover_data={subj_col: True, value_col: True},
        # 将 ID 放入 custom_data 以便回调获取
        custom_data=[subj_col],
    )

    # 显示数值标签 (右侧)
    fig.update_traces(text=tmp[value_col].round(2), textposition="middle right")

    # --- 统计线绘制 (容错处理) ---
    if agg_func:
        try:
            # 计算统计量 (可能是 Mean, Median, 或纯字符串)
            series = tmp[value_col]
            agg_value = agg_func(series)
            
            # 【关键】只有当结果能转为 float 时才画红线
            # 复合函数如 "10.2(3.5)" 会触发 ValueError 被捕获，从而跳过画线
            agg_x = float(agg_value)
            
            fig.add_vline(
                x=agg_x,
                line_width=3,
                line_dash="dash",
                line_color="red",
                annotation_text=f"{agg_name}: {agg_x:.2f}",
                annotation_position="top",
            )
        except Exception:
            # 忽略无法画线的统计量（静默失败，保留散点）
            pass

    # 布局微调
    fig.update_layout(
        title=title,
        xaxis_title=value_col,
        yaxis_title=subj_col,
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    # --- 交互事件 ---
    st.plotly_chart(
        fig,
        width="stretch",
        on_select="rerun",       # 点击后触发 Streamlit 重新运行
        selection_mode="points", # 点选模式
        key=key,
    )

    # --- 点击回调处理 ---
    # 从 Session State 中读取刚才点击的数据
    chart_state = st.session_state.get(key)
    if chart_state:
        # 兼容不同 Streamlit 版本的结构 (dict 或 object)
        selection = chart_state.get("selection") if isinstance(chart_state, dict) else getattr(chart_state, "selection", None)
        
        if selection and selection.get("points"):
            pt = selection["points"][0]
            # 优先取 customdata (ID)，取不到则取 y 轴值
            custom_data = pt.get("customdata")
            selected_id = custom_data[0] if custom_data else pt.get("y")

            if selected_id is not None:
                st.session_state["selected_subject_id"] = selected_id
