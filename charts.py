import plotly.express as px
import pandas as pd
import streamlit as st
from typing import Any

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
    绘制单个“透视单元格”的散点图（原 spaghetti chart）。
    已剥离至单独文件以减轻 Dashboard 负担。
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
        custom_data=[subj_col],
    )

    # 显示数值标签
    fig.update_traces(text=tmp[value_col].round(2), textposition="middle right")

    # 统计量计算与绘制
    series = tmp[value_col]
    
    if agg_func:
        try:
            # 尝试计算统计量
            agg_value = agg_func(series)
            
            # 【关键】只有数值类型才画线，字符串类型（如 Mean(SD)）则跳过
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
            # 忽略非数值结果（不画线，只画点）
            pass

    fig.update_layout(
        title=title,
        xaxis_title=value_col,
        yaxis_title=subj_col,
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    # 交互事件
    st.plotly_chart(
        fig,
        width="stretch",
        on_select="rerun",
        selection_mode="points",
        key=key,
    )

    # 处理点击回传
    chart_state = st.session_state.get(key)
    if chart_state is not None:
        selection = getattr(chart_state, "selection", None)
        if selection is None and isinstance(chart_state, dict):
            selection = chart_state.get("selection")

        if selection and "points" in selection and selection["points"]:
            clicked_point = selection["points"][0]
            custom_data = clicked_point.get("customdata")
            if custom_data:
                selected_id = custom_data[0]
            else:
                selected_id = clicked_point.get("y")

            if selected_id is not None:
                st.session_state["selected_subject_id"] = selected_id