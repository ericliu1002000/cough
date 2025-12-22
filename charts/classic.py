import pandas as pd
import plotly.graph_objects as go  # 改用底层 API
import plotly.express as px        # 保留用于某些颜色序列，如果需要的话
import streamlit as st
from typing import Any, Dict, Optional


def build_spaghetti_fig(
    df: pd.DataFrame,
    subj_col: str,
    value_col: str,
    title: str,
    agg_func: Any = None,
    agg_name: str = "Mean",
    marker_color: Optional[str] = None,
) -> Optional["go.Figure"]:
    """
    【修复版 V3】采用 go.Bar + List 转换。
    确保传入 Plotly 的是纯 Python 列表，彻底消除 Numpy/Pandas 索引歧义。
    """
    if df.empty:
        return None

    # 1. 数据清洗
    tmp = df[[subj_col, value_col]].copy()
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.dropna(subset=[value_col])
    
    if tmp.empty:
        return None

    # 2. 排序
    try:
        tmp["_y_sort_key"] = pd.to_numeric(tmp[subj_col], errors="coerce")
        if tmp["_y_sort_key"].isna().all():
            raise ValueError
        tmp = tmp.sort_values(by="_y_sort_key")
    except Exception:
        tmp = tmp.sort_values(by=subj_col, key=lambda s: s.astype(str))
    
    # -------------------------------------------------------
    # 🚀 关键点 1: 转为纯 Python List
    # -------------------------------------------------------
    # Numpy 数组在某些极其特定的序列化场景下可能会带上元数据。
    # tolist() 后，这就是最普通的 [1.1, 2.2, ...]，没有任何歧义。
    x_vals = tmp[value_col].values.tolist()
    y_vals = tmp[subj_col].values.tolist()

    # 3. 手动构建 Figure
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=x_vals,
        y=y_vals,
        orientation='h',
        marker=dict(
            color=marker_color if marker_color else '#636efa',
            opacity=0.8
        ),
        # 强制显示数值标签
        text=[f"{v:.2f}" for v in x_vals],
        textposition='outside',
        # 手动定义 Hover
        hovertemplate=(
            f"<b>{subj_col}</b>: %{{y}}<br>" +
            f"<b>{value_col}</b>: %{{x}}<br>" +
            "<extra></extra>"
        ),
        # Customdata 也转为 list
        customdata=y_vals
    ))

    # 4. 辅助线
    if agg_func:
        try:
            # 计算时临时转回 Series 方便调用聚合函数
            agg_value = agg_func(pd.Series(x_vals))
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
            pass

    # 5. 布局
    title_lines = 0
    if isinstance(title, str) and title:
        title_lines = title.count("<br>") + 1
    title_font_size = 12
    title_line_height = title_font_size + 4
    title_pad_bottom = 10
    top_margin = (
        max(20, 12 + title_lines * title_line_height + title_pad_bottom)
        if title_lines
        else 20
    )

    layout_kwargs = dict(
        xaxis_title=value_col,
        yaxis_title=subj_col,
        height=400,
        margin=dict(l=20, r=20, t=top_margin, b=20),
        yaxis=dict(type='category', automargin=True, title_standoff=30),
    )
    if title_lines:
        layout_kwargs["title"] = dict(
            text=title,
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
            pad=dict(b=title_pad_bottom),
        )
        layout_kwargs["title_font"] = dict(size=title_font_size)

    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(automargin=True, title_standoff=12)

    return fig


def render_spaghetti_fig(fig, key: str) -> None:
    """
    渲染图表并处理交互。
    """
    # --- 交互事件 ---
    # 注意：go.Figure 同样支持 on_select
    st.plotly_chart(
        fig,
        width="stretch",
        on_select="rerun",       
        selection_mode="points", 
        key=key,
    )

    # --- 点击回调处理 ---
    chart_state = st.session_state.get(key)
    if chart_state:
        selection = (
            chart_state.get("selection")
            if isinstance(chart_state, dict)
            else getattr(chart_state, "selection", None)
        )

        if selection and selection.get("points"):
            pt = selection["points"][0]
            # 优先取 customdata (我们上面塞进去了)
            custom_data = pt.get("customdata")
            # 兼容处理：customdata 在 go 里通常直接就是值，不像 px 可能是列表
            if isinstance(custom_data, list):
                 selected_id = custom_data[0]
            else:
                 selected_id = custom_data

            # 兜底：如果没取到，取 y 轴的值
            if selected_id is None:
                selected_id = pt.get("y")

            if selected_id is not None:
                st.session_state["selected_subject_id"] = selected_id


def draw_spaghetti_chart(
    df: pd.DataFrame,
    subj_col: str,
    value_col: str,
    title: str,
    key: str,
    agg_func: Any = None,
    agg_name: str = "Mean",
    marker_color: Optional[str] = None,
) -> None:
    """
    入口函数
    """
    fig = build_spaghetti_fig(
        df=df,
        subj_col=subj_col,
        value_col=value_col,
        title=title,
        agg_func=agg_func,
        agg_name=agg_name,
        marker_color=marker_color,
    )

    if fig is None:
        st.info("该组合下无有效数值数据。")
        return

    render_spaghetti_fig(fig, key=key)
