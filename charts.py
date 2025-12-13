import pandas as pd
import plotly.express as px
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
) -> Optional["px.Figure"]:
    """
    构建单个“透视单元格”的图表（水平柱状图），仅返回 Plotly Figure，不负责渲染。

    - 使用水平柱表示数值大小；
    - marker_color 用于区分不同行组合下的所有图表颜色；
    - 如果 agg_func 返回数值，在 X 轴上叠加一条红色参考线。
    返回:
        Plotly Figure，若数据为空则返回 None。
    """
    if df.empty:
        return None

    # 数据清洗：确保数值列有效
    tmp = df[[subj_col, value_col]].copy()
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.dropna(subset=[value_col])
    
    if tmp.empty:
        return None

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
    
    # --- 绘图逻辑：水平柱状图 ---
    bar_kwargs: Dict[str, Any] = dict(
        x=value_col,            # 数值在 X 轴
        y=subj_col,             # 受试者 / 行维度在 Y 轴
        orientation="h",        # 水平柱
        opacity=0.8,
        hover_data={subj_col: True, value_col: True},
        # 将 ID 放入 custom_data 以便回调获取
        custom_data=[subj_col],
    )
    # 若指定 marker_color，则强制整张图使用同一颜色，代表所属的行组合
    if marker_color is not None:
        bar_kwargs["color_discrete_sequence"] = [marker_color]

    fig = px.bar(tmp, **bar_kwargs)

    # 显示数值标签（柱子右侧）
    fig.update_traces(text=tmp[value_col].round(2), textposition="outside")

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

    return fig


def render_spaghetti_fig(fig, key: str) -> None:
    """
    使用 Streamlit 渲染图表，并处理点击交互，将选中的 ID 写入 session_state。
    """
    # --- 交互事件 ---
    st.plotly_chart(
        fig,
        width="stretch",
        on_select="rerun",       # 点击后触发 Streamlit 重新运行
        selection_mode="points", # 点选模式
        key=key,
    )

    # --- 点击回调处理 ---
    chart_state = st.session_state.get(key)
    if chart_state:
        # 兼容不同 Streamlit 版本的结构 (dict 或 object)
        selection = (
            chart_state.get("selection")
            if isinstance(chart_state, dict)
            else getattr(chart_state, "selection", None)
        )

        if selection and selection.get("points"):
            pt = selection["points"][0]
            # 优先取 customdata (ID)，取不到则取 y 轴值
            custom_data = pt.get("customdata")
            selected_id = custom_data[0] if custom_data else pt.get("y")

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
    绘制单个“透视单元格”的图表组件，并处理交互：
    - 调用 build_spaghetti_fig 构建水平柱状图；
    - 使用 Streamlit 渲染；
    - 处理点击事件，将选中的 ID 写入 session_state。
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
