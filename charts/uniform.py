import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from typing import Any, List, Optional



def compute_uniform_axes(
    df: pd.DataFrame,
    row_key_cols: List[str],
    col_key_cols: List[str],
    value_col: str,
) -> tuple[Optional[tuple[float, float]], int]:
    """
    计算全局 X 轴范围和最大行数（用于统一坐标系与统一高度）。
    """
    if df.empty or not value_col or value_col not in df.columns:
        return None, 0

    plot_df = df.copy()
    key_cols = [c for c in (row_key_cols + col_key_cols) if c in plot_df.columns]
    for col_name in key_cols:
        plot_df = plot_df[plot_df[col_name].notna()].copy()
        plot_df[col_name] = plot_df[col_name].astype(str)

    # 只保留可转为数值的观测点
    plot_df["_val_num"] = pd.to_numeric(plot_df[value_col], errors="coerce")
    plot_df = plot_df.dropna(subset=["_val_num"])

    if plot_df.empty:
        return None, 0

    x_max = float(plot_df["_val_num"].max())
    x_min = float(plot_df["_val_num"].min())

    # 统计每个单元格里有多少条（用于统一 Y 轴高度）
    if key_cols:
        counts = plot_df.groupby(key_cols)["_val_num"].size()
        y_max_count = int(counts.max()) if not counts.empty else 0
    else:
        y_max_count = int(len(plot_df))

    # 统一 X 轴：默认从 0 开始（便于横向比较）
    if x_min >= 0:
        x_min = 0.0

    if x_max == x_min:
        pad = 1.0 if x_max == 0 else abs(x_max) * 0.05
        x_min -= pad
        x_max += pad

    return (x_min, x_max), y_max_count


def build_uniform_spaghetti_fig(
    df: pd.DataFrame,
    subj_col: str,
    value_col: str,
    title: str,
    x_range: Optional[tuple[float, float]] = None,
    y_max_count: Optional[int] = None,
    agg_funcs: Optional[List[Any]] = None,
    agg_names: Optional[List[str]] = None,
    marker_color: Optional[str] = None,
) -> Optional["go.Figure"]:
    """
    统一坐标系的横向柱状图：
    - X 轴范围统一为全局最大值
    - Y 轴统一为最大行数（较少的单元格留白）
    - 所有图保持正方形尺寸
    """
    if df.empty:
        return None

    # 仅保留绘图所需列
    tmp = df[[subj_col, value_col]].copy()
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.dropna(subset=[value_col])

    if tmp.empty:
        return None

    # 按数值降序排序（柱子从上到下由大到小）
    tmp = tmp.sort_values(by=value_col, ascending=False)

    x_vals = tmp[value_col].values.tolist()
    y_labels = tmp[subj_col].values.tolist()

    # 统一 Y 轴长度：若未给全局最大行数，则退回本图行数
    if y_max_count is None:
        y_max_count = len(x_vals)
    if y_max_count <= 0:
        return None

    # 使用数值轴模拟“类别轴”，便于固定范围并留白
    start_pos = y_max_count - 1
    y_positions = list(range(start_pos, start_pos - len(x_vals), -1))

    # 绘图区由容器宽度决定（在渲染阶段用 CSS 固定为正方形）

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x_vals,
            y=y_positions,
            orientation="h",
            marker=dict(
                color=marker_color if marker_color else "#636efa",
                opacity=0.8,
            ),
            showlegend=False,
            # 直接在柱子末端显示数值标签
            text=[f"{v:.2f}" for v in x_vals],
            textposition="outside",
            # 使用 customdata 保留真实的受试者标识
            customdata=y_labels,
            hovertemplate=(
                f"<b>{subj_col}</b>: %{{customdata}}<br>"
                f"<b>{value_col}</b>: %{{x}}<br>"
                "<extra></extra>"
            ),
        )
    )

    # 画两条参考线：第 1 条虚线，第 2 条实线（数值标注放在图表下方）
    legend_items = []
    if agg_funcs:
        for idx, func in enumerate(agg_funcs[:2]):
            if not callable(func):
                continue
            try:
                agg_value = func(pd.Series(x_vals))
                agg_x = float(agg_value)
            except Exception:
                continue

            agg_label = (
                agg_names[idx]
                if agg_names and idx < len(agg_names)
                else f"Agg {idx + 1}"
            )
            fig.add_vline(
                x=agg_x,
                line_width=3,
                line_dash="dash" if idx == 0 else "solid",
                line_color="red",
            )
            legend_items.append(
                {
                    "label": agg_label,
                    "value": agg_x,
                    "dash": "dash" if idx == 0 else "solid",
                }
            )

    margin_pad = 32
    fig.update_layout(
        xaxis_title=value_col,
        yaxis_title=subj_col,
        autosize=True,
        margin=dict(l=margin_pad, r=margin_pad, t=margin_pad, b=margin_pad),
        showlegend=False,
        # 悬浮提示字号加大，缩小图表后更易读
        hoverlabel=dict(font=dict(size=16)),
        meta={"legend_items": legend_items},
    )

    yaxis_cfg = dict(
        # 固定 Y 轴范围，确保所有图留白一致
        range=[-0.5, y_max_count - 0.5],
        # 将数值轴映射为类别文本
        tickmode="array",
        tickvals=y_positions,
        ticktext=y_labels,
        type="linear",
        automargin=True,
        title_standoff=30,
    )
    fig.update_yaxes(**yaxis_cfg)

    # 统一 X 轴范围（全局最大值）
    if x_range:
        fig.update_xaxes(range=list(x_range))
    fig.update_xaxes(automargin=True, title_standoff=12)

    return fig


def render_uniform_spaghetti_fig(fig: "go.Figure", key: str) -> None:
    """
    渲染图表并处理点击选中。
    """
    st.plotly_chart(
        fig,
        # 让图表宽度跟随列宽，并由 CSS 固定为正方形
        width="stretch",
        on_select="rerun",
        selection_mode="points",
        key=key,
        config={"responsive": True},
    )

    chart_state = st.session_state.get(key)
    if chart_state:
        selection = (
            chart_state.get("selection")
            if isinstance(chart_state, dict)
            else getattr(chart_state, "selection", None)
        )

        if selection and selection.get("points"):
            pt = selection["points"][0]
            custom_data = pt.get("customdata")
            if isinstance(custom_data, list):
                selected_id = custom_data[0]
            else:
                selected_id = custom_data

            if selected_id is None:
                selected_id = pt.get("y")

            if selected_id is not None:
                st.session_state["selected_subject_id"] = selected_id


def draw_uniform_spaghetti_chart(
    df: pd.DataFrame,
    subj_col: str,
    value_col: str,
    title: str,
    key: str,
    x_range: Optional[tuple[float, float]] = None,
    y_max_count: Optional[int] = None,
    agg_funcs: Optional[List[Any]] = None,
    agg_names: Optional[List[str]] = None,
    marker_color: Optional[str] = None,
) -> None:
    fig = build_uniform_spaghetti_fig(
        df=df,
        subj_col=subj_col,
        value_col=value_col,
        title=title,
        x_range=x_range,
        y_max_count=y_max_count,
        agg_funcs=agg_funcs,
        agg_names=agg_names,
        marker_color=marker_color,
    )

    if fig is None:
        st.info("No valid numeric data for this chart.")
        return

    render_uniform_spaghetti_fig(fig, key=key)
