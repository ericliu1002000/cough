"""Uniform log-axis chart helpers for pivoted data views."""

import math
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, List, Optional


def _build_exp_ticks(x_range: tuple[float, float]) -> tuple[list[int], list[str]]:
    start = math.floor(x_range[0])
    end = math.ceil(x_range[1])
    if end < start:
        return [], []
    tick_vals = list(range(start, end + 1))
    tick_text = [f"e<sup>{k}</sup>" for k in tick_vals]
    return tick_vals, tick_text


def compute_uniform_axes(
    df: pd.DataFrame,
    row_key_cols: List[str],
    col_key_cols: List[str],
    value_col: str,
) -> tuple[Optional[tuple[float, float]], int]:
    """
    计算全局 X 轴范围和最大行数（用于统一坐标系与统一高度）。
    X 轴范围为对数空间的最小值到最大值，并向左扩展 10%。
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
    plot_df = plot_df[plot_df["_val_num"] > 0]

    if plot_df.empty:
        return None, 0

    x_max = float(plot_df["_val_num"].max())
    x_min = float(plot_df["_val_num"].min())
    x_max = math.log(x_max)
    x_min = math.log(x_min)

    # 统计每个单元格里有多少条（用于统一 Y 轴高度）
    if key_cols:
        counts = plot_df.groupby(key_cols)["_val_num"].size()
        y_max_count = int(counts.max()) if not counts.empty else 0
    else:
        y_max_count = int(len(plot_df))

    if x_max == x_min:
        pad = 1.0 if x_max == 0 else abs(x_max) * 0.05
        x_min -= pad
        x_max += pad
    else:
        pad = (x_max - x_min) * 0.1
        x_min -= pad
    return (x_min, x_max), y_max_count


def build_uniform_spaghetti_fig(
    df: pd.DataFrame,
    subj_col: str,
    value_col: str,
    title: str,
    x_range: Optional[tuple[float, float]] = None,
    y_max_count: Optional[int] = None,
    control_mean: Optional[float] = None,
    control_median: Optional[float] = None,
    marker_color: Optional[str] = None,
) -> Optional["go.Figure"]:
    """
    对数坐标的统一横向柱状图：
    - X 轴范围使用对数空间的最小/最大值，并向左延伸
    - Y 轴统一为最大行数（较少的单元格留白）
    - 所有图保持正方形尺寸
    """
    if df.empty:
        return None

    # 仅保留绘图所需列
    tmp = df[[subj_col, value_col]].copy()
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.dropna(subset=[value_col])
    tmp = tmp[tmp[value_col] > 0]

    if tmp.empty:
        return None

    # 按数值降序排序（柱子从上到下由大到小）
    tmp = tmp.sort_values(by=value_col, ascending=False)

    x_vals_actual = tmp[value_col].values.tolist()
    x_vals_log = [math.log(float(v)) for v in x_vals_actual]
    y_labels = tmp[subj_col].values.tolist()
    x_base = float(x_range[0]) if x_range else float(min(x_vals_log))
    x_vals = [v - x_base for v in x_vals_log]

    # 统一 Y 轴长度：若未给全局最大行数，则退回本图行数
    if y_max_count is None:
        y_max_count = len(x_vals_actual)
    if y_max_count <= 0:
        return None

    # 预留最下方一行放总体箱线图
    y_total_count = y_max_count + 1

    # 使用数值轴模拟“类别轴”，便于固定范围并留白
    start_pos = y_total_count - 1
    y_positions = list(range(start_pos, start_pos - len(x_vals_actual), -1))
    y_position_map = {
        str(pos): label for pos, label in zip(y_positions, y_labels)
    }

    # 绘图区由容器宽度决定（在渲染阶段用 CSS 固定为正方形）

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x_vals,
            base=x_base,
            y=y_positions,
            orientation="h",
            marker=dict(
                color=marker_color if marker_color else "#636efa",
                opacity=0.8,
            ),
            showlegend=False,
            # 直接在柱子末端显示数值标签
            text=[f"{v:.2f}" for v in x_vals_actual],
            textposition="outside",
            # 使用 customdata 保留真实的受试者标识
            customdata=[
                [label, value]
                for label, value in zip(y_labels, x_vals_actual)
            ],
            hovertemplate=(
                f"<b>{subj_col}</b>: %{{customdata[0]}}<br>"
                f"<b>{value_col}</b>: %{{customdata[1]:.2f}}<br>"
                "<extra></extra>"
            ),
        )
    )

    x_series = pd.Series(x_vals_log)
    q1 = float(x_series.quantile(0.25))
    median = float(x_series.quantile(0.5))
    q3 = float(x_series.quantile(0.75))

    box_trace_index = len(fig.data)
    fig.add_trace(
        go.Box(
            y=[0],
            orientation="h",
            q1=[q1],
            median=[median],
            q3=[q3],
            lowerfence=[q1],
            upperfence=[q3],
            boxpoints=False,
            width=0.95,
            whiskerwidth=0,
            line=dict(color="#111111", width=2),
            fillcolor="rgba(0,0,0,0.08)",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    mean_marker_index = None
    if x_vals_actual:
        current_mean = float(x_series.mean())
        current_mean_actual = float(math.exp(current_mean))
        mean_marker_index = len(fig.data)
        fig.add_trace(
            go.Scatter(
                x=[current_mean],
                y=[0],
                mode="markers",
                marker=dict(
                    symbol="circle",
                    size=10,
                    color="#111111",
                ),
                hovertemplate=(
                    f"当前组均值: {current_mean_actual:.2f}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    line_y_min = 0.5
    line_y_max = y_total_count - 0.5

    # 对照组参考线：红色为均值，绿色为中位数
    legend_items = []
    if control_mean is not None and control_mean > 0:
        control_mean_log = math.log(control_mean)
        fig.add_shape(
            type="line",
            x0=control_mean_log,
            x1=control_mean_log,
            y0=line_y_min,
            y1=line_y_max,
            xref="x",
            yref="y",
            line=dict(color="#c00", width=3, dash="solid"),
        )
        legend_items.append(
            {
                "label": "对照组均值",
                "value": control_mean,
                "dash": "solid",
                "color": "#c00",
            }
        )
    if control_median is not None and control_median > 0:
        control_median_log = math.log(control_median)
        fig.add_shape(
            type="line",
            x0=control_median_log,
            x1=control_median_log,
            y0=line_y_min,
            y1=line_y_max,
            xref="x",
            yref="y",
            line=dict(color="#2ca02c", width=3, dash="solid"),
        )
        legend_items.append(
            {
                "label": "对照组中位数",
                "value": control_median,
                "dash": "solid",
                "color": "#2ca02c",
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
        hovermode="x",
        meta={
            "legend_items": legend_items,
            "boxplot_trace_index": box_trace_index,
            "mean_marker_trace_index": mean_marker_index,
            "y_position_map": y_position_map,
        },
    )

    yaxis_cfg = dict(
        # 固定 Y 轴范围，确保所有图留白一致
        range=[-0.5, y_total_count - 0.5],
        # 将数值轴映射为类别文本
        tickmode="array",
        tickvals=y_positions,
        ticktext=y_labels,
        type="linear",
        automargin=True,
        title_standoff=30,
    )
    fig.update_yaxes(**yaxis_cfg)

    tick_vals = []
    tick_text = []
    tick_range = x_range
    if tick_range is None and x_vals_log:
        tick_range = (min(x_vals_log), max(x_vals_log))
    if tick_range:
        tick_vals, tick_text = _build_exp_ticks(tick_range)

    # 统一 X 轴范围（对数空间最小/最大值）
    if x_range:
        fig.update_xaxes(range=list(x_range))
    xaxis_cfg = dict(
        automargin=True,
        title_standoff=12,
        showspikes=True,
        spikemode="across",
        spikethickness=1,
        spikecolor="#999999",
        spikesnap="cursor",
    )
    if tick_vals:
        xaxis_cfg.update(
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_text,
        )
    fig.update_xaxes(**xaxis_cfg)

    return fig


def resolve_uniform_control_group(
    col_key_cols: List[str],
    col_keys: List[Dict[str, str]],
    default_group: Optional[Dict[str, str]] = None,
    key: str = "uniform_control_group",
) -> Optional[Dict[str, str]]:
    """Select or resolve the control group for uniform charts."""
    if not col_key_cols or not col_keys:
        st.session_state.pop(key, None)
        return None

    def build_key(record: Dict[str, str]) -> str:
        """Return a stable key for a column-group record."""
        return "\x1f".join([str(record.get(col, "")) for col in col_key_cols])

    def build_label(record: Dict[str, str]) -> str:
        """Return a human-readable label for a column-group record."""
        return " / ".join([str(record.get(col, "")) for col in col_key_cols])

    option_map: Dict[str, Dict[str, str]] = {}
    label_map: Dict[str, str] = {}
    for option in col_keys:
        combo_key = build_key(option)
        if combo_key in option_map:
            continue
        option_map[combo_key] = option
        label_map[combo_key] = build_label(option)

    options = list(option_map.keys())
    if not options:
        st.session_state.pop(key, None)
        return None

    default_key = None
    if isinstance(default_group, dict):
        default_key = build_key(default_group)
        if default_key not in option_map:
            default_key = None
    if default_key is None:
        default_key = options[0]

    selected_key = st.selectbox(
        "对照组（列维度）",
        options=options,
        index=options.index(default_key),
        key=f"{key}_ui",
        format_func=lambda k: label_map.get(k, k),
        help="选择列维度中的对照组，用于绘制均值/中位数参考线。",
    )

    selected_group = option_map[selected_key]
    st.session_state[key] = selected_group
    return selected_group


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
            meta = getattr(fig.layout, "meta", None)
            box_trace_index = None
            mean_marker_trace_index = None
            if isinstance(meta, dict):
                box_trace_index = meta.get("boxplot_trace_index")
                mean_marker_trace_index = meta.get("mean_marker_trace_index")
            if box_trace_index is not None and pt.get("curveNumber") == box_trace_index:
                return
            if (
                mean_marker_trace_index is not None
                and pt.get("curveNumber") == mean_marker_trace_index
            ):
                return
            selected_id = None
            custom_data = pt.get("customdata")
            if isinstance(custom_data, list):
                if custom_data:
                    selected_id = custom_data[0]
            elif custom_data is not None:
                selected_id = custom_data

            if selected_id is None:
                y_val = pt.get("y")
                meta = getattr(fig.layout, "meta", None)
                if isinstance(meta, dict):
                    pos_map = meta.get("y_position_map")
                    if isinstance(pos_map, dict) and y_val is not None:
                        if isinstance(y_val, (int, float)) and int(y_val) == y_val:
                            y_key = str(int(y_val))
                        else:
                            y_key = str(y_val)
                        selected_id = pos_map.get(y_key)
                if selected_id is None and y_val is not None:
                    selected_id = y_val

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
    control_mean: Optional[float] = None,
    control_median: Optional[float] = None,
    marker_color: Optional[str] = None,
) -> None:
    """Build and render a uniform spaghetti chart in Streamlit."""
    fig = build_uniform_spaghetti_fig(
        df=df,
        subj_col=subj_col,
        value_col=value_col,
        title=title,
        x_range=x_range,
        y_max_count=y_max_count,
        control_mean=control_mean,
        control_median=control_median,
        marker_color=marker_color,
    )

    if fig is None:
        st.info("No valid numeric data for this chart.")
        return

    render_uniform_spaghetti_fig(fig, key=key)
