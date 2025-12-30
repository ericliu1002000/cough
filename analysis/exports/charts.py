"""Export utilities for chart HTML bundles."""

from __future__ import annotations

import html
from typing import Any


def build_charts_export_html(items: list[dict[str, Any]]) -> str:
    """Return a full HTML document containing exported charts."""
    chart_type = items[0].get("chart_type") if items else None
    if chart_type in {"uniform", "uniform_min_max", "uniform_log"}:
        layout_class = "charts-grid"
    elif chart_type == "line":
        layout_class = "charts-line-grid"
    elif chart_type == "classic":
        layout_class = "charts-classic-grid"
    else:
        layout_class = "charts-stack"

    html_blocks: list[str] = []
    for item in items:
        fig = item["fig"]
        title_html = item.get("title_html", "")
        legend_items = item.get("legend_items", [])
        fig_html = fig.to_html(
            full_html=False,
            include_plotlyjs=False,
            config={"responsive": True},
        )

        legend_lines = []
        for legend_item in legend_items:
            dash_style = (
                "dashed" if legend_item.get("dash") == "dash" else "solid"
            )
            line_color = legend_item.get("color", "#c00")
            label_text = html.escape(str(legend_item.get("label", "Agg")))
            value_text = legend_item.get("value")
            try:
                value_fmt = f"{float(value_text):.2f}"
            except Exception:
                value_fmt = "-"
            legend_lines.append(
                f"<div class='legend-line' style='color:{line_color};'>"
                f"<span class='legend-rule {dash_style}' "
                f"style='border-top-color:{line_color};'></span>"
                f"<span>{label_text}: {value_fmt}</span>"
                "</div>"
            )
        legend_html = (
            "<div class='chart-legend'>" + "".join(legend_lines) + "</div>"
            if legend_lines
            else ""
        )

        html_blocks.append(
            "<div class='chart-card'>"
            f"<div class='chart-title'>{title_html}</div>"
            f"<div class='chart-wrap'>{fig_html}</div>"
            f"{legend_html}"
            "</div>"
        )

    full_html = (
        "<html><head>"
        "<meta charset='utf-8' />"
        "<style>"
        "body{font-family:Arial,Helvetica,sans-serif;}"
        ".charts-grid{display:grid;grid-template-columns:"
        "repeat(3,minmax(0,1fr));gap:16px;}"
        ".charts-line-grid{display:grid;grid-template-columns:"
        "repeat(3,minmax(0,1fr));gap:16px;}"
        ".charts-classic-grid{display:grid;grid-template-columns:"
        "repeat(3,minmax(0,1fr));gap:16px;}"
        ".charts-stack{display:flex;flex-direction:column;gap:16px;}"
        ".chart-card{width:100%;}"
        ".chart-title{text-align:center;font-weight:600;"
        "font-size:16px;line-height:1.2;margin-bottom:8px;}"
        ".chart-wrap{width:100%;}"
        ".charts-grid .chart-wrap{aspect-ratio:1/1;}"
        ".charts-line-grid .chart-wrap{height:360px;}"
        ".chart-wrap .js-plotly-plot,"
        ".chart-wrap .plot-container,"
        ".chart-wrap .svg-container{width:100% !important;"
        "height:100% !important;}"
        ".chart-legend{margin-top:4px;}"
        ".legend-line{display:flex;justify-content:center;"
        "align-items:center;gap:8px;font-size:12px;"
        "line-height:1.2;margin-top:2px;}"
        ".legend-rule{display:inline-block;width:32px;"
        "border-top:3px solid #c00;}"
        ".legend-rule.dashed{border-top-style:dashed;}"
        "@media (max-width: 900px){"
        ".charts-grid,.charts-line-grid,.charts-classic-grid{"
        "grid-template-columns:repeat(2,minmax(0,1fr));}"
        "}"
        "@media (max-width: 600px){"
        ".charts-grid,.charts-line-grid,.charts-classic-grid{"
        "grid-template-columns:1fr;}"
        "}"
        "</style>"
        "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>"
        "</head><body>"
        + f"<div class='{layout_class}'>"
        + "\n".join(html_blocks)
        + "</div>"
        + "</body></html>"
    )
    return full_html
