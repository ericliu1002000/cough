"""业务数据上传服务。"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import warnings


class FileFormatError(ValueError):
    """用于标记文件格式解析失败的异常。"""


_NUMERIC_PATTERN = re.compile(r"^[+-]?\d+(\.\d+)?$")


def _contains_sheet_keyword(name: str) -> bool:
    return "sheet" in name.casefold()


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _is_numeric_string(value: str) -> bool:
    stripped = value.replace(",", "")
    return bool(_NUMERIC_PATTERN.match(stripped))


def _looks_like_date_string(value: str) -> bool:
    if not any(ch in value for ch in ("-", "/", ".", "年", "月", ":")):
        return False
    parsed = pd.to_datetime(value, errors="coerce")
    return not pd.isna(parsed)


def _classify_cell(value: object) -> str:
    if _is_blank(value):
        return "empty"
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return "date"
    if isinstance(value, (int, float)):
        return "numeric"
    if isinstance(value, str):
        text = value.strip()
        if _is_numeric_string(text):
            return "numeric"
        if _looks_like_date_string(text):
            return "date"
        return "text"
    text = str(value).strip()
    if _is_numeric_string(text):
        return "numeric"
    if _looks_like_date_string(text):
        return "date"
    return "text"


def _row_metrics(values: List[object]) -> Dict[str, float]:
    non_empty = 0
    text_count = 0
    numeric_or_date = 0
    for value in values:
        kind = _classify_cell(value)
        if kind == "empty":
            continue
        non_empty += 1
        if kind == "text":
            text_count += 1
        else:
            numeric_or_date += 1
    text_ratio = text_count / non_empty if non_empty else 0.0
    numeric_ratio = numeric_or_date / non_empty if non_empty else 0.0
    return {
        "non_empty": float(non_empty),
        "text_ratio": text_ratio,
        "numeric_ratio": numeric_ratio,
    }


def _valid_indices(header_row: pd.Series) -> List[int]:
    indices: List[int] = []
    for idx, value in enumerate(header_row.tolist()):
        if _is_blank(value):
            continue
        indices.append(idx)
    if not indices:
        indices = list(range(len(header_row)))
    return indices


def _row_values(
    raw_df: pd.DataFrame, row_no: int, indices: List[int]
) -> List[object]:
    if row_no <= 0 or row_no > len(raw_df):
        return []
    row = raw_df.iloc[row_no - 1]
    values = []
    for idx in indices:
        if idx < len(row):
            values.append(row.iloc[idx])
        else:
            values.append(None)
    return values


def _resolve_display_name_row(
    raw_df: pd.DataFrame, forced_row: Optional[int]
) -> Tuple[Optional[int], str]:
    if len(raw_df) < 2:
        return None, "DisplayName 判定：表格行数不足，未启用。"

    if forced_row is not None and forced_row > 0:
        if forced_row <= len(raw_df):
            return forced_row, f"DisplayName 判定：配置第 {forced_row} 行。"
        return (
            None,
            f"DisplayName 判定：配置第 {forced_row} 行，但表格只有 "
            f"{len(raw_df)} 行，已忽略。",
        )

    header_row = raw_df.iloc[0]
    indices = _valid_indices(header_row)
    row2_values = _row_values(raw_df, 2, indices)
    row3_values = _row_values(raw_df, 3, indices)

    row2_stats = _row_metrics(row2_values)
    row3_stats = _row_metrics(row3_values)

    row2_text = row2_stats["text_ratio"]
    row3_text = row3_stats["text_ratio"]
    decision = False
    if row2_stats["non_empty"] >= 2:
        if row3_stats["non_empty"] == 0:
            decision = row2_text >= 0.8
        else:
            decision = row2_text >= 0.6 and (row2_text - row3_text) >= 0.3

    log = (
        "DisplayName 判定："
        f"第2行文本占比 {row2_text:.2f}，"
        f"第3行文本占比 {row3_text:.2f}，"
        f"{'判定第2行为 display_name' if decision else '未识别 display_name 行'}。"
    )
    return (2 if decision else None), log


def _extract_display_name_map(
    raw_df: pd.DataFrame, display_row: int
) -> Dict[str, str]:
    if display_row <= 1 or display_row > len(raw_df):
        return {}
    header_row = raw_df.iloc[0]
    indices = _valid_indices(header_row)
    display_values = _row_values(raw_df, display_row, indices)
    mapping: Dict[str, str] = {}
    for idx, header_idx in enumerate(indices):
        column_name = header_row.iloc[header_idx]
        if _is_blank(column_name):
            continue
        display_value = display_values[idx] if idx < len(display_values) else None
        if _is_blank(display_value):
            continue
        display_text = str(display_value).strip()
        if not display_text:
            continue
        mapping[str(column_name)] = display_text
    return mapping


def _check_columns(
    source_name: str, sheet_name: str | None, columns: pd.Index
) -> None:
    if sheet_name:
        location = f"文件 '{source_name}' 的 Sheet '{sheet_name}'"
    else:
        location = f"文件 '{source_name}'"

    counter = pd.Series(columns).value_counts()
    duplicated = [col for col, count in counter.items() if count > 1]
    if duplicated:
        duplicate_list = ", ".join(repr(col) for col in duplicated)
        raise ValueError(
            f"列名冲突: {location} 中存在重复列名: {duplicate_list}"
        )

    missing_cols = []
    for col in columns:
        if col is None:
            missing_cols.append(col)
        else:
            try:
                is_nan = pd.isna(col)
            except TypeError:
                is_nan = False

            if is_nan:
                missing_cols.append(col)
            elif isinstance(col, str):
                stripped = col.strip()
                if stripped == "":
                    missing_cols.append(col)
                elif stripped.lower().startswith("unnamed:"):
                    missing_cols.append(col)

    if missing_cols:
        raise ValueError(
            f"列名缺失: {location} 中存在缺失或空列名，请检查表头。"
        )


def upload_excel(
    uploaded_file,
    engine,
    skip_sheets: List[str],
    display_name_row: Optional[int],
) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    logs: List[str] = []
    display_name_updates: Dict[str, Dict[str, str]] = {}
    uploaded_file.seek(0)
    warnings.filterwarnings(
        "ignore",
        message="Workbook contains no default style, apply openpyxl's default",
        category=UserWarning,
        module="openpyxl.styles.stylesheet",
    )
    try:
        xls = pd.ExcelFile(uploaded_file)
    except Exception as exc:
        raise FileFormatError("文件格式不正确") from exc

    with xls:
        invalid_sheets = [
            name for name in xls.sheet_names if _contains_sheet_keyword(name)
        ]
        if invalid_sheets:
            joined = ", ".join(invalid_sheets)
            raise ValueError(
                f"Sheet 名称包含 'sheet'（不区分大小写），"
                f"请改名后再上传: {joined}"
            )
        for sheet_name in xls.sheet_names:
            if sheet_name in skip_sheets:
                logs.append(f"跳过 Sheet（配置忽略）: {sheet_name}")
                continue

            logs.append(f"处理 Sheet: {sheet_name}")
            try:
                preview_rows = 3
                if display_name_row and display_name_row > preview_rows:
                    preview_rows = display_name_row
                raw_preview = pd.read_excel(
                    xls, sheet_name=sheet_name, header=None, nrows=preview_rows
                )
                if raw_preview.empty:
                    logs.append("Sheet 内容为空，跳过。")
                    continue
                chosen_row, decision_log = _resolve_display_name_row(
                    raw_preview, display_name_row
                )
                logs.append(decision_log)
                display_map = {}
                if chosen_row:
                    display_map = _extract_display_name_map(
                        raw_preview, chosen_row
                    )
                    if display_map:
                        display_name_updates[sheet_name] = display_map
                    logs.append(
                        f"DisplayName 提取：{len(display_map)} 个。"
                    )
                else:
                    logs.append("DisplayName 提取：未启用。")

                skiprows: List[int] = []
                if chosen_row and chosen_row > 1:
                    skiprows.append(chosen_row - 1)
                df = pd.read_excel(
                    xls, sheet_name=sheet_name, header=0, skiprows=skiprows
                )
            except Exception as exc:
                raise FileFormatError("文件格式不正确") from exc

            if df.empty:
                logs.append("Sheet 内容为空，跳过。")
                continue

            _check_columns(uploaded_file.name, sheet_name, df.columns)

            try:
                df.to_sql(
                    name=sheet_name,
                    con=engine,
                    if_exists="replace",
                    index=False,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"写入表失败: {sheet_name}，原因: {exc}"
                ) from exc

            logs.append(
                f"写入表: {sheet_name}，行数: {len(df)}，列数: {len(df.columns)}"
            )

    return logs, display_name_updates


def upload_csv(
    uploaded_file,
    engine,
    display_name_row: Optional[int],
) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    logs: List[str] = []
    display_name_updates: Dict[str, Dict[str, str]] = {}
    uploaded_file.seek(0)
    table_name = Path(uploaded_file.name).stem
    try:
        preview_rows = 3
        if display_name_row and display_name_row > preview_rows:
            preview_rows = display_name_row
        raw_preview = pd.read_csv(
            uploaded_file, header=None, nrows=preview_rows
        )
        if raw_preview.empty:
            logs.append("CSV 内容为空，跳过。")
            return logs, display_name_updates
        chosen_row, decision_log = _resolve_display_name_row(
            raw_preview, display_name_row
        )
        logs.append(decision_log)
        display_map = {}
        if chosen_row:
            display_map = _extract_display_name_map(raw_preview, chosen_row)
            if display_map:
                display_name_updates[table_name] = display_map
            logs.append(f"DisplayName 提取：{len(display_map)} 个。")
        else:
            logs.append("DisplayName 提取：未启用。")

        uploaded_file.seek(0)
        skiprows: List[int] = []
        if chosen_row and chosen_row > 1:
            skiprows.append(chosen_row - 1)
        df = pd.read_csv(uploaded_file, header=0, skiprows=skiprows)
    except Exception as exc:
        raise FileFormatError("文件格式不正确") from exc

    _check_columns(uploaded_file.name, None, df.columns)

    try:
        df.to_sql(
            name=table_name,
            con=engine,
            if_exists="replace",
            index=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"写入表失败: {table_name}，原因: {exc}"
        ) from exc

    logs.append(
        f"写入表: {table_name}，行数: {len(df)}，列数: {len(df.columns)}"
    )

    return logs, display_name_updates
