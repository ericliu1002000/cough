"""业务数据上传服务。"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import warnings


class FileFormatError(ValueError):
    """用于标记文件格式解析失败的异常。"""


def _contains_sheet_keyword(name: str) -> bool:
    return "sheet" in name.casefold()


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


def _apply_skip_rows(df: pd.DataFrame, skip_rows: List[int]) -> pd.DataFrame:
    if not skip_rows or df.empty:
        return df

    indices_to_drop = set()
    for row_no in skip_rows:
        if row_no <= 1:
            continue
        idx = row_no - 2
        if 0 <= idx < len(df):
            indices_to_drop.add(idx)

    if not indices_to_drop:
        return df

    df = df.drop(df.index[sorted(indices_to_drop)]).reset_index(drop=True)
    return df


def upload_excel(
    uploaded_file,
    engine,
    skip_sheets: List[str],
    skip_rows: List[int],
) -> List[str]:
    logs: List[str] = []
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
                raw_df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
                raw_non_empty_rows = len(raw_df.dropna(how="all"))
                df = pd.read_excel(xls, sheet_name=sheet_name, header=0)
            except Exception as exc:
                raise FileFormatError("文件格式不正确") from exc

            logs.append(
                f"原始非空行数(含表头): {raw_non_empty_rows}，"
                f"读取后数据行数: {len(df)}"
            )
            if df.empty:
                logs.append("Sheet 内容为空，跳过。")
                continue

            _check_columns(uploaded_file.name, sheet_name, df.columns)
            df = _apply_skip_rows(df, skip_rows)
            logs.append(f"跳过配置行后写入行数: {len(df)}")

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

    return logs


def upload_csv(
    uploaded_file,
    engine,
    skip_rows: List[int],
) -> List[str]:
    logs: List[str] = []
    uploaded_file.seek(0)
    table_name = Path(uploaded_file.name).stem
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        raise FileFormatError("文件格式不正确") from exc

    _check_columns(uploaded_file.name, None, df.columns)
    df = _apply_skip_rows(df, skip_rows)

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

    return logs
