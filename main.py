from collections import Counter
from pathlib import Path
from typing import List, Set

import pandas as pd

from settings import DATA_DIR, get_engine

# ========================
# 可配置业务参数
# ========================

# 需要「跳过处理」的 Sheet 名称列表
# 这些 Sheet 不会写入数据库，也不会参与表名唯一性校验
SKIP_SHEETS: List[str] = ["Event workflow", "Database Structure", "Cover Page"]

# 需要「额外跳过」的 Excel 行号列表（对所有未跳过的 Sheet 生效）
# 说明：
# - 行号从 1 开始计数；
# - 第 1 行通常是表头（header），已经作为列名被 pandas 使用；
# - SKIP_ROW_NUMBER 只作用于「数据行」部分：
#     * 比如 SKIP_ROW_NUMBER = [2]：
#         - Excel 第 1 行：表头（不会出现在 DataFrame 里）
#         - Excel 第 2 行：首行数据，对应 DataFrame 的 index=0，将被丢弃
SKIP_ROW_NUMBER: List[int] = [2]


def find_excel_files(data_dir: str) -> list:
    """
    在指定目录下查找所有 Excel 文件（仅 .xlsx）。

    参数：
    - data_dir: 数据目录的路径，通常来自 settings.DATA_DIR。

    返回：
    - 该目录下所有 .xlsx 文件的绝对路径列表（按文件名排序）。

    异常：
    - 如果目录不存在，将抛出 FileNotFoundError。
    """
    base_path = Path(data_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"数据目录不存在: {base_path}")

    excel_files = sorted(base_path.glob("*.xlsx"))
    return [str(p) for p in excel_files]


def _check_columns(file_path: str, sheet_name: str, columns: pd.Index) -> None:
    """
    对列名进行「严格业务校验」，保证与 Excel 中的一致性和可用性。

    规则：
    1. 不允许有重复列名：
       - 同一个 Sheet 内，如果出现完全相同的列名，
         会导致写入数据库时语义不清晰，因此直接报错；
    2. 不允许列名缺失：
       - 列名为 None / NaN / 空字符串或仅空白，都视为缺失。

    一旦发现问题，立即抛出异常并说明具体文件 / Sheet 和冲突原因。
    """
    # 检查重复列名
    counter = Counter(columns)
    duplicated = [col for col, count in counter.items() if count > 1]
    if duplicated:
        duplicate_list = ", ".join(repr(col) for col in duplicated)
        raise ValueError(
            f"列名冲突: 文件 '{file_path}' 的 Sheet '{sheet_name}' 中存在重复列名: {duplicate_list}"
        )

    # 检查缺失列名（None / NaN / 仅空白 / pandas 自动生成的 Unnamed 列名）
    missing_cols = []
    for col in columns:
        if col is None:
            missing_cols.append(col)
        else:
            # pandas 可能将缺失列名表示为 NaN
            try:
                is_nan = pd.isna(col)
            except TypeError:
                is_nan = False

            if is_nan:
                missing_cols.append(col)
            elif isinstance(col, str):
                stripped = col.strip()
                # 1) 纯空白列名
                if stripped == "":
                    missing_cols.append(col)
                # 2) pandas 对空表头默认命名为 "Unnamed: 0", "Unnamed: 1" 等
                #    业务上同样视为「缺失列名」，需要用户修正 Excel。
                elif stripped.lower().startswith("unnamed:"):
                    missing_cols.append(col)

    if missing_cols:
        raise ValueError(
            f"列名缺失: 文件 '{file_path}' 的 Sheet '{sheet_name}' 中存在缺失或空列名，请检查 Excel 表头。"
        )


def _apply_skip_rows(df: pd.DataFrame, skip_rows: List[int]) -> pd.DataFrame:
    """
    根据 SKIP_ROW_NUMBER 中配置的「Excel 行号」跳过指定数据行。

    约定与推导：
    - 我们使用 read_excel(..., header=0) 读取数据：
        * Excel 第 1 行：作为列名（header），不会出现在 DataFrame 中；
        * Excel 第 2 行：首行数据，对应 DataFrame index = 0；
        * Excel 第 n 行：对应 DataFrame index = n - 2（n >= 2）。
    - 因此：
        * 如果想跳过 Excel 的第 2 行，只需删除 DataFrame 的 index=0。

    参数：
    - df: 当前 Sheet 对应的 DataFrame（已包含列名）；
    - skip_rows: 需要跳过的 Excel 行号列表（从 1 开始，包含表头行）。

    返回：
    - 删除指定数据行后的 DataFrame（索引已重置）。
    """
    if not skip_rows or df.empty:
        return df

    indices_to_drop = set()
    for row_no in skip_rows:
        if row_no <= 1:
            # 第 1 行是表头，已作为 columns，不在 DataFrame 中
            continue
        idx = row_no - 2
        if 0 <= idx < len(df):
            indices_to_drop.add(idx)

    if not indices_to_drop:
        return df

    df = df.drop(df.index[sorted(indices_to_drop)]).reset_index(drop=True)
    print(f"    已跳过 Excel 行号: {sorted(indices_to_drop)} (转换后 DataFrame 行索引)")
    return df


def process_excel_file(file_path: str, engine, processed_table_names: Set[str]) -> None:
    """
    处理「单个 Excel 文件」，将该文件中的每个 Sheet 写入 MySQL。

    参数：
    - file_path: Excel 文件的完整路径；
    - engine: SQLAlchemy 的数据库连接 Engine；
    - processed_table_names:
        * 本次脚本运行过程中「已经写入过的表名集合」；
        * 用于做强校验，保证所有 Sheet 生成的表名在本次运行中全局唯一。

    业务规则：
    - 表名：直接使用 Sheet 名称（不清洗，不改名）；
    - 如果不同文件 / 不同 Sheet 生成了同名表：
        * 立即抛出 RuntimeError，并说明是哪个文件、哪个 Sheet、哪个表名冲突；
    - 数据写入模式：
        * 使用 if_exists='replace'，保证幂等性；
        * 即：每次运行会重建表结构并覆盖旧数据。
    """
    print(f"\n开始处理文件: {file_path}")

    with pd.ExcelFile(file_path) as xls:
        for sheet_name in xls.sheet_names:
            # 如果 Sheet 在黑名单中，则跳过
            if sheet_name in SKIP_SHEETS:
                print(f"  -> 跳过 Sheet（配置忽略）: {sheet_name}")
                continue

            print(f"  -> 处理 Sheet: {sheet_name}")

            # 表名直接使用 Sheet 名称
            table_name = sheet_name

            # 全局唯一性检查（跨所有文件和所有 Sheet）
            if table_name in processed_table_names:
                raise RuntimeError(
                    f"表名冲突: 文件 '{file_path}' 的 Sheet '{sheet_name}' 生成的表名 '{table_name}' "
                    f"已在本次运行中使用，请检查不同文件或 Sheet 的命名是否重复。"
                )
            processed_table_names.add(table_name)

            # 先读取原始 Sheet 数据（不指定表头），用于统计原始非空行数
            raw_df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
            raw_non_empty_rows = len(raw_df.dropna(how="all"))

            # 再按业务方式读取：第一行作为列名
            df = pd.read_excel(xls, sheet_name=sheet_name, header=0)
            before_skip_rows = len(df)

            print(
                f"    原始非空行数(含表头): {raw_non_empty_rows}，"
                f"pandas 读取后数据行数: {before_skip_rows}"
            )
            if df.empty:
                print("    Sheet 内容为空，跳过。")
                continue

            # 列名业务校验（不清洗，但强校验重复/缺失）
            _check_columns(file_path, sheet_name, df.columns)

            # 跳过配置的 Excel 行（例如第 2 行）
            df = _apply_skip_rows(df, SKIP_ROW_NUMBER)

            print(f"    跳过配置行后最终写入行数: {len(df)}")

            print(
                f"    写入数据库表: {table_name}，"
                f"行数: {len(df)}，列数: {len(df.columns)}"
            )

            # 写入 MySQL，若表已存在则替换
            df.to_sql(
                name=table_name,
                con=engine,
                if_exists="replace",
                index=False,
            )

            print("    写入完成。")


def main() -> None:
    """
    脚本主入口：
    1. 获取数据库连接（内部会自动建库）；
    2. 扫描 data 目录下所有 .xlsx 文件；
    3. 依次处理每个文件中的所有 Sheet：
       - 做表名全局唯一性检查；
       - 对列名做严格业务校验（不清洗，但检查重复 / 缺失）；
       - 按 SKIP_ROW_NUMBER 跳过指定数据行；
       - 使用 pandas.to_sql 写入 MySQL。

    运行结果：
    - 若中途没有异常：所有 Excel 内容写入数据库，对应表名=Sheet 名；
    - 若发现任何表名 / 列名问题：立即抛错并停止运行。
    """
    # 获取数据库连接（会自动建库）
    engine = get_engine()

    # 全局：本次运行中已使用的表名集合，用于强校验
    processed_table_names: Set[str] = set()

    # 查找 data 目录中的所有 Excel 文件
    excel_files = find_excel_files(DATA_DIR)
    if not excel_files:
        print(f"未在目录 {DATA_DIR} 中找到任何 .xlsx 文件。")
        return

    print(f"将在目录 {DATA_DIR} 中处理 {len(excel_files)} 个 Excel 文件。")
    for file_path in excel_files:
        process_excel_file(file_path, engine, processed_table_names)

    print("\n全部 Excel 文件处理完成，没有发现表名或列名冲突。")


if __name__ == "__main__":
    main()
