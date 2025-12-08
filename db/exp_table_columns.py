import json
from pathlib import Path
from typing import Dict, List

from sqlalchemy import inspect

from settings import get_engine


def export_table_columns(output_path: Path | None = None) -> Path:
    """
    导出当前数据库中所有表的列信息到 JSON 文件。

    JSON 结构（方案 B）：
    {
      "table_name_1": ["col1", "col2", ...],
      "table_name_2": ["colA", "colB", ...],
      ...
    }

    参数：
    - output_path: 输出文件路径，如果为 None，则默认写到 db/table_columns.json。

    返回：
    - 实际写入的 JSON 文件路径。
    """
    if output_path is None:
        # 默认输出到当前 db 目录下
        output_path = Path(__file__).resolve().parent / "table_columns.json"

    engine = get_engine()
    inspector = inspect(engine)

    try:
        # 获取当前数据库下的所有表名，并按字典序排序，保证结果稳定
        table_names: List[str] = sorted(inspector.get_table_names())

        tables: Dict[str, List[str]] = {}
        for table_name in table_names:
            # 获取表的所有列信息，保持列顺序（一般为物理顺序）
            columns_info = inspector.get_columns(table_name)
            column_names = [col["name"] for col in columns_info]
            tables[table_name] = column_names

        # 确保目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 写入 JSON 文件
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(tables, f, ensure_ascii=False, indent=2)

        print(f"[exp_table_columns] 已导出 {len(tables)} 张表的列信息到: {output_path}")
        return output_path
    finally:
        engine.dispose()


if __name__ == "__main__":
    export_table_columns()
