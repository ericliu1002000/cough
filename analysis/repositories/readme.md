# analysis/repositories

用途:
- 数据访问层，负责元数据与 SQL 构建。

文件:
- __init__.py: 包标识。
- metadata_repo.py: 表结构元数据与 ID 列识别。
- setup_repo.py: 兼容层，转发到根目录的 setup_catalog。
- sql_builder.py: SQL 生成与过滤辅助。
- readme.md: 本说明。
