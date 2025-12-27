# 项目根目录

用途:
- 基于 Streamlit 的实验数据分析演示项目，提供数据提取、透视分析、图表和导出能力。

文件:
- .env: 本地环境变量配置（数据库/路径）。
- .gitignore: Git 忽略规则。
- analysis_setups.py: Streamlit 主页（配置管理入口）。
- main.py: CLI/初始化占位入口。
- pytest.ini: Pytest 配置。
- requirements.txt: Python 依赖列表。
- utils.py: 兼容层，转发到 analysis.*。
- readme.md: 本说明。

子目录:
- analysis/: 核心业务包。
- data/: 本地样例数据。
- docs/: 规划与业务说明。
- logs/: 访问与错误日志。
- pages/: Streamlit 页面。
- tests/: 自动化测试。
- tmp/: 临时脚本与草稿。
