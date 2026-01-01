"""Tests for analysis service helpers."""

import contextlib

import pandas as pd

from analysis.services.analysis_service import (
    apply_baseline_mapping,
    apply_calculations,
    calculate_anova_table,
)


def test_apply_baseline_mapping_adds_baseline_column() -> None:
    """Ensure baseline mapping adds the baseline column per subject."""
    df = pd.DataFrame(
        {
            "SUBJ": ["01", "01", "02", "02"],
            "VISIT": ["Baseline", "Week1", "Baseline", "Week1"],
            "VAL": [10, 12, 20, 25],
        }
    )
    config = {
        "subj_col": "SUBJ",
        "visit_col": "VISIT",
        "baseline_val": "Baseline",
        "target_cols": ["VAL"],
    }
    result = apply_baseline_mapping(df, config)
    assert "VAL_BL" in result.columns
    subj01 = result[result["SUBJ"] == "01"]["VAL_BL"].unique().tolist()
    subj02 = result[result["SUBJ"] == "02"]["VAL_BL"].unique().tolist()
    assert subj01 == [10]
    assert subj02 == [20]


def test_apply_calculations_sum_rule() -> None:
    """Ensure calculation rules apply when columns are present."""
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    rules = [{"name": "TOTAL", "cols": ["A", "B"], "method": "Sum - 求和"}]
    result = apply_calculations(df, rules)
    assert result["TOTAL"].tolist() == [4, 6]


def test_apply_calculations_skips_missing_columns() -> None:
    """Ensure calculations are skipped when columns are missing."""
    df = pd.DataFrame({"A": [1, 2]})
    rules = [{"name": "TOTAL", "cols": ["A", "B"], "method": "Sum - 求和"}]
    result = apply_calculations(df, rules)
    assert "TOTAL" not in result.columns


def test_calculate_anova_table_returns_rows() -> None:
    """Ensure ANOVA table returns rows with expected columns."""
    df = pd.DataFrame(
        {
            "Visit": ["V1", "V1", "V1", "V1"],
            "Group": ["A", "A", "B", "B"],
            "Value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    result = calculate_anova_table(df, "Visit", "Group", "Value")
    assert not result.empty
    assert set(result.columns) == {"Layer", "F-value", "P-value", "Note"}


def test_calculate_anova_table_insufficient_groups() -> None:
    """Ensure ANOVA returns a note when groups are insufficient."""
    df = pd.DataFrame(
        {"Visit": ["V1", "V1"], "Group": ["A", "A"], "Value": [1.0, 2.0]}
    )
    result = calculate_anova_table(df, "Visit", "Group", "Value")
    assert result.loc[0, "Note"] == "组数不足(<2)"


def test_run_analysis_returns_empty_when_sql_missing(monkeypatch) -> None:
    """Ensure run_analysis returns empty results when SQL cannot be built."""
    import analysis.services.analysis_service as svc

    monkeypatch.setattr(svc, "load_table_metadata", lambda: {"t": []})
    monkeypatch.setattr(svc, "build_sql", lambda **kwargs: None)

    captured = {}

    def fake_error(message: str) -> None:
        """Capture error messages emitted by Streamlit."""
        captured["message"] = message

    monkeypatch.setattr(svc.st, "error", fake_error)

    sql, df = svc.run_analysis({"selected_tables": ["t"]})
    assert sql == ""
    assert df.empty
    assert "配置错误" in captured.get("message", "")


def test_run_analysis_executes_query(monkeypatch) -> None:
    """Ensure run_analysis executes a query and returns data."""
    import analysis.services.analysis_service as svc

    captured = {}

    def fake_build_sql(**kwargs):
        """Capture SQL builder arguments and return a stub SQL."""
        captured.update(kwargs)
        return "SELECT 1"

    monkeypatch.setattr(svc, "load_table_metadata", lambda: {"t": ["id"]})
    monkeypatch.setattr(svc, "build_sql", fake_build_sql)

    class DummyConn:
        def execution_options(self, **kwargs):
            """Return self to mimic SQLAlchemy chaining."""
            return self

        def __enter__(self):
            """Enter context manager for the dummy connection."""
            return self

        def __exit__(self, exc_type, exc, tb):
            """Exit context manager for the dummy connection."""
            return False

    class DummyEngine:
        def connect(self):
            """Return a dummy connection instance."""
            return DummyConn()

    monkeypatch.setattr(svc, "get_business_engine", lambda: DummyEngine())

    def fake_read_sql(sql, conn):
        """Return a minimal DataFrame for the fake SQL execution."""
        assert sql == "SELECT 1"
        assert isinstance(conn, DummyConn)
        return pd.DataFrame({"x": [1]})

    monkeypatch.setattr(svc.pd, "read_sql", fake_read_sql)
    monkeypatch.setattr(
        svc.st, "spinner", lambda *args, **kwargs: contextlib.nullcontext()
    )

    sql, df = svc.run_analysis(
        {
            "selected_tables": ["t"],
            "table_columns_map": {"t": ["id"]},
            "filters": {"conditions": []},
        }
    )

    assert sql == "SELECT 1"
    assert df["x"].tolist() == [1]
    assert captured["selected_tables"] == ["t"]
