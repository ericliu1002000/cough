"""Tests for SQL builder utilities."""

from analysis.repositories.sql_builder import build_sql, format_value_for_sql


def test_format_value_for_sql_in_list() -> None:
    """Ensure list values are formatted as SQL IN tuples."""
    value = ["alpha", "beta"]
    assert format_value_for_sql(value, "IN") == "('alpha', 'beta')"


def test_build_sql_basic() -> None:
    """Ensure SQL builder emits expected clauses."""
    meta_data = {"adsl": ["SUBJID", "AGE", "GROUP"]}
    sql = build_sql(
        selected_tables=["adsl"],
        table_columns_map={"adsl": ["AGE", "GROUP"]},
        filters={
            "conditions": [
                {"table": "adsl", "col": "AGE", "op": ">", "val": 30}
            ]
        },
        subject_blocklist="101\n102",
        meta_data=meta_data,
    )

    assert sql is not None
    assert "FROM `adsl`" in sql
    assert "`adsl`.`SUBJID` AS `SUBJECTID`" in sql
    assert "`adsl`.`AGE` AS `adsl_AGE`" in sql
    assert "NOT IN ('101', '102')" in sql
    assert "AGE` > 30" in sql
    assert "LIMIT 1000" in sql
