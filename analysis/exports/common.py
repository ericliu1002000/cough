"""Common export helpers."""

from __future__ import annotations

import pandas as pd


def df_to_csv_bytes(df: pd.DataFrame, index: bool = False) -> bytes:
    """Return CSV bytes encoded as UTF-8 with BOM for Excel compatibility."""
    return df.to_csv(index=index).encode("utf-8-sig")
