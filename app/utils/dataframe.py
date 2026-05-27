from typing import Any, Dict, List

import pandas as pd


def records_to_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert JSON records to a pandas DataFrame with basic validation."""

    if records is None:
        raise ValueError("No data records were provided.")
    if not isinstance(records, list):
        raise ValueError("The 'data' field must be a list of JSON records.")
    if len(records) == 0:
        return pd.DataFrame()
    if not all(isinstance(item, dict) for item in records):
        raise ValueError("Every item in 'data' must be a JSON object.")
    return pd.DataFrame.from_records(records)
