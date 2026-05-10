from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import (
    BASE_CATEGORICAL_FEATURES,
    CLASSIFICATION_SOURCE_COLUMNS,
    FIRST_ON_SCENE_DATETIME,
    INCIDENT_DATETIME,
    NUMERIC_FEATURES,
    OPTIONAL_FEATURE_COLUMNS,
    REQUIRED_COLUMNS,
    RESPONSE_TIME_MAX,
    RESPONSE_TIME_MIN,
    TARGET,
)


@dataclass(frozen=True)
class DataSummary:
    rows: int
    columns: int
    duplicate_rows: int
    missing_values: dict[str, int]
    numeric_columns: list[str]
    categorical_columns: list[str]
    selected_features: list[str]


def load_csv(path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path, nrows=max_rows, low_memory=False)


def load_csv_for_training(
    path: str | Path,
    max_rows: int | None = None,
    sample_rows: int | None = None,
    chunksize: int = 50_000,
) -> pd.DataFrame:
    if sample_rows is None:
        return load_csv(path, max_rows=max_rows)

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    chunks: list[pd.DataFrame] = []
    total = 0
    for chunk in pd.read_csv(path, chunksize=chunksize, nrows=max_rows, low_memory=False):
        clean_chunk = clean_and_engineer_features(chunk)
        if clean_chunk.empty:
            continue
        chunks.append(clean_chunk)
        total += len(clean_chunk)
        if total >= sample_rows:
            break

    if not chunks:
        raise ValueError("No valid training rows were found in the CSV.")

    return pd.concat(chunks, ignore_index=True).head(sample_rows)


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [
        str(column).strip().replace(" ", "_").replace("-", "_").upper()
        for column in cleaned.columns
    ]
    return cleaned


def validate_required_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if not any(column in df.columns for column in CLASSIFICATION_SOURCE_COLUMNS):
        missing.append("INCIDENT_CLASSIFICATION or FINAL_CALL_TYPE or INITIAL_CALL_TYPE")
    if missing:
        available = ", ".join(df.columns)
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
            + f". Available columns: {available}"
        )


def summarize_data(df: pd.DataFrame, selected_features: list[str]) -> DataSummary:
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = df.select_dtypes(exclude="number").columns.tolist()
    return DataSummary(
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
        duplicate_rows=int(df.duplicated().sum()),
        missing_values={k: int(v) for k, v in df.isna().sum().to_dict().items()},
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        selected_features=selected_features,
    )


def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = standardize_column_names(df)
    cleaned = add_compatible_columns(cleaned)
    validate_required_columns(cleaned)

    cleaned[INCIDENT_DATETIME] = parse_datetime_column(cleaned[INCIDENT_DATETIME])
    cleaned[FIRST_ON_SCENE_DATETIME] = parse_datetime_column(
        cleaned[FIRST_ON_SCENE_DATETIME]
    )

    cleaned[TARGET] = calculate_response_time_minutes(cleaned)

    required_for_model = REQUIRED_COLUMNS + [TARGET]
    cleaned = cleaned.dropna(subset=required_for_model)
    cleaned = cleaned[
        (cleaned[TARGET] >= RESPONSE_TIME_MIN) & (cleaned[TARGET] <= RESPONSE_TIME_MAX)
    ].copy()

    cleaned["Hour"] = cleaned[INCIDENT_DATETIME].dt.hour
    cleaned["DayOfWeek"] = cleaned[INCIDENT_DATETIME].dt.dayofweek
    cleaned["Month"] = cleaned[INCIDENT_DATETIME].dt.month

    for column in NUMERIC_FEATURES:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    for column in BASE_CATEGORICAL_FEATURES + OPTIONAL_FEATURE_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].astype(str).str.strip()

    return cleaned


def get_model_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [column for column in NUMERIC_FEATURES if column in df.columns]
    categorical = [column for column in BASE_CATEGORICAL_FEATURES if column in df.columns]
    categorical.extend([column for column in OPTIONAL_FEATURE_COLUMNS if column in df.columns])
    return numeric, categorical


def make_model_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    numeric_features, categorical_features = get_model_features(df)
    features = numeric_features + categorical_features
    data = df[features + [TARGET]].dropna().copy()
    X = data[features]
    y = data[TARGET]
    return X, y, numeric_features, categorical_features


def add_compatible_columns(df: pd.DataFrame) -> pd.DataFrame:
    compatible = df.copy()
    if "INCIDENT_CLASSIFICATION" not in compatible.columns:
        for source in CLASSIFICATION_SOURCE_COLUMNS:
            if source in compatible.columns:
                compatible["INCIDENT_CLASSIFICATION"] = compatible[source]
                break
    return compatible


def calculate_response_time_minutes(df: pd.DataFrame) -> pd.Series:
    timestamp_minutes = (
        df[FIRST_ON_SCENE_DATETIME] - df[INCIDENT_DATETIME]
    ).dt.total_seconds() / 60

    if "INCIDENT_RESPONSE_SECONDS_QY" not in df.columns:
        return timestamp_minutes

    response_seconds = pd.to_numeric(df["INCIDENT_RESPONSE_SECONDS_QY"], errors="coerce")
    response_minutes = response_seconds / 60
    if "VALID_INCIDENT_RSPNS_TIME_INDC" in df.columns:
        valid = df["VALID_INCIDENT_RSPNS_TIME_INDC"].astype(str).str.upper().eq("Y")
        response_minutes = response_minutes.where(valid)

    return response_minutes.fillna(timestamp_minutes)


def parse_datetime_column(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    if parsed.notna().sum() == 0:
        parsed = pd.to_datetime(series, errors="coerce")
    return parsed
