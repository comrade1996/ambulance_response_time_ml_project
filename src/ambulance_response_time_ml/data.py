from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import numpy as np

from .config import (
    BASE_CATEGORICAL_FEATURES,
    CLASSIFICATION_SOURCE_COLUMNS,
    CONTEXTUAL_FEATURES,
    CYCLICAL_FEATURES,
    ENHANCED_NUMERIC_FEATURES,
    FIRST_ON_SCENE_DATETIME,
    INCIDENT_DATETIME,
    INTERACTION_FEATURES,
    NUMERIC_FEATURES,
    OPTIONAL_FEATURE_COLUMNS,
    REQUIRED_COLUMNS,
    RESPONSE_TIME_MAX,
    RESPONSE_TIME_MIN,
    TARGET,
    TARGET_ENCODED_FEATURES,
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

    if "DISPATCH_RESPONSE_SECONDS_QY" in cleaned.columns:
        cleaned["DISPATCH_RESPONSE_SECONDS_QY"] = pd.to_numeric(
            cleaned["DISPATCH_RESPONSE_SECONDS_QY"], errors="coerce"
        ).fillna(0).clip(lower=0, upper=600)

    if "HELD_INDICATOR" in cleaned.columns:
        cleaned["IsHeld"] = (cleaned["HELD_INDICATOR"].astype(str).str.strip().str.upper() == "Y").astype(int)
    else:
        cleaned["IsHeld"] = 0

    if "REOPEN_INDICATOR" in cleaned.columns:
        cleaned["IsReopen"] = (cleaned["REOPEN_INDICATOR"].astype(str).str.strip().str.upper() == "Y").astype(int)
    else:
        cleaned["IsReopen"] = 0

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


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    enhanced = df.copy()
    if "Hour" in enhanced.columns:
        enhanced["Hour_sin"] = np.sin(2 * np.pi * enhanced["Hour"] / 24)
        enhanced["Hour_cos"] = np.cos(2 * np.pi * enhanced["Hour"] / 24)
    if "DayOfWeek" in enhanced.columns:
        enhanced["DayOfWeek_sin"] = np.sin(2 * np.pi * enhanced["DayOfWeek"] / 7)
        enhanced["DayOfWeek_cos"] = np.cos(2 * np.pi * enhanced["DayOfWeek"] / 7)
    if "Month" in enhanced.columns:
        enhanced["Month_sin"] = np.sin(2 * np.pi * (enhanced["Month"] - 1) / 12)
        enhanced["Month_cos"] = np.cos(2 * np.pi * (enhanced["Month"] - 1) / 12)
    return enhanced


def add_contextual_features(df: pd.DataFrame) -> pd.DataFrame:
    enhanced = df.copy()
    if "DayOfWeek" in enhanced.columns:
        enhanced["IsWeekend"] = (enhanced["DayOfWeek"] >= 5).astype(int)
    if "Hour" in enhanced.columns:
        enhanced["IsNight"] = ((enhanced["Hour"] >= 22) | (enhanced["Hour"] <= 5)).astype(int)
        enhanced["IsRushHour"] = enhanced["Hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    return enhanced


def add_target_encoding(df: pd.DataFrame, smoothing: int = 20) -> pd.DataFrame:
    enhanced = df.copy()
    global_mean = enhanced[TARGET].mean() if TARGET in enhanced.columns else 0.0
    if TARGET in enhanced.columns and "INCIDENT_CLASSIFICATION" in enhanced.columns:
        stats = enhanced.groupby("INCIDENT_CLASSIFICATION")[TARGET].agg(["mean", "count"])
        smooth = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
        enhanced["Classification_TargetEnc"] = enhanced["INCIDENT_CLASSIFICATION"].map(smooth).fillna(global_mean)
    if TARGET in enhanced.columns and "INCIDENT_DISPATCH_AREA" in enhanced.columns:
        stats = enhanced.groupby("INCIDENT_DISPATCH_AREA")[TARGET].agg(["mean", "count"])
        smooth = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
        enhanced["DispatchArea_TargetEnc"] = enhanced["INCIDENT_DISPATCH_AREA"].map(smooth).fillna(global_mean)
    if TARGET in enhanced.columns and "BOROUGH" in enhanced.columns:
        stats = enhanced.groupby("BOROUGH")[TARGET].agg(["mean", "count"])
        smooth = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
        enhanced["Borough_TargetEnc"] = enhanced["BOROUGH"].map(smooth).fillna(global_mean)
    return enhanced


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    enhanced = df.copy()
    if TARGET in enhanced.columns and "BOROUGH" in enhanced.columns and "Hour" in enhanced.columns:
        borough_hour_avg = enhanced.groupby(["BOROUGH", "Hour"])[TARGET].transform("mean")
        enhanced["Borough_Hour_Avg"] = borough_hour_avg
    if TARGET in enhanced.columns and "INCIDENT_CLASSIFICATION" in enhanced.columns and "INITIAL_SEVERITY_LEVEL_CODE" in enhanced.columns:
        cls_sev_avg = enhanced.groupby(["INCIDENT_CLASSIFICATION", "INITIAL_SEVERITY_LEVEL_CODE"])[TARGET].transform("mean")
        enhanced["Classification_Severity_Avg"] = cls_sev_avg
    if TARGET in enhanced.columns and "INCIDENT_DISPATCH_AREA" in enhanced.columns and "Hour" in enhanced.columns:
        dispatch_hour_avg = enhanced.groupby(["INCIDENT_DISPATCH_AREA", "Hour"])[TARGET].transform("mean")
        enhanced["DispatchArea_Hour_Avg"] = dispatch_hour_avg
    return enhanced


def make_enhanced_model_frame(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    enhanced = add_cyclical_features(df)
    enhanced = add_contextual_features(enhanced)
    enhanced = add_target_encoding(enhanced)
    enhanced = add_interaction_features(enhanced)
    numeric_features = [col for col in ENHANCED_NUMERIC_FEATURES if col in enhanced.columns]
    categorical_features = [col for col in BASE_CATEGORICAL_FEATURES if col in enhanced.columns]
    categorical_features.extend([col for col in OPTIONAL_FEATURE_COLUMNS if col in enhanced.columns])
    features = numeric_features + categorical_features
    data = enhanced[features + [TARGET]].dropna().copy()
    X = data[features]
    y = data[TARGET]
    return X, y, numeric_features, categorical_features


def compute_interaction_lookup(df: pd.DataFrame) -> dict:
    lookup = {}
    global_mean = df[TARGET].mean() if TARGET in df.columns else 0.0
    lookup["_global_mean"] = global_mean
    if TARGET in df.columns and "BOROUGH" in df.columns and "Hour" in df.columns:
        lookup["Borough_Hour_Avg"] = df.groupby(["BOROUGH", "Hour"])[TARGET].mean().to_dict()
    if TARGET in df.columns and "INCIDENT_CLASSIFICATION" in df.columns and "INITIAL_SEVERITY_LEVEL_CODE" in df.columns:
        lookup["Classification_Severity_Avg"] = df.groupby(["INCIDENT_CLASSIFICATION", "INITIAL_SEVERITY_LEVEL_CODE"])[TARGET].mean().to_dict()
    if TARGET in df.columns and "INCIDENT_DISPATCH_AREA" in df.columns and "Hour" in df.columns:
        lookup["DispatchArea_Hour_Avg"] = df.groupby(["INCIDENT_DISPATCH_AREA", "Hour"])[TARGET].mean().to_dict()
    smoothing = 20
    if TARGET in df.columns and "INCIDENT_CLASSIFICATION" in df.columns:
        stats = df.groupby("INCIDENT_CLASSIFICATION")[TARGET].agg(["mean", "count"])
        smooth = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
        lookup["Classification_TargetEnc"] = smooth.to_dict()
    if TARGET in df.columns and "INCIDENT_DISPATCH_AREA" in df.columns:
        stats = df.groupby("INCIDENT_DISPATCH_AREA")[TARGET].agg(["mean", "count"])
        smooth = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
        lookup["DispatchArea_TargetEnc"] = smooth.to_dict()
    if TARGET in df.columns and "BOROUGH" in df.columns:
        stats = df.groupby("BOROUGH")[TARGET].agg(["mean", "count"])
        smooth = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
        lookup["Borough_TargetEnc"] = smooth.to_dict()
    return lookup
