from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ambulance_mpl_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/ambulance_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import TARGET
from .data import DataSummary
from .modeling import ModelRun


def ensure_output_dirs(output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    paths = {
        "root": output_dir,
        "figures": output_dir / "figures",
        "models": output_dir / "models",
        "reports": output_dir / "reports",
        "processed": Path("data") / "processed",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_data_summary(summary: DataSummary, reports_dir: str | Path) -> None:
    reports_dir = Path(reports_dir)
    payload = asdict(summary)
    (reports_dir / "data_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def save_cleaned_data(df: pd.DataFrame, processed_dir: str | Path) -> Path:
    processed_dir = Path(processed_dir)
    path = processed_dir / "cleaned_ems_dispatch_data.csv"
    df.to_csv(path, index=False)
    return path


def create_eda_plots(df: pd.DataFrame, figures_dir: str | Path) -> list[Path]:
    figures_dir = Path(figures_dir)
    outputs: list[Path] = []

    outputs.append(
        _plot_histogram(
            df,
            figures_dir / "response_time_distribution.png",
            title="Distribution of Ambulance Response Time",
            xlabel="Response Time in Minutes",
        )
    )

    if "BOROUGH" in df.columns:
        outputs.append(
            _plot_group_bar(
                df,
                group_column="BOROUGH",
                output_path=figures_dir / "average_response_time_by_borough.png",
                title="Average Response Time by Borough",
                xlabel="Borough",
                ylabel="Average Response Time",
            )
        )

    outputs.append(
        _plot_hour_line(
            df,
            figures_dir / "average_response_time_by_hour.png",
        )
    )

    if "INCIDENT_CLASSIFICATION" in df.columns:
        outputs.append(
            _plot_group_barh(
                df,
                group_column="INCIDENT_CLASSIFICATION",
                output_path=figures_dir
                / "top_incident_classifications_by_response_time.png",
                title="Top Incident Classifications by Average Response Time",
                xlabel="Average Response Time",
                ylabel="Incident Classification",
                top_n=10,
            )
        )

    return outputs


def _plot_histogram(
    df: pd.DataFrame, output_path: Path, title: str, xlabel: str
) -> Path:
    plt.figure(figsize=(9, 5))
    df[TARGET].hist(bins=50)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def _plot_group_bar(
    df: pd.DataFrame,
    group_column: str,
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> Path:
    plt.figure(figsize=(9, 5))
    df.groupby(group_column)[TARGET].mean().sort_values().plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def _plot_hour_line(df: pd.DataFrame, output_path: Path) -> Path:
    plt.figure(figsize=(9, 5))
    df.groupby("Hour")[TARGET].mean().sort_index().plot(kind="line", marker="o")
    plt.title("Average Response Time by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Response Time")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def _plot_group_barh(
    df: pd.DataFrame,
    group_column: str,
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    top_n: int,
) -> Path:
    plt.figure(figsize=(10, 6))
    (
        df.groupby(group_column)[TARGET]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .sort_values()
        .plot(kind="barh")
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def save_feature_importance_plot(
    feature_importance: pd.DataFrame, figures_dir: str | Path
) -> Path | None:
    if feature_importance.empty:
        return None
    figures_dir = Path(figures_dir)
    output_path = figures_dir / "top_factors_affecting_response_time.png"
    plt.figure(figsize=(10, 6))
    feature_importance.sort_values("Importance").plot(
        kind="barh", x="Feature", y="Importance", legend=False
    )
    plt.title("Top Factors Affecting Ambulance Response Time")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def write_final_report(
    reports_dir: str | Path,
    summary: DataSummary,
    comparison: pd.DataFrame,
    feature_importance: pd.DataFrame,
    best_run: ModelRun,
) -> Path:
    reports_dir = Path(reports_dir)
    path = reports_dir / "final_report.md"
    comparison_md = dataframe_to_markdown(comparison)
    feature_md = (
        dataframe_to_markdown(feature_importance)
        if not feature_importance.empty
        else "Feature importance is available only for Random Forest."
    )
    response_mean_text = summary.missing_values.get(TARGET, "N/A")

    content = f"""# Ambulance Response Time Prediction Report

## 1. Project Title

Ambulance Response Time Prediction Using Linear Regression and Random Forest.

## 2. Introduction

This project uses supervised machine learning regression to predict the expected
ambulance response time in minutes. The target variable is `{TARGET}`, calculated
from the difference between incident creation time and first on-scene time.

## 3. Research Problem

Can historical EMS incident dispatch records be used to predict ambulance
response time and identify the most important operational factors behind delay?

## 4. Objectives

- Clean EMS incident dispatch records.
- Calculate response time in minutes.
- Create time-based features: hour, day of week, and month.
- Explore response time by borough, hour, and incident type.
- Train Linear Regression and Random Forest Regressor.
- Compare models with MAE and RMSE.
- Identify the most important factors affecting response time.

## 5. Dataset Summary

- Rows after loading: {summary.rows}
- Columns after loading: {summary.columns}
- Duplicate rows: {summary.duplicate_rows}
- Selected features: {", ".join(summary.selected_features)}
- Missing target values before final model frame, if present: {response_mean_text}

## 6. Data Cleaning and Feature Engineering

The pipeline standardizes column names, converts incident and arrival timestamps,
calculates `{TARGET}`, removes invalid response times below 0 or above 120
minutes, and drops rows missing required model fields.

## 7. Exploratory Data Analysis

The run creates EDA figures in `outputs/figures`:

- Response time distribution.
- Average response time by borough.
- Average response time by hour.
- Top incident classifications by average response time.

## 8. Algorithms

Linear Regression is used as the baseline model. Random Forest Regressor is used
to capture nonlinear relationships and provide feature importance.

## 9. Model Comparison

{comparison_md}

In this run, **{best_run.name}** performed better based on the lowest MAE and
lowest RMSE ordering.

## 10. Feature Importance

{feature_md}

## 11. Discussion

Response time is influenced by a combination of time, location, and incident
classification. A Random Forest model often performs better when the relationship
between these factors and response time is nonlinear.

## 12. Operational Recommendations

- Increase ambulance availability in high-delay boroughs or dispatch areas.
- Review staffing and unit distribution during peak response-time hours.
- Monitor incident categories that show longer average response times.
- Improve timestamp quality in dispatch systems.
- Retrain the model periodically with recent dispatch records.

## 13. Project Limits

- Results depend on the quality of timestamp and incident classification data.
- Traffic, weather, live ambulance availability, and station distance are not
  included unless they exist in the source dataset.
- The model is for analysis and planning support, not emergency dispatch
  automation without operational validation.

## 14. Conclusion

The project demonstrates how machine learning can support civil defense and EMS
planning by predicting response time, comparing algorithms, and identifying the
main factors linked with delays.
"""
    path.write_text(content, encoding="utf-8")
    return path


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    columns = [str(column) for column in df.columns]
    rows = [
        [
            f"{value:.4f}" if isinstance(value, float) else str(value)
            for value in row
        ]
        for row in df.itertuples(index=False, name=None)
    ]
    widths = [
        max(len(column), *(len(row[index]) for row in rows)) if rows else len(column)
        for index, column in enumerate(columns)
    ]
    header = "| " + " | ".join(
        column.ljust(widths[index]) for index, column in enumerate(columns)
    ) + " |"
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [
        "| " + " | ".join(
            row[index].ljust(widths[index]) for index in range(len(columns))
        ) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])
