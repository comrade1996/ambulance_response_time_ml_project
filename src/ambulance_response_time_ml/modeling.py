from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

@dataclass(frozen=True)
class ModelRun:
    name: str
    pipeline: Pipeline
    predictions: np.ndarray
    metrics: dict[str, float]


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", drop="first", sparse=False)


def make_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", make_one_hot_encoder(), categorical_features),
        ],
        remainder="drop",
    )


def make_models(
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int,
) -> dict[str, Pipeline]:
    linear = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(numeric_features, categorical_features)),
            ("model", LinearRegression()),
        ]
    )
    random_forest = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(numeric_features, categorical_features)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=12,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return {
        "Linear Regression": linear,
        "Random Forest Regressor": random_forest,
    }


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[list[ModelRun], pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    if len(X) < 10:
        raise ValueError("At least 10 valid rows are required to train and evaluate models.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    models = make_models(numeric_features, categorical_features, random_state)
    runs: list[ModelRun] = []

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                module="sklearn.linear_model._base",
            )
            warnings.filterwarnings(
                "ignore",
                message="Found unknown categories.*",
                category=UserWarning,
                module="sklearn.preprocessing._encoders",
            )
            predictions = pipeline.predict(X_test)
        runs.append(
            ModelRun(
                name=name,
                pipeline=pipeline,
                predictions=predictions,
                metrics=calculate_metrics(y_test, predictions),
            )
        )

    return runs, X_train, y_train, X_test, y_test


def comparison_table(runs: list[ModelRun]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Algorithm": run.name,
                "MAE": run.metrics["MAE"],
                "RMSE": run.metrics["RMSE"],
            }
            for run in runs
        ]
    ).sort_values(["MAE", "RMSE"], ascending=[True, True])


def choose_best_model(runs: list[ModelRun]) -> ModelRun:
    return sorted(
        runs,
        key=lambda run: (run.metrics["MAE"], run.metrics["RMSE"]),
    )[0]


def transformed_feature_names(pipeline: Pipeline) -> list[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    names = preprocessor.get_feature_names_out()
    return [
        name.replace("numeric__", "").replace("categorical__", "")
        for name in names
    ]


def random_forest_feature_importance(run: ModelRun, top_n: int = 20) -> pd.DataFrame:
    model = run.pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["Feature", "Importance"])

    features = transformed_feature_names(run.pipeline)
    importances = pd.DataFrame(
        {
            "Feature": features,
            "Importance": model.feature_importances_,
        }
    )
    return importances.sort_values("Importance", ascending=False).head(top_n)


def save_model_artifacts(runs: list[ModelRun], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        joblib.dump(run.pipeline, output_dir / f"{slugify(run.name)}.joblib")


def slugify(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_")
