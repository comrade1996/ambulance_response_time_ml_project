from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

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
    models = {
        "Linear Regression": linear,
        "Random Forest Regressor": random_forest,
    }
    if HAS_XGBOOST:
        xgb = Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(numeric_features, categorical_features)),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=300,
                        max_depth=8,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=random_state,
                        n_jobs=-1,
                        verbosity=0,
                    ),
                ),
            ]
        )
        models["XGBoost"] = xgb
    if HAS_LIGHTGBM:
        lgbm = Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(numeric_features, categorical_features)),
                (
                    "model",
                    LGBMRegressor(
                        n_estimators=300,
                        max_depth=8,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=random_state,
                        n_jobs=-1,
                        verbose=-1,
                    ),
                ),
            ]
        )
        models["LightGBM"] = lgbm
    if HAS_XGBOOST and HAS_LIGHTGBM:
        estimators = [
            ("rf", RandomForestRegressor(n_estimators=100, max_depth=12, random_state=random_state, n_jobs=-1)),
            ("xgb", XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.05, random_state=random_state, n_jobs=-1, verbosity=0)),
            ("lgbm", LGBMRegressor(n_estimators=300, max_depth=8, learning_rate=0.05, random_state=random_state, n_jobs=-1, verbose=-1)),
        ]
        stacking = Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(numeric_features, categorical_features)),
                (
                    "model",
                    StackingRegressor(
                        estimators=estimators,
                        final_estimator=Ridge(alpha=1.0),
                        cv=3,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        models["Stacking Ensemble"] = stacking
    return models


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


def temporal_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    datetime_series: pd.Series,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    sorted_idx = datetime_series.sort_values().index
    X_sorted = X.loc[sorted_idx]
    y_sorted = y.loc[sorted_idx]
    split_point = int(len(X_sorted) * (1 - test_size))
    return (
        X_sorted.iloc[:split_point],
        X_sorted.iloc[split_point:],
        y_sorted.iloc[:split_point],
        y_sorted.iloc[split_point:],
    )


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int = 42,
    n_iter: int = 20,
) -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(numeric_features, categorical_features)),
            ("model", RandomForestRegressor(random_state=random_state, n_jobs=-1)),
        ]
    )
    param_distributions = {
        "model__n_estimators": [100, 200, 300, 500],
        "model__max_depth": [8, 12, 16, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    }
    search = RandomizedSearchCV(
        pipeline,
        param_distributions,
        n_iter=n_iter,
        cv=3,
        scoring="neg_mean_absolute_error",
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int = 42,
    n_iter: int = 20,
) -> Pipeline | None:
    if not HAS_XGBOOST:
        return None
    pipeline = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(numeric_features, categorical_features)),
            ("model", XGBRegressor(random_state=random_state, n_jobs=-1, verbosity=0)),
        ]
    )
    param_distributions = {
        "model__n_estimators": [200, 300, 500, 800],
        "model__max_depth": [4, 6, 8, 10],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 0.9],
        "model__colsample_bytree": [0.7, 0.8, 0.9],
    }
    search = RandomizedSearchCV(
        pipeline,
        param_distributions,
        n_iter=n_iter,
        cv=3,
        scoring="neg_mean_absolute_error",
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_


def residual_analysis(
    y_true: pd.Series, y_pred: np.ndarray, X: pd.DataFrame
) -> pd.DataFrame:
    residuals = y_true.values - y_pred
    analysis = X.copy()
    analysis["Actual"] = y_true.values
    analysis["Predicted"] = y_pred
    analysis["Residual"] = residuals
    analysis["Abs_Error"] = np.abs(residuals)
    return analysis
