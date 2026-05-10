from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import pandas as pd

from .config import DEFAULT_OUTPUT_DIR
from .data import (
    clean_and_engineer_features,
    load_csv_for_training,
    make_model_frame,
    summarize_data,
)
from .modeling import (
    choose_best_model,
    comparison_table,
    random_forest_feature_importance,
    save_model_artifacts,
    train_and_evaluate,
)
from .reporting import (
    create_eda_plots,
    ensure_output_dirs,
    save_cleaned_data,
    save_feature_importance_plot,
    write_data_summary,
    write_final_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ambulance response time ML pipeline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train", help="Clean data, run EDA, train models, and write reports."
    )
    train_parser.add_argument("--data", required=True, help="Path to input CSV file.")
    train_parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for figures, models, and reports.",
    )
    train_parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional raw CSV row limit before cleaning.",
    )
    train_parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Optional number of cleaned rows to collect with chunked loading.",
    )
    train_parser.add_argument(
        "--chunksize",
        type=int,
        default=50_000,
        help="CSV chunk size when using --sample-rows.",
    )
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.add_argument("--random-state", type=int, default=42)
    train_parser.add_argument(
        "--check-cases",
        type=int,
        default=20,
        help="Number of case-level predictions to save for manual checking.",
    )

    predict_parser = subparsers.add_parser(
        "predict", help="Predict response time for one incident."
    )
    predict_parser.add_argument("--model", required=True, help="Path to joblib model.")
    predict_parser.add_argument("--hour", type=int, required=True)
    predict_parser.add_argument(
        "--day-of-week",
        type=int,
        required=True,
        help="Monday = 0, Sunday = 6.",
    )
    predict_parser.add_argument("--month", type=int, required=True)
    predict_parser.add_argument("--borough", required=True)
    predict_parser.add_argument("--incident-classification", required=True)
    predict_parser.add_argument("--incident-dispatch-area", default="")
    predict_parser.add_argument("--initial-severity", type=float, default=4)
    predict_parser.add_argument("--final-severity", type=float, default=4)
    predict_parser.add_argument("--zipcode", default="")

    case_parser = subparsers.add_parser(
        "predict-case",
        help="Predict one saved case by CAD incident id with both trained models.",
    )
    case_parser.add_argument("--case-id", required=True, help="CAD_INCIDENT_ID to check.")
    case_parser.add_argument(
        "--data",
        default="data/processed/ems_training_dataset_100000.csv",
        help="Cleaned training dataset CSV that contains the case.",
    )
    case_parser.add_argument(
        "--linear-model",
        default="outputs/models/linear_regression.joblib",
        help="Path to the trained Linear Regression model.",
    )
    case_parser.add_argument(
        "--random-forest-model",
        default="outputs/models/random_forest_regressor.joblib",
        help="Path to the trained Random Forest model.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "predict":
        predict(args)
    elif args.command == "predict-case":
        predict_case(args)


def train(args: argparse.Namespace) -> None:
    paths = ensure_output_dirs(args.output_dir)

    clean_df = load_csv_for_training(
        args.data,
        max_rows=args.max_rows,
        sample_rows=args.sample_rows,
        chunksize=args.chunksize,
    )
    if args.sample_rows is None:
        clean_df = clean_and_engineer_features(clean_df)
    X, y, numeric_features, categorical_features = make_model_frame(clean_df)
    selected_features = numeric_features + categorical_features

    summary = summarize_data(clean_df, selected_features)
    write_data_summary(summary, paths["reports"])
    save_cleaned_data(clean_df, paths["processed"])
    training_dataset_path = (
        paths["processed"] / f"ems_training_dataset_{len(clean_df)}.csv"
    )
    clean_df.to_csv(training_dataset_path, index=False)
    create_eda_plots(clean_df, paths["figures"])

    runs, _X_train, _y_train, X_test, y_test = train_and_evaluate(
        X=X,
        y=y,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    comparison = comparison_table(runs)
    comparison.to_csv(paths["reports"] / "model_comparison.csv", index=False)

    best_run = choose_best_model(runs)
    rf_run = next(run for run in runs if run.name == "Random Forest Regressor")
    feature_importance = random_forest_feature_importance(rf_run, top_n=20)
    feature_importance.to_csv(paths["reports"] / "feature_importance.csv", index=False)
    save_feature_importance_plot(feature_importance, paths["figures"])
    save_model_artifacts(runs, paths["models"])

    case_predictions = build_case_predictions(clean_df, X_test, y_test, runs)
    case_predictions_path = paths["reports"] / "case_level_model_predictions.csv"
    case_predictions.to_csv(case_predictions_path, index=False)
    check_cases_path = paths["reports"] / "sample_cases_for_manual_check.csv"
    case_predictions.head(args.check_cases).to_csv(check_cases_path, index=False)

    predictions = pd.DataFrame(
        {
            "Actual_Response_Time_Minutes": y_test.to_numpy(),
            "Linear_Regression_Predicted_Minutes": get_run_predictions(
                runs, "Linear Regression"
            ),
            "Random_Forest_Predicted_Minutes": get_run_predictions(
                runs, "Random Forest Regressor"
            ),
        }
    )
    predictions.to_csv(paths["reports"] / "model_predictions.csv", index=False)

    report_path = write_final_report(
        reports_dir=paths["reports"],
        summary=summary,
        comparison=comparison,
        feature_importance=feature_importance,
        best_run=best_run,
    )

    print("Training complete.")
    print(f"Saved training dataset: {training_dataset_path}")
    print(f"Cleaned data: {paths['processed'] / 'cleaned_ems_dispatch_data.csv'}")
    print(f"Final report: {report_path}")
    print(f"Linear Regression model: {paths['models'] / 'linear_regression.joblib'}")
    print(f"Random Forest model: {paths['models'] / 'random_forest_regressor.joblib'}")
    print(f"Case-level predictions: {case_predictions_path}")
    print(f"Sample cases for manual check: {check_cases_path}")
    print("Model comparison:")
    print(json.dumps(comparison.to_dict(orient="records"), indent=2))


def get_run_predictions(runs, name: str):
    return next(run.predictions for run in runs if run.name == name)


def build_case_predictions(
    clean_df: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series, runs
) -> pd.DataFrame:
    metadata_columns = [
        "CAD_INCIDENT_ID",
        "INCIDENT_DATETIME",
        "FIRST_ON_SCENE_DATETIME",
        "BOROUGH",
        "INCIDENT_CLASSIFICATION",
        "INCIDENT_DISPATCH_AREA",
        "INITIAL_SEVERITY_LEVEL_CODE",
        "FINAL_SEVERITY_LEVEL_CODE",
    ]
    existing_metadata = [column for column in metadata_columns if column in clean_df.columns]
    cases = clean_df.loc[X_test.index, existing_metadata].copy()
    if "CAD_INCIDENT_ID" in cases.columns:
        cases.insert(0, "Case_Number", cases["CAD_INCIDENT_ID"].astype(str))
    else:
        cases.insert(0, "Case_Number", X_test.index.astype(str))

    cases["Actual_Response_Time_Minutes"] = y_test.to_numpy()
    for run in runs:
        prefix = "Linear_Regression" if run.name == "Linear Regression" else "Random_Forest"
        cases[f"{prefix}_Predicted_Minutes"] = run.predictions
        cases[f"{prefix}_Error_Minutes"] = (
            cases[f"{prefix}_Predicted_Minutes"] - cases["Actual_Response_Time_Minutes"]
        )
    return cases


def predict(args: argparse.Namespace) -> None:
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)
    payload = {
        "Hour": args.hour,
        "DayOfWeek": args.day_of_week,
        "Month": args.month,
        "BOROUGH": args.borough,
        "INCIDENT_CLASSIFICATION": args.incident_classification,
    }

    feature_names = []
    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)

    if "INCIDENT_DISPATCH_AREA" in feature_names or args.incident_dispatch_area:
        payload["INCIDENT_DISPATCH_AREA"] = args.incident_dispatch_area
    if "INITIAL_SEVERITY_LEVEL_CODE" in feature_names:
        payload["INITIAL_SEVERITY_LEVEL_CODE"] = args.initial_severity
    if "FINAL_SEVERITY_LEVEL_CODE" in feature_names:
        payload["FINAL_SEVERITY_LEVEL_CODE"] = args.final_severity
    if "ZIPCODE" in feature_names or args.zipcode:
        payload["ZIPCODE"] = args.zipcode

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Found unknown categories.*",
            category=UserWarning,
            module="sklearn.preprocessing._encoders",
        )
        prediction = float(model.predict(pd.DataFrame([payload]))[0])
    print(f"Predicted Response Time: {prediction:.2f} minutes")


def predict_case(args: argparse.Namespace) -> None:
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Case dataset not found: {data_path}")

    df = pd.read_csv(data_path, low_memory=False)
    if "CAD_INCIDENT_ID" not in df.columns:
        raise ValueError("The case dataset must include CAD_INCIDENT_ID.")

    matches = df[df["CAD_INCIDENT_ID"].astype(str) == str(args.case_id)]
    if matches.empty:
        raise ValueError(f"Case id not found in dataset: {args.case_id}")

    row = matches.iloc[[0]].copy()
    actual = float(row["Response_Time_Minutes"].iloc[0])

    models = {
        "Linear Regression": Path(args.linear_model),
        "Random Forest": Path(args.random_forest_model),
    }
    print(f"Case Number: {args.case_id}")
    print(f"Actual Response Time: {actual:.2f} minutes")

    for name, model_path in models.items():
        if not model_path.exists():
            raise FileNotFoundError(f"{name} model not found: {model_path}")
        model = joblib.load(model_path)
        prediction_input = row
        if hasattr(model, "feature_names_in_"):
            prediction_input = row[list(model.feature_names_in_)]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Found unknown categories.*",
                category=UserWarning,
                module="sklearn.preprocessing._encoders",
            )
            prediction = float(model.predict(prediction_input)[0])
        error = prediction - actual
        print(f"{name} Prediction: {prediction:.2f} minutes")
        print(f"{name} Error: {error:.2f} minutes")
