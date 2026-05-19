# Technical Documentation

## Purpose

The project predicts ambulance response time in minutes from historical EMS
incident dispatch records. It is a regression project, so the model output is a
number of minutes, not a class label.

Two models are trained and used:

- Linear Regression
- Random Forest Regressor

The CLI and the Streamlit app show both model outputs.

## Data Contract

The input is a CSV file. Column names are normalized by trimming whitespace,
replacing spaces and hyphens with underscores, and converting names to uppercase.

Required fields:

| Column | Purpose |
|---|---|
| `INCIDENT_DATETIME` | Time the incident was created. |
| `FIRST_ON_SCENE_DATETIME` | Time the first unit arrived on scene. |
| `BOROUGH` | Main geographic area. |
| `INCIDENT_CLASSIFICATION` | Incident type. If absent, the pipeline maps it from `FINAL_CALL_TYPE` or `INITIAL_CALL_TYPE`. |

Optional fields used when available:

| Column | Purpose |
|---|---|
| `INCIDENT_DISPATCH_AREA` | EMS dispatch area. |
| `INITIAL_SEVERITY_LEVEL_CODE` | Initial severity level. |
| `FINAL_SEVERITY_LEVEL_CODE` | Final severity level. |
| `ZIPCODE` | Kept in cleaned data, but not used by the default models. |

## Target Variable

The target is:

```text
Response_Time_Minutes
```

When possible, it is calculated from:

```text
FIRST_ON_SCENE_DATETIME - INCIDENT_DATETIME
```

Rows are removed when timestamps cannot be parsed, required fields are missing,
response time is negative, or response time is above 120 minutes.

## Features

Numeric features:

| Feature | Meaning |
|---|---|
| `Hour` | Hour of day, 0 to 23. |
| `DayOfWeek` | Monday = 0, Sunday = 6. |
| `Month` | Month number, 1 to 12. |
| `INITIAL_SEVERITY_LEVEL_CODE` | Initial severity level. |
| `FINAL_SEVERITY_LEVEL_CODE` | Final severity level. |

Categorical features:

- `BOROUGH`
- `INCIDENT_CLASSIFICATION`
- `INCIDENT_DISPATCH_AREA`

## Pipeline Architecture

| File | Responsibility |
|---|---|
| `main.py` | Project entry point. |
| `src/ambulance_response_time_ml/cli.py` | Training, prediction, and case-check commands. |
| `src/ambulance_response_time_ml/data.py` | Data loading, cleaning, and feature engineering. |
| `src/ambulance_response_time_ml/modeling.py` | Preprocessing, training, metrics, and model saving. |
| `src/ambulance_response_time_ml/reporting.py` | Reports, CSV outputs, charts, and summaries. |
| `streamlit_app.py` | Arabic web app proof of concept. |

## Model Preprocessing

The model uses a `ColumnTransformer`:

- Numeric columns use `StandardScaler`.
- Categorical columns use `OneHotEncoder(handle_unknown="ignore", drop="first")`.

This lets the models handle unseen categories during prediction without failing.

## Algorithms

### Linear Regression

Linear Regression is the baseline. It is fast, simple, and useful for comparing
against a more flexible model.

### Random Forest Regressor

Random Forest captures nonlinear relationships between time, location, incident
type, severity, and response time. It also provides feature importance values.

Current Random Forest settings:

```text
n_estimators = 100
max_depth = 12
random_state = 42
n_jobs = -1
```

## Evaluation

The training command uses an 80/20 train-test split by default.

Metrics:

| Metric | Meaning |
|---|---|
| MAE | Mean absolute error in minutes. Lower is better. |
| RMSE | Root mean squared error. It penalizes large errors more strongly. Lower is better. |

Latest 100,000-row run:

| Algorithm | MAE | RMSE |
|---|---:|---:|
| Random Forest Regressor | 4.9649 | 8.7104 |
| Linear Regression | 5.1820 | 9.0771 |

## Outputs

| Output | Description |
|---|---|
| `data/processed/ems_training_dataset_100000.csv` | The 100,000-row working dataset used for training and deployment. |
| `data/processed/cleaned_ems_dispatch_data.csv` | Generated cleaned data file. Ignored by git. |
| `outputs/models/linear_regression.joblib` | Trained Linear Regression pipeline. |
| `outputs/models/random_forest_regressor.joblib` | Trained Random Forest pipeline. |
| `outputs/reports/model_comparison.csv` | MAE and RMSE for both models. |
| `outputs/reports/case_level_model_predictions.csv` | Case-number-level actual and predicted response times. |
| `outputs/reports/sample_cases_for_manual_check.csv` | Small sample for spreadsheet checking. |
| `outputs/reports/model_predictions.csv` | Actual vs predicted values for both models. |
| `outputs/reports/feature_importance.csv` | Random Forest feature importances. |
| `outputs/figures/*.png` | EDA and feature-importance charts. |

## Arabic Streamlit App

The app is implemented in `streamlit_app.py`.

Current behavior:

- Arabic UI with right-to-left layout.
- The first tab is new prediction.
- Inputs are mostly dropdowns.
- Dispatch area is filtered by selected borough.
- Predictions appear automatically for both models.
- A saved-case tab lets the user compare real response time with both model predictions.
- A summary tab shows the dataset and model metrics.

The app uses:

```text
requirements.txt
data/processed/ems_training_dataset_deploy.csv
outputs/models/linear_regression.joblib
outputs/models/random_forest_regressor.joblib
outputs/reports/case_level_model_predictions.csv
outputs/reports/model_comparison.csv
```

During deployment, the app first tries to load the saved `.joblib` model files.
If the cloud runtime cannot unpickle them because of a package-version mismatch,
the app retrains both models from the saved 100,000-row dataset and uses those
runtime-trained models for prediction.

## Deployment Notes

The full raw CSV is ignored because it is several gigabytes:

```text
data/raw/*.csv
```

Deploy only the 100,000-row dataset, model files, and report CSVs required by
the web app. Manual deployment steps are in `docs/WEBAPP_HOSTING_GUIDE.md`.

## Limitations

This is an educational proof of concept. It should support analysis and
presentation, but it should not replace validated emergency dispatch systems.
