# Technical Documentation

## 1. Purpose

This project predicts ambulance response time in minutes from historical EMS
incident dispatch records. It implements the project plan using supervised
machine learning regression and compares:

- Linear Regression
- Random Forest Regressor

The target variable is `Response_Time_Minutes`.

## 2. Data Contract

The input must be a CSV file. Column names are normalized by trimming whitespace,
replacing spaces and hyphens with underscores, and converting names to uppercase.

Required columns:

| Column | Description |
|---|---|
| `INCIDENT_DATETIME` | Time the incident was created in the dispatch system. |
| `FIRST_ON_SCENE_DATETIME` | Time the first ambulance unit arrived on scene. |
| `BOROUGH` | Borough or main geographic area. |
| `INCIDENT_CLASSIFICATION` | Incident type or clinical classification. If absent, `FINAL_CALL_TYPE` or `INITIAL_CALL_TYPE` is mapped into this feature. |

Optional feature columns:

| Column | Description |
|---|---|
| `INCIDENT_DISPATCH_AREA` | Dispatch area or operational zone. |
| `ZIPCODE` | Postal code. Retained in cleaned data when present, but not used by default. |

## 3. Target Calculation

`Response_Time_Minutes` is calculated as:

```text
FIRST_ON_SCENE_DATETIME - INCIDENT_DATETIME
```

The result is converted from seconds to minutes.

Rows are removed when:

- Required timestamps cannot be parsed.
- Required model fields are missing.
- Response time is below 0 minutes.
- Response time is above 120 minutes.

The 120-minute limit removes extreme operational or data-entry outliers that are
not useful for the educational prediction task.

## 4. Feature Engineering

The following time features are extracted from `INCIDENT_DATETIME`:

| Feature | Type | Meaning |
|---|---|---|
| `Hour` | Numeric | Hour of day, 0 to 23. |
| `DayOfWeek` | Numeric | Monday = 0, Sunday = 6. |
| `Month` | Numeric | Month number, 1 to 12. |
| `INITIAL_SEVERITY_LEVEL_CODE` | Numeric | Initial dispatch priority when available. |
| `FINAL_SEVERITY_LEVEL_CODE` | Numeric | Final dispatch priority when available. |

Categorical features:

- `BOROUGH`
- `INCIDENT_CLASSIFICATION`
- `INCIDENT_DISPATCH_AREA`, when available

`ZIPCODE` can be added in `config.py` for experiments, but it is excluded by
default to keep the Linear Regression baseline stable.

## 5. Pipeline Architecture

Main modules:

| File | Responsibility |
|---|---|
| `src/ambulance_response_time_ml/config.py` | Shared constants and default paths. |
| `src/ambulance_response_time_ml/data.py` | Loading, column normalization, validation, cleaning, feature engineering. |
| `src/ambulance_response_time_ml/modeling.py` | Preprocessing, model creation, training, metrics, feature importance, model saving. |
| `src/ambulance_response_time_ml/reporting.py` | EDA plots, CSV outputs, JSON summary, final Markdown report. |
| `src/ambulance_response_time_ml/cli.py` | Command-line interface for training and prediction. |
| `scripts/generate_sample_data.py` | Creates a synthetic EMS-like dataset for testing. |

## 6. Model Preprocessing

The model uses a `ColumnTransformer`:

- Numeric columns are scaled with `StandardScaler`.
- Categorical columns are encoded with `OneHotEncoder(handle_unknown="ignore", drop="first")`.

This design lets the trained model accept new category values at prediction time
without failing.

## 7. Algorithms

### Linear Regression

Used as the baseline model. It is fast, easy to interpret, and useful for
checking whether simple linear relationships explain response time.

### Random Forest Regressor

Used to model nonlinear relationships between time, area, incident type, and
response time. It also provides `feature_importances_`, which is exported as a
CSV file and plotted as a figure.

Default Random Forest settings:

```text
n_estimators = 100
max_depth = 12
random_state = 42
n_jobs = -1
```

## 8. Evaluation

The data is split into train and test sets with an 80/20 split by default.

Metrics:

| Metric | Meaning |
|---|---|
| MAE | Mean absolute error in minutes. Lower is better. |
| RMSE | Penalizes large errors more strongly. Lower is better. |

The best model is selected by:

1. Lowest MAE.
2. Lowest RMSE.

## 9. Outputs

Training creates these artifacts:

| Output | Description |
|---|---|
| `data/processed/cleaned_ems_dispatch_data.csv` | Cleaned and feature-engineered dataset. |
| `outputs/reports/data_summary.json` | Row count, columns, missing values, selected features. |
| `outputs/reports/model_comparison.csv` | MAE and RMSE for both models. |
| `outputs/reports/feature_importance.csv` | Top Random Forest feature importances. |
| `outputs/reports/model_predictions.csv` | Actual vs predicted values for both models. |
| `outputs/reports/case_level_model_predictions.csv` | Case-number-level actual response time and both model predictions. |
| `outputs/reports/sample_cases_for_manual_check.csv` | Small set of cases for manual checking in a spreadsheet. |
| `outputs/reports/final_report.md` | Final project report generated from the run. |
| `outputs/models/linear_regression.joblib` | Trained Linear Regression pipeline. |
| `outputs/models/random_forest_regressor.joblib` | Trained Random Forest Regressor pipeline. |
| `outputs/figures/*.png` | EDA and feature-importance plots. |

## 10. Operational Interpretation

The model should be interpreted as a planning and analysis tool. It can support:

- Understanding average delay by area.
- Identifying peak hours with longer response times.
- Comparing incident classifications.
- Improving ambulance placement and resource planning.

It should not replace validated emergency dispatch systems.

## 11. Extending The Project

Recommended future additions:

- Live or historical traffic data.
- Weather data.
- Ambulance station locations.
- Unit availability at incident time.
- A delay classification model.
- A dashboard for operations monitoring.
- API deployment for integration testing.
