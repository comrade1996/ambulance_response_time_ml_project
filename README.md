# Ambulance Response Time ML Project

This project predicts ambulance response time in minutes using real EMS incident
dispatch data and two supervised regression models:

- Linear Regression
- Random Forest Regressor

The pipeline loads EMS incident dispatch data, cleans timestamp fields, creates
`Response_Time_Minutes`, performs exploratory analysis, trains both models,
saves both model files, compares MAE/RMSE, exports feature importance, and
writes a final report.

## Project Structure

```text
.
├── ambulance_response_time_ml_project_plan.md
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   ├── TECHNICAL_DOCUMENTATION.md
├── outputs/
│   ├── figures/
│   ├── models/
│   └── reports/
├── scripts/
│   └── generate_sample_data.py
├── src/
│   └── ambulance_response_time_ml/
├── main.py
├── pyproject.toml
└── requirements.txt
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Train Both Models

Use the project virtual environment:

```bash
source .venv/bin/activate
```

Train Linear Regression and Random Forest with the real NYC EMS CSV:

```bash
python main.py train \
  --data data/raw/EMS_Incident_Dispatch_Data.csv \
  --sample-rows 100000 \
  --chunksize 50000 \
  --check-cases 20
```

The raw file is large, so `--sample-rows` and `--chunksize` allow the project to
read the CSV in smaller parts. This command first saves the 100,000-row cleaned
working dataset, then trains both models.

After training, check:

- `data/processed/ems_training_dataset_100000.csv`
- `outputs/reports/final_report.md`
- `outputs/reports/model_comparison.csv`
- `outputs/reports/feature_importance.csv`
- `outputs/reports/case_level_model_predictions.csv`
- `outputs/reports/sample_cases_for_manual_check.csv`
- `outputs/reports/model_predictions.csv`
- `outputs/figures/`
- `outputs/models/linear_regression.joblib`
- `outputs/models/random_forest_regressor.joblib`

## Open the Saved CSV Outputs

Open the 100,000-row cleaned dataset:

```bash
open data/processed/ems_training_dataset_100000.csv
```

Open the small manual-check file:

```bash
open outputs/reports/sample_cases_for_manual_check.csv
```

Open the full case-level prediction file:

```bash
open outputs/reports/case_level_model_predictions.csv
```

The important columns in `case_level_model_predictions.csv` are:

| Column | Meaning |
|---|---|
| `Case_Number` | The CAD incident id used to identify the case. |
| `Actual_Response_Time_Minutes` | The real response time from the dataset. |
| `Linear_Regression_Predicted_Minutes` | Linear Regression prediction. |
| `Linear_Regression_Error_Minutes` | Linear prediction minus actual response time. |
| `Random_Forest_Predicted_Minutes` | Random Forest prediction. |
| `Random_Forest_Error_Minutes` | Random Forest prediction minus actual response time. |

## Dataset Fields

The expected key columns are:

- `INCIDENT_DATETIME`
- `FIRST_ON_SCENE_DATETIME`
- `BOROUGH`
- `INCIDENT_CLASSIFICATION`, or `FINAL_CALL_TYPE`, or `INITIAL_CALL_TYPE`

Optional columns used when available:

- `INCIDENT_DISPATCH_AREA`
- `INITIAL_SEVERITY_LEVEL_CODE`
- `FINAL_SEVERITY_LEVEL_CODE`

## Run Both Models on One Case

After training, run Linear Regression:

```bash
python main.py predict \
  --model outputs/models/linear_regression.joblib \
  --hour 18 \
  --day-of-week 0 \
  --month 5 \
  --borough BROOKLYN \
  --incident-classification INJURY \
  --incident-dispatch-area K5 \
  --initial-severity 4 \
  --final-severity 4
```

Run Random Forest:

```bash
python main.py predict \
  --model outputs/models/random_forest_regressor.joblib \
  --hour 18 \
  --day-of-week 0 \
  --month 5 \
  --borough BROOKLYN \
  --incident-classification INJURY \
  --incident-dispatch-area K5 \
  --initial-severity 4 \
  --final-severity 4
```

`day-of-week` uses Monday = 0 and Sunday = 6.

Example latest outputs for the same case:

```text
Linear Regression: 11.26 minutes
Random Forest: 10.13 minutes
```

## Check a Real Saved Case by Case Number

Take a `Case_Number` from:

```text
outputs/reports/sample_cases_for_manual_check.csv
```

Then run:

```bash
python main.py predict-case --case-id 230192243
```

Example output:

```text
Case Number: 230192243
Actual Response Time: 31.68 minutes
Linear Regression Prediction: 12.68 minutes
Linear Regression Error: -19.01 minutes
Random Forest Prediction: 13.26 minutes
Random Forest Error: -18.43 minutes
```

This lets you open the CSV, find the same case number, and compare the real
response time with both model predictions.

## Compare Both Models

View the comparison table:

```bash
cat outputs/reports/model_comparison.csv
```

Latest real-data sample result:

| Algorithm | MAE | RMSE |
|---|---:|---:|
| Random Forest Regressor | 4.9649 | 8.7104 |
| Linear Regression | 5.1820 | 9.0771 |

Lower MAE and RMSE are better.

## Git Ignore and Dataset Size

The full raw EMS CSV is ignored because it is several gigabytes:

```text
data/raw/*.csv
```

For the repository and web app proof of concept, use the trained 100,000-row
dataset instead:

```text
data/processed/ems_training_dataset_100000.csv
```

## Run the Web App POC

The project includes a Streamlit web app with dropdown-based inputs. It can:

- Check a saved real case and compare actual response time with both models.
- Predict a new incident using dropdowns for borough, incident type, dispatch
  area, time profile, day, month, and severity.
- Show the model comparison table with MAE and RMSE.

Run it locally:

```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```

Then open:

```text
http://localhost:8501
```

Hosting steps are documented in:

```text
docs/WEBAPP_HOSTING_GUIDE.md
```

## Documentation

- Technical details: `docs/TECHNICAL_DOCUMENTATION.md`
- Dataset description: `docs/DATASET_DESCRIPTION.md`
- Web app hosting guide: `docs/WEBAPP_HOSTING_GUIDE.md`
