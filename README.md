# Ambulance Response Time ML Project

This project predicts ambulance response time in minutes using real EMS incident
dispatch data. It trains and compares five regression models:

- Linear Regression
- Random Forest Regressor
- XGBoost
- LightGBM
- Stacking Ensemble

The project includes a command-line training pipeline, saved model artifacts,
case-level prediction CSV files, charts, an Arabic Streamlit web app, and a
PowerPoint presentation.

## Current Status

Latest training run:

| Item | Value |
|---|---:|
| Clean training rows | 500,000 |
| Test rows in case-level prediction file | 100,000 |
| Best model | Stacking Ensemble |
| Best MAE | 3.5040 |
| Best RMSE | 5.1943 |
| XGBoost MAE | 3.5069 |
| LightGBM MAE | 3.5149 |

Lower MAE and RMSE are better. The app shows all available model predictions.

## Important Files

```text
streamlit_app.py
requirements.txt
data/processed/ems_training_dataset_500000.parquet
outputs/models/linear_regression.joblib
outputs/models/random_forest_regressor.joblib
outputs/models/xgboost.joblib
outputs/models/lightgbm.joblib
outputs/models/stacking_ensemble.joblib
outputs/reports/case_level_model_predictions.csv
outputs/reports/model_comparison.csv
ambulance response time - omir gibreel.pptx
```

The full raw EMS CSV is about 6.2 GB and is ignored by git:

```text
data/raw/*.csv
```

For GitHub and Streamlit deployment, use the compressed 500,000-row Parquet
file instead of uploading the full raw CSV.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Streamlit Community Cloud, the app first loads the compressed 500,000-row
Parquet dataset and the saved `.joblib` model files. If Cloud cannot load the
saved models because of a runtime mismatch, the app retrains deploy-safe models
from the available processed dataset and continues running.

## Train Both Models

```bash
source .venv/bin/activate
python main.py train \
  --data data/raw/EMS_Incident_Dispatch_Data.csv \
  --sample-rows 500000 \
  --chunksize 50000 \
  --check-cases 20
```

This command reads the large raw CSV in chunks, saves
`data/processed/ems_training_dataset_500000.csv`, trains all models, and writes
the reports and prediction files. For deployment, convert the CSV to
`data/processed/ems_training_dataset_500000.parquet`.

## Run the Arabic Web App

```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```

Open:

```text
http://localhost:8501
```

The first screen is a clear Arabic dashboard. The prediction page shows all
available model results automatically.

## Check One Saved Case

Take a `Case_Number` from:

```text
outputs/reports/sample_cases_for_manual_check.csv
```

Then run:

```bash
python main.py predict-case --case-id 230192243
```

Example:

```text
Case Number: 230192243
Actual Response Time: 31.68 minutes
Linear Regression Prediction: 12.68 minutes
Linear Regression Error: -19.01 minutes
Random Forest Prediction: 13.26 minutes
Random Forest Error: -18.43 minutes
```

## Predict with Each Model from CLI

Linear Regression:

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

Random Forest:

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

## Manual Free Streamlit Deployment

1. Create a GitHub repository.
2. Push this project to GitHub.
3. Go to `https://share.streamlit.io`.
4. Click `Create app`.
5. Choose the repository.
6. Use branch `main`.
7. Use main file path `streamlit_app.py`.
8. Deploy.

After deployment, Streamlit will give a URL like:

```text
https://your-app-name.streamlit.app
```

Put the final URL in the last slide of:

```text
ambulance response time - omir gibreel.pptx
```

## Documentation

- Technical details: `docs/TECHNICAL_DOCUMENTATION.md`
- Dataset description: `docs/DATASET_DESCRIPTION.md`
- Web app hosting guide: `docs/WEBAPP_HOSTING_GUIDE.md`
