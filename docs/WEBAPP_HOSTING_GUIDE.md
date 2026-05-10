# Ambulance Response Time Web App Hosting Guide

This guide explains how to run and host the Streamlit proof-of-concept web app.
The app uses dropdowns, compares both trained models, and lets the user check
saved real cases against actual response time.

## Files Used by the Web App

The web app entry point is:

```text
streamlit_app.py
```

The app expects these project artifacts:

```text
data/processed/ems_training_dataset_100000.csv
outputs/models/linear_regression.joblib
outputs/models/random_forest_regressor.joblib
outputs/reports/case_level_model_predictions.csv
outputs/reports/model_comparison.csv
```

The full raw EMS CSV is too large for a simple demo repository, so it is ignored
by git. Deploy the saved 100,000-row training dataset and the trained model
artifacts instead.

## Train Before Running

From the project folder:

```bash
source .venv/bin/activate
python main.py train \
  --data data/raw/EMS_Incident_Dispatch_Data.csv \
  --sample-rows 100000 \
  --chunksize 50000 \
  --check-cases 20
```

This creates the 100,000-row dataset, model files, reports, and prediction CSVs
needed by the web app.

## Run Locally

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Start the app:

```bash
streamlit run streamlit_app.py
```

Open:

```text
http://localhost:8501
```

## Deploy on Streamlit Community Cloud

1. Push the project to GitHub.
2. Make sure the repository includes:
   - `streamlit_app.py`
   - `requirements.txt`
   - `.streamlit/config.toml`
   - `data/processed/ems_training_dataset_100000.csv`
   - `outputs/models/linear_regression.joblib`
   - `outputs/models/random_forest_regressor.joblib`
   - `outputs/reports/case_level_model_predictions.csv`
   - `outputs/reports/model_comparison.csv`
3. Create a new Streamlit Cloud app from the GitHub repository.
4. Set the app file to `streamlit_app.py`.
5. Deploy.

## Deploy on Render

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0
```

## Demo Flow

1. Open the web app.
2. Show the model performance table in the sidebar.
3. Open `Saved case check`.
4. Select a case number and compare actual response time with both predictions.
5. Open `New prediction`.
6. Choose values from dropdowns.
7. Click `Run both models`.

The app always uses both models. It does not use a hidden selected model.
