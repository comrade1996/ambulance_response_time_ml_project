# Web App Hosting Guide

This guide explains how to run and manually deploy the Arabic Streamlit proof of
concept for the ambulance response time project.

## App Summary

Entry point:

```text
streamlit_app.py
```

Main behavior:

- Arabic right-to-left UI.
- New prediction is the first tab.
- Inputs use dropdowns.
- Dispatch area is filtered by borough.
- Linear Regression and Random Forest predictions appear automatically.
- Saved-case tab compares real response time with both model predictions.

## Required Files for Deployment

The Streamlit app needs these files in the GitHub repository:

```text
streamlit_app.py
runtime.txt
requirements.txt
.streamlit/config.toml
data/processed/ems_training_dataset_100000.csv
outputs/models/linear_regression.joblib
outputs/models/random_forest_regressor.joblib
outputs/reports/case_level_model_predictions.csv
outputs/reports/model_comparison.csv
```

Do not upload the full raw EMS CSV. It is ignored by git:

```text
data/raw/*.csv
```

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open:

```text
http://localhost:8501
```

## Manual Free Deployment on Streamlit Community Cloud

1. Create a GitHub repository.
2. Push the project files to GitHub.
3. Go to:

```text
https://share.streamlit.io
```

4. Click `Create app`.
5. Choose `Yup, I have an app`.
6. Select your GitHub repository.
7. Select branch:

```text
main
```

8. Set the main file path:

```text
streamlit_app.py
```

9. Open advanced settings if shown.
10. Select Python 3.11.
11. Optional: choose a custom app URL.
12. Click `Deploy`.

After deployment, Streamlit gives a public URL like:

```text
https://your-app-name.streamlit.app
```

## Git Commands

If the repository does not have a remote yet:

```bash
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

If the remote already exists:

```bash
git push
```

## Update PowerPoint Link

The presentation file is:

```text
ambulance response time - omir gibreel.pptx
```

The last slide contains a placeholder:

```text
https://your-app-name.streamlit.app
```

After deployment, replace that placeholder with the real Streamlit URL.

## Troubleshooting

If Streamlit says a file is missing, confirm these files were committed and
pushed:

```text
data/processed/ems_training_dataset_100000.csv
outputs/models/linear_regression.joblib
outputs/models/random_forest_regressor.joblib
outputs/reports/case_level_model_predictions.csv
outputs/reports/model_comparison.csv
```

If dependencies fail, confirm `requirements.txt` includes:

```text
pandas==2.3.3
numpy==2.0.2
scikit-learn==1.6.1
matplotlib==3.9.4
joblib==1.5.3
openpyxl==3.1.5
streamlit==1.50.0
```

If the app fails while loading `.joblib` model files, redeploy after confirming
that `runtime.txt` contains `python-3.11`. The model files need package versions
close to the training environment.

If the app is slow on first load, wait for Streamlit to download dependencies
and load the model files.
