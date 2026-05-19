# Project Overview and Enhancement Guide

## Table of Contents

0. [Glossary of Terms](#0-glossary-of-terms)
1. [Project Summary](#1-project-summary)
2. [Problem Statement](#2-problem-statement)
3. [Data Pipeline](#3-data-pipeline)
4. [Feature Engineering](#4-feature-engineering)
5. [Algorithms and Techniques](#5-algorithms-and-techniques)
6. [Model Evaluation](#6-model-evaluation)
7. [Current Limitations](#7-current-limitations)
8. [Enhancement Roadmap: More Accurate Response Time](#8-enhancement-roadmap-more-accurate-response-time)
9. [Enhancement Roadmap: Root Cause Analysis](#9-enhancement-roadmap-root-cause-analysis)

---

## 0. Glossary of Terms

This section defines every technical term used in the document. Refer back here
whenever a term is unclear.

### Machine Learning Fundamentals

| Term | Definition |
|---|---|
| **Machine Learning (ML)** | A branch of artificial intelligence where a computer learns patterns from data instead of being explicitly programmed with rules. |
| **Supervised Learning** | A type of ML where the model learns from labeled examples (input → known correct output). This project is supervised because each training record has a known response time. |
| **Regression** | A supervised learning task where the output is a continuous number (e.g., 7.5 minutes), as opposed to classification where the output is a category (e.g., "fast" or "slow"). |
| **Training** | The process of feeding data to an algorithm so it learns the relationship between input features and the target variable. |
| **Prediction / Inference** | Using a trained model to estimate the output for new, unseen input data. |
| **Model** | The mathematical representation that the algorithm produces after training. It encodes the learned patterns and can make predictions on new data. |
| **Pipeline** | A sequence of data processing steps chained together (e.g., scale numbers → encode categories → run the algorithm). Ensures the same steps are applied during training and prediction. |

### Data Terms

| Term | Definition |
|---|---|
| **Feature** | An individual measurable property used as input to the model. Examples: `Hour`, `BOROUGH`, `INITIAL_SEVERITY_LEVEL_CODE`. Also called an "input variable" or "predictor." |
| **Target Variable** | The value the model is trying to predict. In this project it is `Response_Time_Minutes`. Also called the "label" or "dependent variable." |
| **Row / Record / Observation** | A single data point — one ambulance incident with all its associated fields. |
| **Dataset** | The complete collection of rows used for training and evaluation. |
| **Numeric Feature** | A feature whose values are numbers with meaningful magnitude (e.g., Hour = 14, Severity = 3). |
| **Categorical Feature** | A feature whose values are labels or names without numeric order (e.g., BOROUGH = "BROOKLYN", INCIDENT_CLASSIFICATION = "INJURY"). |
| **Missing Value (NaN)** | A cell in the data that has no value. NaN stands for "Not a Number." Rows with critical missing values are dropped during cleaning. |
| **Outlier** | A data point that is unusually far from the rest. Response times above 120 minutes are treated as outliers and removed. |
| **CSV (Comma-Separated Values)** | A plain-text file format where each line is a row and columns are separated by commas. The standard way tabular data is stored and exchanged. |

### Preprocessing Terms

| Term | Definition |
|---|---|
| **Data Cleaning** | Removing or fixing invalid, missing, or inconsistent data so the model receives high-quality input. |
| **Feature Engineering** | Creating new features from raw data to help the model. Example: extracting `Hour` from a timestamp. |
| **StandardScaler** | Transforms numeric features so they have a mean of 0 and a standard deviation of 1. Prevents features with large numbers (e.g., month 1-12 vs. severity 1-8) from dominating. |
| **OneHotEncoder** | Converts a categorical feature into multiple binary (0/1) columns, one per category. Example: BOROUGH with 5 values becomes 4 binary columns (one is dropped to avoid redundancy). |
| **ColumnTransformer** | A scikit-learn tool that applies different preprocessing steps to different columns in one pass. Numeric columns get scaling; categorical columns get one-hot encoding. |
| **Train-Test Split** | Dividing the dataset into two parts: one to train the model (80%) and one to evaluate it on unseen data (20%). This measures how well the model generalizes. |

### Algorithm Terms

| Term | Definition |
|---|---|
| **Linear Regression** | An algorithm that fits a straight-line equation to the data. Each feature gets a weight (coefficient); the prediction is the weighted sum of all features plus a constant. Simple and fast, but cannot capture curves or complex interactions. |
| **Coefficient (β)** | The weight assigned to a feature in Linear Regression. A coefficient of +2.5 on `Hour` means each additional hour adds 2.5 minutes to the predicted response time (all else equal). |
| **Decision Tree** | An algorithm that splits data using a series of yes/no questions (e.g., "Is Hour > 17?") to arrive at a prediction. Easy to visualize but prone to overfitting. |
| **Random Forest** | An ensemble of many decision trees (default: 100), each trained on a random sample of the data and features. The final prediction is the average of all trees. More accurate and stable than a single tree. |
| **Ensemble** | A method that combines multiple models to produce a better prediction than any single model alone. Random Forest and Gradient Boosting are both ensemble methods. |
| **Gradient Boosting (XGBoost / LightGBM)** | An ensemble method where trees are built one after another, each one correcting the mistakes of the previous. Often the most accurate algorithm for tabular data. |
| **Stacking** | An ensemble technique where predictions from multiple different algorithms are combined by a second-level model to produce a final prediction. |
| **Neural Network** | An algorithm inspired by the brain, made of layers of interconnected nodes. Powerful for very large datasets but harder to interpret and tune. |
| **Overfitting** | When a model memorizes the training data too closely and performs poorly on new data. Like a student who memorizes answers instead of understanding the subject. |
| **Underfitting** | When a model is too simple to capture the patterns in the data. Like using a straight line to approximate a curve. |

### Hyperparameter Terms

| Term | Definition |
|---|---|
| **Hyperparameter** | A setting that controls how the algorithm learns, chosen before training begins. Unlike model parameters (weights), hyperparameters are not learned from data. |
| **n_estimators** | Number of trees in a Random Forest or Gradient Boosting model. More trees usually improve accuracy but increase training time. |
| **max_depth** | Maximum number of levels in each decision tree. Deeper trees capture more detail but risk overfitting. |
| **learning_rate** | (Gradient Boosting) How much each new tree's contribution is scaled down. Smaller values need more trees but often give better accuracy. |
| **random_state** | A seed number that makes random operations reproducible. Using the same seed gives the same results every run. |
| **n_jobs** | Number of CPU cores to use. `-1` means "use all available cores" for faster training. |
| **Hyperparameter Tuning** | Systematically trying different hyperparameter combinations to find the best-performing settings. |
| **Cross-Validation (CV)** | Splitting the training data into K folds, training on K-1 folds and testing on the remaining fold, repeated K times. Gives a more reliable estimate of model performance than a single split. |
| **RandomizedSearchCV** | A tuning method that randomly samples hyperparameter combinations and evaluates each with cross-validation. Faster than trying every combination (GridSearchCV). |

### Evaluation Metric Terms

| Term | Definition |
|---|---|
| **MAE (Mean Absolute Error)** | The average of the absolute differences between predicted and actual values. If MAE = 5.0, the model is off by 5 minutes on average. Easy to interpret. |
| **RMSE (Root Mean Squared Error)** | The square root of the average of squared differences. Penalizes large errors more than MAE. If RMSE is much larger than MAE, a few predictions have very large errors. |
| **Residual** | The difference between the actual value and the predicted value for a single data point: `residual = actual - predicted`. Positive means the model underestimated. |
| **R² (R-Squared)** | The proportion of variance in the target explained by the model. R² = 1.0 is perfect; R² = 0.0 means the model is no better than predicting the average every time. |

### Explainability Terms

| Term | Definition |
|---|---|
| **Feature Importance** | A score (0 to 1) assigned to each feature by Random Forest, indicating how much that feature contributed to reducing prediction error across all trees. Higher = more influential. |
| **SHAP (SHapley Additive exPlanations)** | A method from game theory that calculates the exact contribution of each feature to a specific prediction. Answers: "Why did the model predict 15 minutes for this case?" |
| **Partial Dependence Plot (PDP)** | A chart showing how the predicted value changes as one feature varies, while all other features are held at their average. Reveals the shape of the relationship (linear, curved, etc.). |
| **Root Cause Analysis** | The process of identifying the underlying reasons why something happens (e.g., why response time is long), not just predicting it. Goes beyond correlation to actionable explanation. |

### Infrastructure and Deployment Terms

| Term | Definition |
|---|---|
| **scikit-learn (sklearn)** | The Python library used for preprocessing, training, and evaluating the models in this project. |
| **joblib** | A Python library for saving and loading trained models to/from disk as `.joblib` files. |
| **Streamlit** | An open-source Python framework for building interactive web apps from Python scripts. Used here for the Arabic prediction interface. |
| **CLI (Command-Line Interface)** | A text-based way to run programs by typing commands in a terminal, as opposed to clicking buttons in a graphical interface. |
| **EMS (Emergency Medical Services)** | The system responsible for dispatching ambulances and paramedics to emergency incidents. |
| **Chunked Reading** | Loading a large file in smaller pieces (chunks) instead of all at once, to avoid running out of memory. |
| **EDA (Exploratory Data Analysis)** | The initial phase of examining data using statistics and charts to understand distributions, relationships, and anomalies before building models. |
| **Causal Inference** | Statistical methods that attempt to determine whether one variable actually causes a change in another, rather than just being correlated with it. |

---

## 1. Project Summary

This project predicts **ambulance response time in minutes** using historical
EMS (Emergency Medical Services) incident dispatch records from New York City.
The response time is defined as the elapsed time between when an incident is
reported (`INCIDENT_DATETIME`) and when the first ambulance unit arrives on
scene (`FIRST_ON_SCENE_DATETIME`).

The system trains two regression models simultaneously and always presents both
predictions side by side so users can compare:

| Model | Role |
|---|---|
| **Linear Regression** | Simple baseline; assumes a straight-line relationship between features and response time. |
| **Random Forest Regressor** | Ensemble of decision trees; captures nonlinear patterns and provides feature importance. |

### Deliverables

- **CLI pipeline** (`main.py`) for data cleaning, training, evaluation, and
  single-case prediction.
- **Arabic Streamlit web app** (`streamlit_app.py`) for interactive prediction
  and case comparison.
- **Saved model artifacts** (`.joblib` files), comparison reports, EDA charts,
  and a PowerPoint presentation.

### Architecture at a Glance

```
data/raw/EMS_Incident_Dispatch_Data.csv   (raw ~6.2 GB, git-ignored)
        │
        ▼
  src/.../data.py       ── clean, validate, engineer features
        │
        ▼
  src/.../modeling.py    ── preprocess (scale + encode), train, evaluate
        │
        ▼
  src/.../reporting.py   ── charts, CSVs, markdown report
        │
        ▼
  outputs/models/        ── linear_regression.joblib
                            random_forest_regressor.joblib
  outputs/reports/       ── model_comparison.csv, feature_importance.csv, etc.
  outputs/figures/       ── EDA and feature importance PNGs
```

---

## 2. Problem Statement

**Question:** Given the borough, dispatch area, incident type, severity, time of
day, day of week, and month, how many minutes will it take for the first
ambulance unit to arrive on scene?

This is a **supervised regression** problem. The target variable
(`Response_Time_Minutes`) is continuous, measured in minutes.

### Why It Matters

- Helps EMS planners identify **when and where** delays happen.
- Supports staffing and resource allocation decisions.
- Enables comparison between different geographic and temporal patterns.

---

## 3. Data Pipeline

### 3.1 Data Loading

The raw NYC EMS CSV is ~6.2 GB. To make training practical, the pipeline reads
it in chunks (`chunksize=50000`) and stops after collecting the desired number
of clean rows (`--sample-rows 500000` in the current run). This avoids loading the entire file into
memory.

### 3.2 Column Standardization

All column names are uppercased, trimmed, and have spaces/hyphens replaced with
underscores. This prevents mismatches caused by inconsistent CSV headers.

### 3.3 Validation

The pipeline enforces the presence of:

- `INCIDENT_DATETIME`
- `FIRST_ON_SCENE_DATETIME`
- `BOROUGH`
- At least one of: `INCIDENT_CLASSIFICATION`, `FINAL_CALL_TYPE`, or
  `INITIAL_CALL_TYPE`

### 3.4 Target Calculation

```
Response_Time_Minutes = (FIRST_ON_SCENE_DATETIME - INCIDENT_DATETIME) / 60
```

If the dataset contains the pre-calculated column
`INCIDENT_RESPONSE_SECONDS_QY` (with a validity flag), that value is used
instead, falling back to the timestamp-based calculation when invalid or
missing.

### 3.5 Filtering

- Rows with unparseable timestamps are dropped.
- Rows with response time < 0 or > 120 minutes are removed.
- Rows missing any required field after cleaning are dropped.

---

## 4. Feature Engineering

### 4.1 Numeric Features

| Feature | Source | Description |
|---|---|---|
| `Hour` | `INCIDENT_DATETIME.dt.hour` | Hour of day (0-23). Captures rush-hour and overnight patterns. |
| `DayOfWeek` | `INCIDENT_DATETIME.dt.dayofweek` | Monday=0 through Sunday=6. Captures weekday vs. weekend effects. |
| `Month` | `INCIDENT_DATETIME.dt.month` | Captures seasonal variation (e.g., winter storms, summer heat). |
| `INITIAL_SEVERITY_LEVEL_CODE` | Raw field | Severity assessment at time of call. Higher severity may trigger faster dispatch. |
| `FINAL_SEVERITY_LEVEL_CODE` | Raw field | Updated severity after on-scene evaluation. |

### 4.2 Categorical Features

| Feature | Description |
|---|---|
| `BOROUGH` | One of: BRONX, BROOKLYN, MANHATTAN, QUEENS, STATEN ISLAND. |
| `INCIDENT_CLASSIFICATION` | Type of emergency (INJURY, CARD, SICK, etc.). |
| `INCIDENT_DISPATCH_AREA` | Specific EMS dispatch zone within a borough. |

### 4.3 Preprocessing

A `ColumnTransformer` handles both types:

- **Numeric columns**: `StandardScaler` (zero mean, unit variance). Ensures all
  numeric features are on the same scale, preventing large-valued features from
  dominating.
- **Categorical columns**: `OneHotEncoder(handle_unknown="ignore", drop="first")`.
  Creates binary columns for each category, drops the first to avoid
  multicollinearity, and ignores unknown categories at prediction time.

---

## 5. Algorithms and Techniques

### 5.1 Linear Regression

**What it does:** Fits a linear equation that maps the weighted sum of all
features to the predicted response time.

```
ŷ = β₀ + β₁·Hour + β₂·DayOfWeek + ... + βₙ·OneHot_Borough_BROOKLYN + ...
```

**Strengths:**
- Fast to train and interpret.
- Useful baseline: if a nonlinear model cannot beat it, the added complexity is
  not justified.
- Coefficients directly show each feature's marginal effect in minutes.

**Weaknesses:**
- Assumes a strictly linear relationship. If response time changes nonlinearly
  with hour (e.g., sharp spike at 5 PM), linear regression cannot capture it
  without manual feature engineering (polynomial terms, interaction terms).

### 5.2 Random Forest Regressor

**What it does:** Builds an ensemble of 100 decision trees, each trained on a
random bootstrap sample with a random subset of features. The final prediction
is the average of all tree predictions.

**Current hyperparameters:**

| Parameter | Value | Purpose |
|---|---|---|
| `n_estimators` | 100 | Number of trees in the forest. |
| `max_depth` | 12 | Maximum tree depth. Prevents overfitting by limiting complexity. |
| `random_state` | 42 | Ensures reproducibility. |
| `n_jobs` | -1 | Uses all CPU cores for parallel training. |

**Strengths:**
- Captures nonlinear interactions (e.g., "BRONX at 2 AM in January" behaves
  differently from "MANHATTAN at noon in July").
- Provides **feature importance** scores that rank which inputs most influence
  predictions.
- Robust to outliers and does not require feature scaling (though scaling is
  applied here for pipeline uniformity).

**Weaknesses:**
- Slower to train and predict than linear regression.
- `max_depth=12` limits tree complexity; too shallow may underfit, too deep may
  overfit.
- Cannot extrapolate beyond the range of training data.

### 5.3 Why Multiple Models?

Presenting multiple models side by side serves two purposes:

1. **Validation**: If models agree, confidence in the prediction is higher.
   Large disagreement signals that the case is unusual or model assumptions differ.
2. **Transparency**: The audience (planners, stakeholders) can see how much
   accuracy improves from the simple baseline to the more complex model.

---

## 6. Model Evaluation

### 6.1 Train-Test Split

The dataset is split 80/20 with `random_state=42`:

- **80% (400,000 rows)**: Used to train the models.
- **20% (100,000 rows)**: Held out to evaluate prediction accuracy on unseen
  data.

### 6.2 Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| **MAE** (Mean Absolute Error) | `mean(|actual - predicted|)` | Average prediction error in minutes. Easy to interpret: "on average the model is off by X minutes." |
| **RMSE** (Root Mean Squared Error) | `sqrt(mean((actual - predicted)²))` | Penalizes large errors more heavily. If RMSE is much larger than MAE, a few cases have very large errors. |

### 6.3 Current Results (500,000 rows)

| Algorithm | MAE (min) | RMSE (min) |
|---|---:|---:|
| Stacking Ensemble | 3.50 | 5.19 |
| XGBoost | 3.51 | 5.20 |
| LightGBM | 3.51 | 5.21 |
| Random Forest Regressor | 3.60 | 5.32 |
| Linear Regression | 3.69 | 5.48 |

**Interpretation:** The ensemble and boosted-tree models perform best. The gap
between MAE and RMSE still indicates some large-error long-tail incidents.

---

## 7. Current Limitations

| Limitation | Impact |
|---|---|
| Only 8 features used | Many real-world factors that affect response time are missing. |
| No traffic or weather data | These are major drivers of delay but are not in the dataset. |
| No ambulance availability data | Whether units are free or busy is unknown. |
| No geographic distance feature | Straight-line or road distance to station is not calculated. |
| No hyperparameter tuning | Random Forest uses default-like settings without optimization. |
| No cross-validation | Single 80/20 split may give optimistic or pessimistic estimates. |
| 120-minute cap | Extreme outliers are removed, but near-120 cases still distort RMSE. |
| No temporal validation | Train/test split is random, not time-ordered; the model may "see the future." |

---

## 8. Enhancement Roadmap: More Accurate Response Time

### 8.1 Add More Features (High Impact)

#### 8.1.1 Weather Data

Integrate hourly weather data (rain, snow, temperature, visibility) by joining
on `INCIDENT_DATETIME` rounded to the nearest hour and borough/zip code.

```python
# Example: merge weather on date-hour and borough
weather = pd.read_csv("weather_hourly.csv")
weather["datetime_hour"] = pd.to_datetime(weather["datetime"]).dt.floor("h")
df["datetime_hour"] = df["INCIDENT_DATETIME"].dt.floor("h")
df = df.merge(weather, on=["datetime_hour", "BOROUGH"], how="left")
```

**Expected improvement:** 5-15% reduction in MAE for weather-sensitive incidents.

#### 8.1.2 Traffic Congestion

Add average traffic speed or congestion index for the area and time of day.
NYC publishes traffic speed data via NYC Open Data.

#### 8.1.3 Geographic Distance

Calculate the straight-line distance between the incident location (latitude,
longitude) and the nearest ambulance station. If GPS coordinates are available
in the raw data:

```python
from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))
```

#### 8.1.4 Ambulance Workload

If dispatch logs include unit IDs and timestamps, calculate the number of
concurrent active incidents at the time of the new call. High workload
correlates with longer response times.

#### 8.1.5 Holiday and Special Events

Add a boolean `is_holiday` feature and flags for known large events (New Year,
July 4th, marathons) that affect traffic and incident volume.

### 8.2 Better Algorithms (Medium-High Impact)

#### 8.2.1 Gradient Boosting (XGBoost / LightGBM)

Gradient boosting typically outperforms Random Forest on tabular data:

```python
from xgboost import XGBRegressor

xgb_model = XGBRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)
```

**Why:** Boosting builds trees sequentially, each correcting the errors of the
previous ones. This often gives 10-20% lower MAE than Random Forest.

#### 8.2.2 Neural Networks (for large-scale data)

With millions of rows and many features, a simple feedforward neural network
(e.g., 3 hidden layers with 256-128-64 units) can capture complex interactions.
Use only if data volume justifies the complexity.

#### 8.2.3 Ensemble Stacking

Combine Linear Regression, Random Forest, and XGBoost into a stacking ensemble:

```python
from sklearn.ensemble import StackingRegressor

stacking = StackingRegressor(
    estimators=[
        ("lr", LinearRegression()),
        ("rf", RandomForestRegressor(n_estimators=100, max_depth=12)),
        ("xgb", XGBRegressor(n_estimators=300, max_depth=8)),
    ],
    final_estimator=LinearRegression(),
    cv=5,
)
```

### 8.3 Hyperparameter Tuning (Medium Impact)

Use `RandomizedSearchCV` or `Optuna` to optimize Random Forest/XGBoost
hyperparameters:

```python
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    "model__n_estimators": [100, 200, 500, 800],
    "model__max_depth": [8, 12, 16, 20, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2", 0.5],
}

search = RandomizedSearchCV(
    pipeline,
    param_distributions,
    n_iter=50,
    cv=5,
    scoring="neg_mean_absolute_error",
    random_state=42,
    n_jobs=-1,
)
search.fit(X_train, y_train)
```

### 8.4 Use More Training Data (Medium Impact)

The full raw dataset has millions of rows. Currently 500,000 cleaned rows are
used. Increasing beyond 500,000 may give the model more patterns to learn,
especially for rare dispatch areas and incident types.

### 8.5 Temporal Validation (Important for Reliability)

Replace the random 80/20 split with a **time-based split**: train on older data,
test on newer data. This simulates real deployment where the model predicts
future incidents.

```python
# Sort by time, use the last 20% as test
df_sorted = df.sort_values("INCIDENT_DATETIME")
split_index = int(len(df_sorted) * 0.8)
X_train, X_test = df_sorted.iloc[:split_index], df_sorted.iloc[split_index:]
```

### 8.6 Advanced Feature Engineering

| Technique | Description |
|---|---|
| **Cyclical encoding** | Encode `Hour`, `DayOfWeek`, `Month` as sin/cos pairs instead of raw integers, so the model knows that hour 23 and hour 0 are adjacent. |
| **Target encoding** | Replace high-cardinality categories (e.g., `INCIDENT_DISPATCH_AREA` with 100+ values) with the mean response time for that category. Reduces dimensionality. |
| **Interaction features** | Create `Borough_x_Hour` or `Severity_x_IncidentType` combinations to capture joint effects. |
| **Rolling averages** | Average response time in the same borough over the previous 7 days. Captures local trends. |
| **Lag features** | Response time of the last N incidents in the same dispatch area. Captures short-term workload. |

### 8.7 Summary: Expected Impact per Enhancement

| Enhancement | Effort | Expected MAE Reduction |
|---|---|---|
| Add weather data | Medium | 5-15% |
| Add traffic data | Medium | 5-10% |
| Add geographic distance | Low-Medium | 5-10% |
| Switch to XGBoost/LightGBM | Low | 10-20% |
| Hyperparameter tuning | Low | 5-10% |
| More training data (500K+) | Low | 3-8% |
| Cyclical time encoding | Low | 2-5% |
| Stacking ensemble | Medium | 5-10% |
| Rolling/lag features | Medium | 5-10% |

Combining multiple enhancements can potentially reduce MAE from ~5.0 to ~2.5-3.0 minutes.

---

## 9. Enhancement Roadmap: Root Cause Analysis

Understanding **why** response time is long matters as much as predicting it.
Below are techniques to move from prediction to explanation.

### 9.1 Feature Importance Analysis (Already Available)

The Random Forest already provides feature importance. To get more out of it:

- **Sort and visualize** the top 10-15 features.
- **Compare importance across boroughs**: train separate models per borough and
  compare which features matter most in each.

### 9.2 SHAP Values (Recommended)

SHAP (SHapley Additive exPlanations) gives a per-prediction breakdown of each
feature's contribution:

```python
import shap

explainer = shap.TreeExplainer(rf_model.named_steps["model"])
X_test_transformed = rf_model.named_steps["preprocessor"].transform(X_test)
shap_values = explainer.shap_values(X_test_transformed)

# Summary plot: which features push response time up or down
shap.summary_plot(shap_values, X_test_transformed,
                  feature_names=transformed_feature_names(rf_model))

# Single-case explanation: why was THIS case predicted as 15 minutes?
shap.waterfall_plot(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    feature_names=transformed_feature_names(rf_model),
))
```

**What it reveals:**
- For a specific incident with a 25-minute predicted response: "Being in the
  BRONX added +3 min, hour=2AM added +4 min, severity=1 added +2 min."
- Across the dataset: "BOROUGH is the dominant factor overall, but Hour
  matters more during evening hours."

### 9.3 Partial Dependence Plots (PDP)

Show how changing one feature affects the predicted response time while
averaging over all other features:

```python
from sklearn.inspection import PartialDependenceDisplay

PartialDependenceDisplay.from_estimator(
    rf_model, X_test, features=["Hour", "INITIAL_SEVERITY_LEVEL_CODE"],
    kind="average"
)
```

**Use case:** Visualize the exact hour when response time peaks, or how
response time changes as severity increases.

### 9.4 Residual Analysis

Analyze cases where the model prediction is far from the actual value:

```python
residuals = y_test - predictions
large_errors = X_test[abs(residuals) > 15]  # cases with >15 min error
```

Then examine:
- **Which boroughs** have the most large errors?
- **Which hours** are most unpredictable?
- **Which incident types** are hardest to predict?

This reveals **systematic blind spots** in the model, pointing to missing
features or data quality issues.

### 9.5 Segmented Analysis

Train or evaluate separate models per segment to isolate root causes:

| Segment | What You Learn |
|---|---|
| Per borough | Which borough has the worst average response time and why. |
| Per incident type | Which emergencies take longest; are cardiac cases prioritized? |
| Per time band (peak vs. off-peak) | Does staffing during peak hours match demand? |
| Per severity level | Are high-severity cases actually dispatched faster? |

### 9.6 Clustering Delay Patterns

Use unsupervised learning to group incidents by their delay profile:

```python
from sklearn.cluster import KMeans

# Cluster on features + response time
kmeans = KMeans(n_clusters=5, random_state=42)
df["delay_cluster"] = kmeans.fit_predict(X_scaled)
```

Then describe each cluster:
- **Cluster 0:** Low-severity, daytime, Manhattan, ~5 min response.
- **Cluster 3:** High-severity, overnight, Bronx, ~20 min response.

This reveals **natural groups of delay patterns** that can inform targeted
operational improvements.

### 9.7 Causal Inference (Advanced)

To move beyond correlation to causation:

- **Difference-in-Differences**: Compare response times before and after a
  policy change (e.g., adding a new station).
- **Instrumental Variables**: Use weather as an instrument for traffic when
  studying traffic's causal effect on response time.
- **Propensity Score Matching**: Compare similar incidents across boroughs to
  isolate the effect of location.

### 9.8 Actionable Root Cause Dashboard

Build on the Streamlit app to add a "Root Cause" tab that shows:

1. **Borough-level heatmap** of average response time by hour.
2. **SHAP waterfall** for the selected case showing why the model predicted
   that value.
3. **Worst-performing segments** table: top 5 borough + hour combinations
   with the highest average delay.
4. **Trend chart** showing monthly response time trends per borough.

---

## Summary: Priority Action Items

| Priority | Action | Impact |
|---|---|---|
| 1 | Add SHAP values for per-case explanations | Root cause visibility |
| 2 | Switch to XGBoost/LightGBM | 10-20% accuracy gain |
| 3 | Add weather + traffic features | 10-20% accuracy gain |
| 4 | Hyperparameter tuning with cross-validation | 5-10% accuracy gain |
| 5 | Temporal train/test split | More realistic evaluation |
| 6 | Residual analysis + segmented reports | Identifies blind spots |
| 7 | Increase training data to 500K+ rows | Better coverage of rare cases |
| 8 | Cyclical encoding + interaction features | 2-10% accuracy gain |
| 9 | Root cause dashboard in Streamlit | Stakeholder communication |
| 10 | Causal inference studies | Policy-level insights |
