# Ambulance Response Time Prediction Report

## 1. Project Title

Ambulance Response Time Prediction Using Linear Regression and Random Forest.

## 2. Introduction

This project uses supervised machine learning regression to predict the expected
ambulance response time in minutes. The target variable is `Response_Time_Minutes`, calculated
from the difference between incident creation time and first on-scene time.

## 3. Research Problem

Can historical EMS incident dispatch records be used to predict ambulance
response time and identify the most important operational factors behind delay?

## 4. Objectives

- Clean EMS incident dispatch records.
- Calculate response time in minutes.
- Create time-based features: hour, day of week, and month.
- Explore response time by borough, hour, and incident type.
- Train Linear Regression and Random Forest Regressor.
- Compare models with MAE and RMSE.
- Identify the most important factors affecting response time.

## 5. Dataset Summary

- Rows after loading: 500000
- Columns after loading: 44
- Duplicate rows: 0
- Selected features: Hour, DayOfWeek, Month, INITIAL_SEVERITY_LEVEL_CODE, FINAL_SEVERITY_LEVEL_CODE, DISPATCH_RESPONSE_SECONDS_QY, IsHeld, IsReopen, BOROUGH, INCIDENT_CLASSIFICATION, INCIDENT_DISPATCH_AREA, ZIPCODE
- Missing target values before final model frame, if present: 0

## 6. Data Cleaning and Feature Engineering

The pipeline standardizes column names, converts incident and arrival timestamps,
calculates `Response_Time_Minutes`, removes invalid response times below 0 or above 120
minutes, and drops rows missing required model fields.

## 7. Exploratory Data Analysis

The run creates EDA figures in `outputs/figures`:

- Response time distribution.
- Average response time by borough.
- Average response time by hour.
- Top incident classifications by average response time.

## 8. Algorithms

Linear Regression is used as the baseline model. Random Forest Regressor is used
to capture nonlinear relationships and provide feature importance.

## 9. Model Comparison

| Algorithm               | MAE    | RMSE   |
| ----------------------- | ------ | ------ |
| Stacking Ensemble       | 3.5040 | 5.1943 |
| XGBoost                 | 3.5069 | 5.1994 |
| LightGBM                | 3.5149 | 5.2082 |
| Random Forest Regressor | 3.6041 | 5.3157 |
| Linear Regression       | 3.6878 | 5.4842 |

In this run, **Stacking Ensemble** performed better based on the lowest MAE and
lowest RMSE ordering.

## 10. Feature Importance

| Feature                        | Importance |
| ------------------------------ | ---------- |
| DISPATCH_RESPONSE_SECONDS_QY   | 0.6300     |
| INITIAL_SEVERITY_LEVEL_CODE    | 0.1263     |
| IsHeld                         | 0.1127     |
| Hour                           | 0.0280     |
| DayOfWeek                      | 0.0141     |
| INCIDENT_DISPATCH_AREA_K4      | 0.0085     |
| FINAL_SEVERITY_LEVEL_CODE      | 0.0077     |
| Month                          | 0.0048     |
| BOROUGH_BROOKLYN               | 0.0031     |
| ZIPCODE_11208.0                | 0.0026     |
| ZIPCODE_11370.0                | 0.0022     |
| INCIDENT_DISPATCH_AREA_Q1      | 0.0022     |
| INCIDENT_DISPATCH_AREA_M2      | 0.0012     |
| INCIDENT_DISPATCH_AREA_K2      | 0.0012     |
| ZIPCODE_10036.0                | 0.0012     |
| BOROUGH_MANHATTAN              | 0.0012     |
| IsReopen                       | 0.0012     |
| INCIDENT_CLASSIFICATION_EDPC   | 0.0011     |
| INCIDENT_CLASSIFICATION_CDBRFC | 0.0010     |
| ZIPCODE_11207.0                | 0.0009     |

## 11. Discussion

Response time is influenced by a combination of time, location, and incident
classification. A Random Forest model often performs better when the relationship
between these factors and response time is nonlinear.

## 12. Operational Recommendations

- Increase ambulance availability in high-delay boroughs or dispatch areas.
- Review staffing and unit distribution during peak response-time hours.
- Monitor incident categories that show longer average response times.
- Improve timestamp quality in dispatch systems.
- Retrain the model periodically with recent dispatch records.

## 13. Project Limits

- Results depend on the quality of timestamp and incident classification data.
- Traffic, weather, live ambulance availability, and station distance are not
  included unless they exist in the source dataset.
- The model is for analysis and planning support, not emergency dispatch
  automation without operational validation.

## 14. Conclusion

The project demonstrates how machine learning can support civil defense and EMS
planning by predicting response time, comparing algorithms, and identifying the
main factors linked with delays.
