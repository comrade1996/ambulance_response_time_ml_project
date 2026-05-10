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

- Rows after loading: 100000
- Columns after loading: 36
- Duplicate rows: 0
- Selected features: Hour, DayOfWeek, Month, INITIAL_SEVERITY_LEVEL_CODE, FINAL_SEVERITY_LEVEL_CODE, BOROUGH, INCIDENT_CLASSIFICATION, INCIDENT_DISPATCH_AREA
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
| Random Forest Regressor | 4.9649 | 8.7104 |
| Linear Regression       | 5.1820 | 9.0771 |

In this run, **Random Forest Regressor** performed better based on the lowest MAE and
lowest RMSE ordering.

## 10. Feature Importance

| Feature                        | Importance |
| ------------------------------ | ---------- |
| INITIAL_SEVERITY_LEVEL_CODE    | 0.2597     |
| Hour                           | 0.2052     |
| DayOfWeek                      | 0.1353     |
| INCIDENT_DISPATCH_AREA_M9      | 0.0254     |
| FINAL_SEVERITY_LEVEL_CODE      | 0.0188     |
| INCIDENT_DISPATCH_AREA_K4      | 0.0157     |
| INCIDENT_CLASSIFICATION_ABDPN  | 0.0157     |
| INCIDENT_DISPATCH_AREA_M7      | 0.0155     |
| INCIDENT_CLASSIFICATION_INJURY | 0.0145     |
| INCIDENT_DISPATCH_AREA_M8      | 0.0143     |
| BOROUGH_MANHATTAN              | 0.0129     |
| INCIDENT_CLASSIFICATION_EDP    | 0.0123     |
| INCIDENT_DISPATCH_AREA_B2      | 0.0116     |
| INCIDENT_CLASSIFICATION_SICK   | 0.0108     |
| BOROUGH_BROOKLYN               | 0.0106     |
| INCIDENT_CLASSIFICATION_RESPFC | 0.0102     |
| INCIDENT_DISPATCH_AREA_M6      | 0.0100     |
| INCIDENT_CLASSIFICATION_EDPC   | 0.0092     |
| INCIDENT_CLASSIFICATION_UNKNOW | 0.0087     |
| INCIDENT_CLASSIFICATION_CARDFC | 0.0078     |

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
