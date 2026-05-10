# Dataset Description

## Source

The project uses NYC EMS Incident Dispatch Data.

Local source files:

```text
data/raw/EMS_Incident_Dispatch_Data.csv
data/raw/EMS_incident_dispatch_data_description.xlsx
```

The raw CSV is about 6.2 GB and is ignored by git. The deployable project uses
the saved 100,000-row working dataset:

```text
data/processed/ems_training_dataset_100000.csv
```

## What One Row Means

Each row represents one EMS incident. It includes the incident id, timestamps,
call type, severity level, response indicators, borough, dispatch area, ZIP
code, and incident outcome fields.

## Important Fields

| Field | Description |
|---|---|
| `CAD_INCIDENT_ID` | Incident identifier. |
| `INCIDENT_DATETIME` | Time the incident was created. |
| `INITIAL_CALL_TYPE` | Call type at incident creation. |
| `INITIAL_SEVERITY_LEVEL_CODE` | Initial severity level. |
| `FINAL_CALL_TYPE` | Final call type. |
| `FINAL_SEVERITY_LEVEL_CODE` | Final severity level. |
| `FIRST_ASSIGNMENT_DATETIME` | Time first unit was assigned. |
| `DISPATCH_RESPONSE_SECONDS_QY` | Seconds from incident creation to first assignment. |
| `FIRST_ACTIVATION_DATETIME` | Time first unit went enroute. |
| `FIRST_ON_SCENE_DATETIME` | Time first unit arrived on scene. |
| `INCIDENT_RESPONSE_SECONDS_QY` | Seconds from incident creation to first unit on scene. |
| `INCIDENT_TRAVEL_TM_SECONDS_QY` | Seconds from first assignment to first on scene. |
| `INCIDENT_CLOSE_DATETIME` | Time incident closed. |
| `INCIDENT_DISPOSITION_CODE` | Final outcome code. |
| `BOROUGH` | Borough where the incident occurred. |
| `INCIDENT_DISPATCH_AREA` | EMS dispatch area. |
| `ZIPCODE` | Incident ZIP code. |

## Project Mapping

The model feature `INCIDENT_CLASSIFICATION` is created from the real call-type
fields:

```text
INCIDENT_CLASSIFICATION = FINAL_CALL_TYPE
```

If `FINAL_CALL_TYPE` is missing, `INITIAL_CALL_TYPE` can be used.

## Target Variable

The model predicts:

```text
Response_Time_Minutes
```

This means the time between incident creation and first unit arrival on scene.

When available:

```text
Response_Time_Minutes = INCIDENT_RESPONSE_SECONDS_QY / 60
```

Otherwise:

```text
Response_Time_Minutes = FIRST_ON_SCENE_DATETIME - INCIDENT_DATETIME
```

## Current Training Dataset

The current training/deployment dataset is:

```text
data/processed/ems_training_dataset_100000.csv
```

It contains 100,000 cleaned real EMS rows plus one header line in the CSV.

The case-level prediction file contains 20,000 test rows:

```text
outputs/reports/case_level_model_predictions.csv
```

This file is useful for checking a `Case_Number` manually in a spreadsheet and
comparing:

- actual response time
- Linear Regression prediction
- Random Forest prediction

## Latest Metrics

| Algorithm | MAE | RMSE |
|---|---:|---:|
| Random Forest Regressor | 4.96 | 8.71 |
| Linear Regression | 5.18 | 9.08 |

MAE and RMSE are measured in minutes. Lower values are better.
