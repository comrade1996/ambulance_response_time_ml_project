# Dataset Description

## Source Files

- Raw data: `data/raw/EMS_Incident_Dispatch_Data.csv`
- Description workbook: `data/raw/EMS_incident_dispatch_data_description.xlsx`

## Dataset

The dataset is the NYC EMS Incident Dispatch Data. It contains emergency medical
incident records created by the dispatch system. Each row represents an incident
and includes timestamps, call type, severity, dispatch response indicators,
borough, dispatch area, ZIP code, and final incident outcome.

## Important Fields

| Field | Description |
|---|---|
| `CAD_INCIDENT_ID` | Incident identifier. |
| `INCIDENT_DATETIME` | Date and time the incident was created in the dispatch system. |
| `INITIAL_CALL_TYPE` | Call type assigned when the incident was created. |
| `INITIAL_SEVERITY_LEVEL_CODE` | Initial priority or severity level. |
| `FINAL_CALL_TYPE` | Call type when the incident closes. |
| `FINAL_SEVERITY_LEVEL_CODE` | Final priority or severity level. |
| `FIRST_ASSIGNMENT_DATETIME` | Date and time the first unit was assigned. |
| `DISPATCH_RESPONSE_SECONDS_QY` | Seconds between incident creation and first assignment. |
| `FIRST_ACTIVATION_DATETIME` | Date and time the first unit went enroute. |
| `FIRST_ON_SCENE_DATETIME` | Date and time the first unit arrived on scene. |
| `INCIDENT_RESPONSE_SECONDS_QY` | Seconds between incident creation and first unit on scene. |
| `INCIDENT_TRAVEL_TM_SECONDS_QY` | Seconds between first assignment and first on scene. |
| `INCIDENT_CLOSE_DATETIME` | Date and time the incident closed. |
| `INCIDENT_DISPOSITION_CODE` | Final incident outcome code. |
| `BOROUGH` | Borough where the incident occurred. |
| `INCIDENT_DISPATCH_AREA` | EMS dispatch area. |
| `ZIPCODE` | Incident ZIP code. |

## Project Mapping

The original project plan expected `INCIDENT_CLASSIFICATION`. In the real
dataset, this is represented by call type fields, so the implementation maps:

```text
INCIDENT_CLASSIFICATION = FINAL_CALL_TYPE
```

If `FINAL_CALL_TYPE` is unavailable, `INITIAL_CALL_TYPE` can be used.

The target variable is:

```text
Response_Time_Minutes = INCIDENT_RESPONSE_SECONDS_QY / 60
```

When the seconds field is unavailable, the pipeline calculates response time
from:

```text
FIRST_ON_SCENE_DATETIME - INCIDENT_DATETIME
```

## Real Training Run

Because the raw CSV is about 6.2 GB, the current implementation uses chunked
loading. The latest run used 100,000 cleaned real rows from the official CSV.

Latest model comparison:

| Algorithm | MAE | RMSE |
|---|---:|---:|
| Random Forest Regressor | 4.96 | 8.71 |
| Linear Regression | 5.18 | 9.08 |

The best model in the latest run is Random Forest Regressor.
