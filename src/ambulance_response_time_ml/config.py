from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"

INCIDENT_DATETIME = "INCIDENT_DATETIME"
FIRST_ON_SCENE_DATETIME = "FIRST_ON_SCENE_DATETIME"
TARGET = "Response_Time_Minutes"

REQUIRED_COLUMNS = [
    INCIDENT_DATETIME,
    FIRST_ON_SCENE_DATETIME,
    "BOROUGH",
]

OPTIONAL_FEATURE_COLUMNS = [
    "INCIDENT_DISPATCH_AREA",
    "ZIPCODE",
]

NUMERIC_FEATURES = [
    "Hour",
    "DayOfWeek",
    "Month",
    "INITIAL_SEVERITY_LEVEL_CODE",
    "FINAL_SEVERITY_LEVEL_CODE",
    "DISPATCH_RESPONSE_SECONDS_QY",
    "IsHeld",
    "IsReopen",
]

CYCLICAL_FEATURES = [
    "Hour_sin",
    "Hour_cos",
    "DayOfWeek_sin",
    "DayOfWeek_cos",
    "Month_sin",
    "Month_cos",
]

INTERACTION_FEATURES = [
    "Borough_Hour_Avg",
    "Classification_Severity_Avg",
    "DispatchArea_Hour_Avg",
]

CONTEXTUAL_FEATURES = [
    "IsWeekend",
    "IsNight",
    "IsRushHour",
]

TARGET_ENCODED_FEATURES = [
    "Classification_TargetEnc",
    "DispatchArea_TargetEnc",
    "Borough_TargetEnc",
]

ENHANCED_NUMERIC_FEATURES = (
    NUMERIC_FEATURES + CYCLICAL_FEATURES + INTERACTION_FEATURES + CONTEXTUAL_FEATURES + TARGET_ENCODED_FEATURES
)

BASE_CATEGORICAL_FEATURES = [
    "BOROUGH",
    "INCIDENT_CLASSIFICATION",
]

CLASSIFICATION_SOURCE_COLUMNS = [
    "INCIDENT_CLASSIFICATION",
    "FINAL_CALL_TYPE",
    "INITIAL_CALL_TYPE",
]

RESPONSE_TIME_MIN = 0.5
RESPONSE_TIME_MAX = 60
