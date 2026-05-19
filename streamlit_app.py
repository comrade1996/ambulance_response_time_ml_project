from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning

from src.ambulance_response_time_ml.data import (
    make_model_frame,
    make_enhanced_model_frame,
    add_cyclical_features,
    add_contextual_features,
    compute_interaction_lookup,
)
from src.ambulance_response_time_ml.modeling import make_models, residual_analysis


ROOT = Path(__file__).resolve().parent
TRAINING_DATA_CANDIDATES = [
    ROOT / "data" / "processed" / "ems_training_dataset_500000.parquet",
    ROOT / "data" / "processed" / "ems_training_dataset_500000.csv",
    ROOT / "data" / "processed" / "ems_training_dataset_deploy.csv",
    ROOT / "data" / "processed" / "ems_training_dataset_100000.csv",
]
TRAINING_DATA_PATH = next(
    (path for path in TRAINING_DATA_CANDIDATES if path.exists()),
    TRAINING_DATA_CANDIDATES[0],
)
CASE_PREDICTIONS_PATH = ROOT / "outputs" / "reports" / "case_level_model_predictions.csv"
MODEL_COMPARISON_PATH = ROOT / "outputs" / "reports" / "model_comparison.csv"
LINEAR_MODEL_PATH = ROOT / "outputs" / "models" / "linear_regression.joblib"
RANDOM_FOREST_MODEL_PATH = ROOT / "outputs" / "models" / "random_forest_regressor.joblib"
XGBOOST_MODEL_PATH = ROOT / "outputs" / "models" / "xgboost.joblib"
LIGHTGBM_MODEL_PATH = ROOT / "outputs" / "models" / "lightgbm.joblib"
STACKING_MODEL_PATH = ROOT / "outputs" / "models" / "stacking_ensemble.joblib"
TARGET_COL = "Response_Time_Minutes"
SLOW_RESPONSE_LIMIT = 15.0
ORIGINAL_DATA_URL = "https://data.cityofnewyork.us/Public-Safety/EMS-Incident-Dispatch-Data/76xm-jjuj"
ORIGINAL_CSV_URL = "https://data.cityofnewyork.us/api/views/76xm-jjuj/rows.csv?accessType=DOWNLOAD"

FEATURE_COLUMNS = [
    "Hour",
    "DayOfWeek",
    "Month",
    "INITIAL_SEVERITY_LEVEL_CODE",
    "FINAL_SEVERITY_LEVEL_CODE",
    "BOROUGH",
    "INCIDENT_CLASSIFICATION",
    "INCIDENT_DISPATCH_AREA",
]

DAY_NAMES = {
    0: "الاثنين",
    1: "الثلاثاء",
    2: "الأربعاء",
    3: "الخميس",
    4: "الجمعة",
    5: "السبت",
    6: "الأحد",
}

MONTH_NAMES = {
    1: "يناير",
    2: "فبراير",
    3: "مارس",
    4: "أبريل",
    5: "مايو",
    6: "يونيو",
    7: "يوليو",
    8: "أغسطس",
    9: "سبتمبر",
    10: "أكتوبر",
    11: "نوفمبر",
    12: "ديسمبر",
}

TIME_PROFILES = {
    "الصباح الباكر، 06:00": 6,
    "ذروة الصباح، 08:00": 8,
    "منتصف اليوم، 12:00": 12,
    "بعد الظهر، 15:00": 15,
    "ذروة المساء، 18:00": 18,
    "الليل، 22:00": 22,
    "آخر الليل، 02:00": 2,
}

BOROUGH_NAMES = {
    "BRONX": "برونكس",
    "BROOKLYN": "بروكلين",
    "MANHATTAN": "مانهاتن",
    "QUEENS": "كوينز",
    "STATEN ISLAND": "ستاتن آيلاند",
    "RICHMOND / STATEN ISLAND": "ستاتن آيلاند",
}

INCIDENT_CLASSIFICATION_NAMES = {
    "INJURY": "إصابة",
    "INJMIN": "إصابة بسيطة",
    "INJMAJ": "إصابة خطيرة",
    "CARD": "حالة قلب",
    "CARDBR": "توقف قلب أو تنفس",
    "DIFFFC": "صعوبة تنفس",
    "ASTHFC": "ربو أو ضيق تنفس",
    "UNC": "فقدان وعي",
    "UNKNOW": "بلاغ غير معروف",
    "SICK": "حالة مرضية عامة",
    "DRUG": "حالة مرتبطة بمادة أو دواء",
    "EDP": "حالة نفسية أو سلوكية",
    "ALTMEN": "تغير في الحالة الذهنية",
    "ANAPH": "حساسية شديدة",
}


st.set_page_config(
    page_title="توقع زمن استجابة الإسعاف",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1180px;
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
        direction: rtl;
        text-align: right;
    }
    .stApp {
        background: #f6f8f7;
    }
    h1, h2, h3 {
        letter-spacing: 0;
        text-align: right;
    }
    div[data-testid="stSidebar"],
    div[data-testid="stSelectbox"] label,
    div[data-testid="stButton"] button,
    div[data-testid="stMetric"],
    .stTabs {
        direction: rtl;
        text-align: right;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #d9e6e2;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        box-shadow: 0 8px 22px rgba(15, 31, 29, 0.06);
    }
    div[data-testid="stMetricLabel"] p {
        color: #526b66;
        font-size: 0.85rem;
    }
    div[data-testid="stMetricValue"] {
        color: #10201f;
        font-size: 1.55rem;
    }
    .small-note {
        color: #526b66;
        font-size: 0.92rem;
        line-height: 1.45;
    }
    .hero-panel {
        background: linear-gradient(135deg, #0f766e 0%, #155e75 58%, #334155 100%);
        color: #ffffff;
        border-radius: 8px;
        padding: 1.35rem 1.45rem;
        margin-bottom: 1.1rem;
        box-shadow: 0 14px 32px rgba(15, 31, 29, 0.14);
    }
    .hero-panel h1 {
        color: #ffffff;
        margin: 0 0 0.35rem 0;
        font-size: 2rem;
    }
    .hero-panel p {
        color: #dff7f2;
        line-height: 1.65;
        margin: 0;
        max-width: 860px;
    }
    .section-note {
        background: #ffffff;
        border-right: 4px solid #0f766e;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin: 0.6rem 0 1rem;
        color: #334155;
        line-height: 1.65;
    }
    .plain-explain {
        color: #475569;
        font-size: 0.94rem;
        line-height: 1.6;
        margin-top: -0.25rem;
    }
    .recommendation-box {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 8px;
        padding: 1rem 1.1rem;
        color: #7c2d12;
        line-height: 1.65;
    }
    .confidence-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        color: #1e3a8a;
        line-height: 1.6;
    }
    .status-good {
        color: #0f766e;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_training_data() -> pd.DataFrame:
    if TRAINING_DATA_PATH.suffix == ".parquet":
        return pd.read_parquet(TRAINING_DATA_PATH)
    return pd.read_csv(TRAINING_DATA_PATH, low_memory=False)


@st.cache_data(show_spinner=False)
def load_case_predictions() -> pd.DataFrame:
    return pd.read_csv(CASE_PREDICTIONS_PATH, low_memory=False)


@st.cache_data(show_spinner=False)
def load_model_comparison() -> pd.DataFrame:
    comparison = pd.read_csv(MODEL_COMPARISON_PATH)
    comparison["MAE"] = comparison["MAE"].round(2)
    comparison["RMSE"] = comparison["RMSE"].round(2)
    return comparison


@st.cache_resource(show_spinner=False)
def load_models() -> tuple[dict[str, object], str]:
    try:
        models = {}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
            models["Linear Regression"] = joblib.load(LINEAR_MODEL_PATH)
            models["Random Forest Regressor"] = joblib.load(RANDOM_FOREST_MODEL_PATH)
            if XGBOOST_MODEL_PATH.exists():
                models["XGBoost"] = joblib.load(XGBOOST_MODEL_PATH)
            if LIGHTGBM_MODEL_PATH.exists():
                models["LightGBM"] = joblib.load(LIGHTGBM_MODEL_PATH)
            if STACKING_MODEL_PATH.exists():
                models["Stacking Ensemble"] = joblib.load(STACKING_MODEL_PATH)
        return models, "ملفات النماذج المحفوظة"
    except (Exception, InconsistentVersionWarning):
        training_df = load_training_data()
        training_df = training_df.sample(
            n=min(8000, len(training_df)),
            random_state=42,
        )
        X, y, numeric_features, categorical_features = make_enhanced_model_frame(training_df)
        fallback_models = make_models(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            random_state=42,
        )
        models = {}
        for name, model in fallback_models.items():
            try:
                model.fit(X, y)
                models[name] = model
            except Exception:
                continue
        if not models:
            raise RuntimeError("No models could be loaded or trained.")
        return models, "نماذج مدربة داخل التطبيق"


@st.cache_data(show_spinner=False)
def load_enhanced_data() -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    training_df = load_training_data()
    return make_enhanced_model_frame(training_df)


@st.cache_data(show_spinner=False)
def get_interaction_lookup() -> dict:
    training_df = load_training_data()
    return compute_interaction_lookup(training_df)


def require_files() -> None:
    required_paths = [
        TRAINING_DATA_PATH,
        CASE_PREDICTIONS_PATH,
        MODEL_COMPARISON_PATH,
        LINEAR_MODEL_PATH,
        RANDOM_FOREST_MODEL_PATH,
    ]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        st.error("بعض الملفات المطلوبة غير موجودة. شغل أمر التدريب أولا.")
        st.code(
            "python main.py train --data data/raw/EMS_Incident_Dispatch_Data.csv "
            "--sample-rows 500000 --chunksize 50000 --check-cases 20",
            language="bash",
        )
        st.write("الملفات الناقصة:")
        for path in missing_paths:
            st.write(f"- {path.relative_to(ROOT)}")
        st.stop()


def unique_values(df: pd.DataFrame, column: str) -> list:
    values = df[column].dropna().unique().tolist()
    return sorted(values, key=lambda value: str(value))


def most_common_values(df: pd.DataFrame, column: str, limit: int = 60) -> list[str]:
    values = df[column].fillna("").astype(str)
    values = values[values.str.strip() != ""]
    return values.value_counts().head(limit).index.tolist()


def format_minutes(value: float) -> str:
    return f"{float(value):.2f} دقيقة"


def arabic_model_name(name: str) -> str:
    names = {
        "Linear Regression": "الانحدار الخطي",
        "Random Forest Regressor": "الغابة العشوائية",
        "XGBoost": "XGBoost تعزيز التدرج",
        "LightGBM": "LightGBM تعزيز خفيف",
        "Stacking Ensemble": "نموذج التكديس المتقدم",
    }
    return names.get(name, name)


def comparison_for_display(comparison_df: pd.DataFrame) -> pd.DataFrame:
    display_df = comparison_df.copy()
    display_df["Algorithm"] = display_df["Algorithm"].map(arabic_model_name)
    return display_df.rename(
        columns={
            "Algorithm": "الخوارزمية",
            "MAE": "متوسط الخطأ المطلق",
            "RMSE": "جذر متوسط مربع الخطأ",
        }
    )


def format_borough(value: str) -> str:
    arabic_name = BOROUGH_NAMES.get(str(value), str(value))
    return f"{arabic_name} ({value})"


def format_incident(value: str) -> str:
    arabic_name = INCIDENT_CLASSIFICATION_NAMES.get(str(value))
    if arabic_name:
        return f"{arabic_name} ({value})"
    return str(value)


def readable_payload(payload: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "الساعة": payload["Hour"],
                "اليوم": DAY_NAMES[payload["DayOfWeek"]],
                "الشهر": MONTH_NAMES[payload["Month"]],
                "الخطورة الأولية": payload["INITIAL_SEVERITY_LEVEL_CODE"],
                "الخطورة النهائية": payload["FINAL_SEVERITY_LEVEL_CODE"],
                "المنطقة": format_borough(payload["BOROUGH"]),
                "تصنيف البلاغ": format_incident(payload["INCIDENT_CLASSIFICATION"]),
                "منطقة الإرسال": payload["INCIDENT_DISPATCH_AREA"],
                "الرمز البريدي": payload.get("ZIPCODE", ""),
                "تأخير الإرسال (ثواني)": payload.get("DISPATCH_RESPONSE_SECONDS_QY", 0),
                "في الانتظار": "نعم" if payload.get("IsHeld", 0) else "لا",
                "معاد فتحها": "نعم" if payload.get("IsReopen", 0) else "لا",
            }
        ]
    )


def prediction_input_from_payload(payload: dict, model: object) -> pd.DataFrame:
    frame = pd.DataFrame([payload])
    frame = add_cyclical_features(frame)
    frame = add_contextual_features(frame)
    lookup = get_interaction_lookup()
    borough = payload.get("BOROUGH", "")
    hour = payload.get("Hour", 0)
    classification = payload.get("INCIDENT_CLASSIFICATION", "")
    severity = payload.get("INITIAL_SEVERITY_LEVEL_CODE", 0)
    dispatch_area = payload.get("INCIDENT_DISPATCH_AREA", "")
    global_mean = lookup.get("_global_mean", 0.0)
    if "Borough_Hour_Avg" in lookup:
        frame["Borough_Hour_Avg"] = lookup["Borough_Hour_Avg"].get((borough, hour), global_mean)
    if "Classification_Severity_Avg" in lookup:
        frame["Classification_Severity_Avg"] = lookup["Classification_Severity_Avg"].get((classification, severity), global_mean)
    if "DispatchArea_Hour_Avg" in lookup:
        frame["DispatchArea_Hour_Avg"] = lookup["DispatchArea_Hour_Avg"].get((dispatch_area, hour), global_mean)
    if "Classification_TargetEnc" in lookup:
        frame["Classification_TargetEnc"] = lookup["Classification_TargetEnc"].get(classification, global_mean)
    if "DispatchArea_TargetEnc" in lookup:
        frame["DispatchArea_TargetEnc"] = lookup["DispatchArea_TargetEnc"].get(dispatch_area, global_mean)
    if "Borough_TargetEnc" in lookup:
        frame["Borough_TargetEnc"] = lookup["Borough_TargetEnc"].get(borough, global_mean)
    feature_names = list(getattr(model, "feature_names_in_", FEATURE_COLUMNS))
    for col in feature_names:
        if col not in frame.columns:
            frame[col] = 0.0
    preprocessor = getattr(model, "named_steps", {}).get("preprocessor")
    if preprocessor is not None and hasattr(preprocessor, "transformers_"):
        for name, _transformer, columns in preprocessor.transformers_:
            if name == "categorical":
                for col in columns:
                    if col in frame.columns:
                        frame[col] = frame[col].fillna("").astype(str).str.strip()
            if name == "numeric":
                for col in columns:
                    if col in frame.columns:
                        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0)
    return frame[feature_names]


def predict_minutes(model: object, payload: dict) -> float:
    input_frame = prediction_input_from_payload(payload, model)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Found unknown categories.*",
            category=UserWarning,
            module="sklearn.preprocessing._encoders",
        )
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names.*",
            category=UserWarning,
            module="sklearn.utils.validation",
        )
        return float(model.predict(input_frame)[0])


def comparison_chart(values: dict[str, float]) -> None:
    chart_data = pd.DataFrame(
        {"الدقائق": [round(value, 2) for value in values.values()]},
        index=list(values.keys()),
    )
    st.bar_chart(chart_data, height=260)


def common_value(series: pd.Series) -> str:
    values = series.dropna().astype(str)
    if values.empty:
        return ""
    return values.mode().iloc[0]


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def response_summary(training_df: pd.DataFrame) -> dict[str, float]:
    response = training_df[TARGET_COL].dropna()
    return {
        "count": float(len(response)),
        "mean": float(response.mean()),
        "median": float(response.median()),
        "p90": float(response.quantile(0.9)),
        "p95": float(response.quantile(0.95)),
        "slow_share": float((response > SLOW_RESPONSE_LIMIT).mean()),
    }


def best_model_label(comparison_df: pd.DataFrame) -> tuple[str, float]:
    if comparison_df.empty or "MAE" not in comparison_df.columns:
        return "غير متوفر", 0.0
    best = comparison_df.sort_values("MAE", ascending=True).iloc[0]
    return arabic_model_name(str(best["Algorithm"])), float(best["MAE"])


def borough_summary(training_df: pd.DataFrame) -> pd.DataFrame:
    global_mean = training_df[TARGET_COL].mean()
    summary = (
        training_df.groupby("BOROUGH")
        .agg(
            الحالات=(TARGET_COL, "count"),
            المتوسط=(TARGET_COL, "mean"),
            الوسيط=(TARGET_COL, "median"),
            P90=(TARGET_COL, lambda s: s.quantile(0.9)),
            نسبة_أعلى_من_15=(TARGET_COL, lambda s: (s > SLOW_RESPONSE_LIMIT).mean()),
        )
        .reset_index()
    )
    summary["المنطقة"] = summary["BOROUGH"].map(format_borough)
    summary["فرصة_التحسين"] = (
        (summary["المتوسط"] - global_mean).clip(lower=0) * summary["الحالات"]
    )
    return (
        summary[
            [
                "المنطقة",
                "الحالات",
                "المتوسط",
                "الوسيط",
                "P90",
                "نسبة_أعلى_من_15",
                "فرصة_التحسين",
            ]
        ]
        .sort_values(["فرصة_التحسين", "المتوسط"], ascending=False)
        .round(2)
    )


def dispatch_area_summary(training_df: pd.DataFrame, min_cases: int = 80) -> pd.DataFrame:
    global_mean = training_df[TARGET_COL].mean()
    summary = (
        training_df.groupby("INCIDENT_DISPATCH_AREA")
        .agg(
            المنطقة=("BOROUGH", common_value),
            الحالات=(TARGET_COL, "count"),
            المتوسط=(TARGET_COL, "mean"),
            P90=(TARGET_COL, lambda s: s.quantile(0.9)),
            تأخير_الإرسال_دقائق=("DISPATCH_RESPONSE_SECONDS_QY", lambda s: s.mean() / 60),
            نسبة_أعلى_من_15=(TARGET_COL, lambda s: (s > SLOW_RESPONSE_LIMIT).mean()),
        )
        .reset_index()
        .query("الحالات >= @min_cases")
    )
    summary["الأولوية"] = np.select(
        [
            (summary["المتوسط"] >= global_mean + 1.25) & (summary["نسبة_أعلى_من_15"] >= 0.2),
            summary["المتوسط"] >= global_mean + 0.5,
        ],
        ["عالية", "متوسطة"],
        default="مراقبة",
    )
    summary["المنطقة"] = summary["المنطقة"].map(format_borough)
    return (
        summary.rename(
            columns={
                "INCIDENT_DISPATCH_AREA": "منطقة الإرسال",
                "الحالات": "عدد الحالات",
                "المتوسط": "متوسط الاستجابة",
                "P90": "زمن 90% من الحالات",
                "تأخير_الإرسال_دقائق": "متوسط تأخير الإرسال",
                "نسبة_أعلى_من_15": "نسبة أعلى من 15 دقيقة",
            }
        )
        .sort_values(["الأولوية", "متوسط الاستجابة"], ascending=[True, False])
        .round(2)
    )


def incident_summary(training_df: pd.DataFrame, min_cases: int = 80) -> pd.DataFrame:
    summary = (
        training_df.groupby("INCIDENT_CLASSIFICATION")
        .agg(
            الحالات=(TARGET_COL, "count"),
            المتوسط=(TARGET_COL, "mean"),
            P90=(TARGET_COL, lambda s: s.quantile(0.9)),
            نسبة_أعلى_من_15=(TARGET_COL, lambda s: (s > SLOW_RESPONSE_LIMIT).mean()),
        )
        .reset_index()
        .query("الحالات >= @min_cases")
    )
    summary["نوع البلاغ"] = summary["INCIDENT_CLASSIFICATION"].map(format_incident)
    return (
        summary[
            ["نوع البلاغ", "الحالات", "المتوسط", "P90", "نسبة_أعلى_من_15"]
        ]
        .rename(
            columns={
                "الحالات": "عدد الحالات",
                "المتوسط": "متوسط الاستجابة",
                "نسبة_أعلى_من_15": "نسبة أعلى من 15 دقيقة",
            }
        )
        .sort_values("متوسط الاستجابة", ascending=False)
        .round(2)
    )


def model_quality_table(case_df: pd.DataFrame) -> pd.DataFrame:
    actual = case_df["Actual_Response_Time_Minutes"]
    rows = []
    model_arabic = {
        "Linear_Regression": "الانحدار الخطي",
        "Random_Forest": "الغابة العشوائية",
        "XGBoost": "XGBoost",
        "LightGBM": "LightGBM",
        "Stacking_Ensemble": "نموذج التكديس",
    }
    for column in [c for c in case_df.columns if c.endswith("_Predicted_Minutes")]:
        key = column.replace("_Predicted_Minutes", "")
        error = (case_df[column] - actual).abs()
        rows.append(
            {
                "النموذج": model_arabic.get(key, key),
                "متوسط الخطأ": error.mean(),
                "الوسيط": error.median(),
                "ضمن 5 دقائق": (error <= 5).mean(),
                "ضمن 10 دقائق": (error <= 10).mean(),
                "أكبر خطأ": error.max(),
            }
        )
    return pd.DataFrame(rows).sort_values("متوسط الخطأ").round(2)


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-panel">
            <h1>لوحة زمن استجابة الإسعاف</h1>
            <p>
            عرض واضح يشرح مستوى الخدمة الحالي، أين يظهر التأخير، وما الذي يمكن
            مراقبته لتحسين زمن الوصول. كل رقم تحته معنى عملي بدون مصطلحات معقدة.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(
    training_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    model_source: str,
) -> None:
    summary = response_summary(training_df)
    best_name, best_mae = best_model_label(comparison_df)
    st.sidebar.header("ملخص سريع")
    st.sidebar.metric("عدد الحالات", f"{int(summary['count']):,}")
    st.sidebar.metric("متوسط الاستجابة", format_minutes(summary["mean"]))
    st.sidebar.metric("أفضل نموذج", best_name)
    st.sidebar.caption(
        f"متوسط خطأ أفضل نموذج: {format_minutes(best_mae)}. "
        "الأرقام هنا للمساعدة في فهم الأداء، وليست بديلا عن بروتوكولات التشغيل."
    )
    st.sidebar.divider()
    st.sidebar.caption("مصدر النماذج")
    st.sidebar.write(model_source)
    st.sidebar.caption(
        "ابدأ من ملخص الأداء، ثم انتقل إلى فرص التحسين لمعرفة أين تركز الموارد."
    )


def render_case_checker(case_df: pd.DataFrame) -> None:
    st.subheader("فحص حالة حقيقية محفوظة")

    pred_columns = [c for c in case_df.columns if c.endswith("_Predicted_Minutes")]
    model_keys = [c.replace("_Predicted_Minutes", "") for c in pred_columns]
    model_arabic = {
        "Linear_Regression": "الانحدار الخطي",
        "Random_Forest": "الغابة العشوائية",
        "XGBoost": "XGBoost تعزيز التدرج",
        "LightGBM": "LightGBM تعزيز خفيف",
        "Stacking_Ensemble": "نموذج التكديس",
    }

    st.caption(
        f"اختر رقم حالة من نتائج الاختبار المحفوظة، ثم قارن زمن الاستجابة الحقيقي "
        f"مع مخرجات {len(model_keys)} نماذج."
    )

    case_options = case_df.index.tolist()

    def case_label(index: int) -> str:
        row = case_df.loc[index]
        return (
            f"{row['Case_Number']} | {row['BOROUGH']} | "
            f"{row['INCIDENT_CLASSIFICATION']} | الفعلي {format_minutes(row['Actual_Response_Time_Minutes'])}"
        )

    selected_index = st.selectbox(
        "رقم الحالة",
        case_options,
        format_func=case_label,
        help="هذه الحالات مأخوذة من outputs/reports/case_level_model_predictions.csv.",
    )
    row = case_df.loc[selected_index]
    actual = float(row["Actual_Response_Time_Minutes"])

    st.markdown("##### الزمن الفعلي")
    st.metric("زمن الاستجابة الحقيقي", format_minutes(actual))

    st.markdown("##### توقعات النماذج")
    row1_keys = model_keys[:3]
    row2_keys = model_keys[3:]
    cols1 = st.columns(3)
    for i, key in enumerate(row1_keys):
        pred = float(row[f"{key}_Predicted_Minutes"])
        label = model_arabic.get(key, key)
        cols1[i].metric(
            label,
            format_minutes(pred),
            delta=format_minutes(pred - actual),
            delta_color="inverse",
        )
    if row2_keys:
        cols2 = st.columns(3)
        for i, key in enumerate(row2_keys):
            pred = float(row[f"{key}_Predicted_Minutes"])
            label = model_arabic.get(key, key)
            cols2[i].metric(
                label,
                format_minutes(pred),
                delta=format_minutes(pred - actual),
                delta_color="inverse",
            )

    info_cols = st.columns([1.2, 1])
    with info_cols[0]:
        with st.container(border=True):
            st.write("تفاصيل الحالة")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "رقم الحالة": row["Case_Number"],
                            "المنطقة": row["BOROUGH"],
                            "نوع البلاغ": row["INCIDENT_CLASSIFICATION"],
                            "منطقة الإرسال": row["INCIDENT_DISPATCH_AREA"],
                            "الخطورة الأولية": row["INITIAL_SEVERITY_LEVEL_CODE"],
                            "الخطورة النهائية": row["FINAL_SEVERITY_LEVEL_CODE"],
                            "وقت البلاغ": row["INCIDENT_DATETIME"],
                        }
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )
    with info_cols[1]:
        chart_data = {"الفعلي": actual}
        for key in model_keys:
            chart_data[model_arabic.get(key, key)] = float(row[f"{key}_Predicted_Minutes"])
        comparison_chart(chart_data)


def render_new_prediction(training_df: pd.DataFrame, models: dict[str, object]) -> None:
    st.subheader("توقع حالة جديدة")
    st.caption(
        f"اختر وصف الحالة من القوائم. سيتم تشغيل كل النماذج المتاحة وعددها {len(models)} "
        "ثم عرض زمن الاستجابة المتوقع بالدقائق."
    )
    st.info(
        "طريقة القراءة: إذا ظهرت النتيجة 10 دقائق فهذا يعني أن النموذج يتوقع "
        "وصول أول وحدة إسعاف بعد حوالي 10 دقائق من وقت البلاغ."
    )

    borough_options = unique_values(training_df, "BOROUGH")
    incident_options = most_common_values(training_df, "INCIDENT_CLASSIFICATION")
    initial_severity_options = unique_values(training_df, "INITIAL_SEVERITY_LEVEL_CODE")
    final_severity_options = unique_values(training_df, "FINAL_SEVERITY_LEVEL_CODE")

    st.markdown("#### 1. موقع البلاغ")
    location_cols = st.columns(3)
    with location_cols[0]:
        borough = st.selectbox(
            "المنطقة",
            borough_options,
            format_func=format_borough,
            help="اختر المنطقة الرئيسية للحالة.",
        )
    borough_df = training_df[training_df["BOROUGH"] == borough]
    dispatch_options = unique_values(borough_df, "INCIDENT_DISPATCH_AREA")
    with location_cols[1]:
        dispatch_area = st.selectbox(
            "منطقة الإرسال",
            dispatch_options,
            help="تمت تصفية هذه القائمة حسب المنطقة المختارة.",
        )
    zipcode_options = unique_values(borough_df, "ZIPCODE") if "ZIPCODE" in borough_df.columns else ["0"]
    with location_cols[2]:
        zipcode = st.selectbox(
            "الرمز البريدي",
            zipcode_options,
            help="الرمز البريدي لموقع الحادث.",
        )

    st.markdown("#### 2. نوع الحالة ووقتها")
    incident_cols = st.columns(3)
    with incident_cols[0]:
        incident = st.selectbox(
            "تصنيف البلاغ",
            incident_options,
            format_func=format_incident,
            help="القيمة بين القوسين هي الرمز الأصلي في بيانات EMS.",
        )
    with incident_cols[1]:
        time_profile = st.selectbox("الفترة الزمنية", list(TIME_PROFILES.keys()), index=4)
        hour = TIME_PROFILES[time_profile]
    with incident_cols[2]:
        day_of_week = st.selectbox(
            "اليوم",
            list(DAY_NAMES.keys()),
            index=0,
            format_func=lambda value: DAY_NAMES[value],
        )

    detail_cols = st.columns(4)
    with detail_cols[0]:
        month = st.selectbox(
            "الشهر",
            list(MONTH_NAMES.keys()),
            index=4,
            format_func=lambda value: MONTH_NAMES[value],
        )
    with detail_cols[1]:
        initial_severity = st.selectbox(
            "الخطورة الأولية",
            initial_severity_options,
            index=min(3, len(initial_severity_options) - 1),
            help="قيمة رقمية من البيانات الأصلية تمثل تقدير الخطورة عند البلاغ.",
        )
    with detail_cols[2]:
        final_severity = st.selectbox(
            "الخطورة النهائية",
            final_severity_options,
            index=min(3, len(final_severity_options) - 1),
            help="قيمة رقمية من البيانات الأصلية بعد تحديث تصنيف الحالة.",
        )
    with detail_cols[3]:
        dispatch_delay = st.number_input(
            "تأخير الإرسال (ثواني)",
            min_value=0,
            max_value=600,
            value=60,
            step=10,
            help="الوقت بالثواني من استقبال البلاغ حتى إرسال الوحدة.",
        )

    st.markdown("#### 3. معلومات إضافية")
    extra_cols = st.columns(2)
    with extra_cols[0]:
        is_held = st.checkbox("البلاغ كان في الانتظار", value=False, help="هل تم تعليق البلاغ في قائمة الانتظار؟")
    with extra_cols[1]:
        is_reopen = st.checkbox("حالة معاد فتحها", value=False, help="هل تم إعادة فتح الحالة؟")

    payload = {
        "Hour": hour,
        "DayOfWeek": day_of_week,
        "Month": month,
        "INITIAL_SEVERITY_LEVEL_CODE": initial_severity,
        "FINAL_SEVERITY_LEVEL_CODE": final_severity,
        "BOROUGH": borough,
        "INCIDENT_CLASSIFICATION": incident,
        "INCIDENT_DISPATCH_AREA": dispatch_area,
        "DISPATCH_RESPONSE_SECONDS_QY": dispatch_delay,
        "ZIPCODE": zipcode,
        "IsHeld": int(is_held),
        "IsReopen": int(is_reopen),
    }

    with st.container(border=True):
        st.write("ملخص القيم المختارة")
        st.dataframe(readable_payload(payload), hide_index=True, use_container_width=True)

    st.markdown("#### 4. النتيجة المتوقعة")
    predictions = {}
    for model_name, model in models.items():
        predictions[model_name] = predict_minutes(model, payload)

    st.markdown("##### جميع نتائج النماذج المتاحة")
    prediction_table = pd.DataFrame(
        [
            {
                "النموذج": arabic_model_name(model_name),
                "التوقع بالدقائق": round(value, 2),
            }
            for model_name, value in predictions.items()
        ]
    ).sort_values("التوقع بالدقائق")
    st.dataframe(prediction_table, hide_index=True, use_container_width=True)

    pred_items = list(predictions.items())
    row1 = pred_items[:3]
    row2 = pred_items[3:]
    cols1 = st.columns(3)
    for idx, (model_name, pred_value) in enumerate(row1):
        cols1[idx].metric(
            f"توقع {arabic_model_name(model_name)}",
            format_minutes(pred_value),
        )
    if row2:
        cols2 = st.columns(3)
        for idx, (model_name, pred_value) in enumerate(row2):
            cols2[idx].metric(
                f"توقع {arabic_model_name(model_name)}",
                format_minutes(pred_value),
            )
    comparison_chart(
        {arabic_model_name(k): v for k, v in predictions.items()}
    )
    st.markdown(
        '<p class="small-note"><span class="status-good">ملاحظة:</span> '
        "هذا توقع تقريبي وليس قرارا تشغيليا نهائيا. استخدم تبويب الحالة "
        "المحفوظة عندما تريد مقارنة التوقع مع زمن استجابة حقيقي معروف.</p>",
        unsafe_allow_html=True,
    )


def render_shap_analysis(training_df: pd.DataFrame, models: dict[str, object]) -> None:
    st.subheader("تحليل SHAP — تفسير التوقعات")
    st.caption(
        "SHAP يوضح مساهمة كل متغير في التوقع. القيم الموجبة تزيد زمن الاستجابة، "
        "والسالبة تقلله."
    )

    try:
        import shap
    except ImportError:
        st.warning("مكتبة SHAP غير مثبتة. قم بتثبيتها: pip install shap")
        return

    shap_models = {k: v for k, v in models.items() if k != "Linear Regression"}
    model_choice = st.selectbox(
        "اختر النموذج للتحليل",
        list(shap_models.keys()),
        format_func=arabic_model_name,
        key="shap_model_select",
    )
    if not model_choice:
        return

    model = models[model_choice]
    X, y, num_feats, cat_feats = make_enhanced_model_frame(training_df)
    sample_size = min(500, len(X))
    X_sample = X.sample(n=sample_size, random_state=42)

    with st.spinner("جاري حساب قيم SHAP..."):
        try:
            preprocessor = model.named_steps["preprocessor"]
            X_transformed = preprocessor.transform(X_sample)
            feature_names = preprocessor.get_feature_names_out()
            feature_names = [
                n.replace("numeric__", "").replace("categorical__", "")
                for n in feature_names
            ]

            tree_model = model.named_steps["model"]
            if model_choice == "Stacking Ensemble":
                raise ValueError("Stacking uses fallback method")
            explainer = shap.TreeExplainer(tree_model)
            shap_values = explainer.shap_values(X_transformed)

            st.markdown("##### أهم المتغيرات المؤثرة (ملخص SHAP)")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(
                shap_values, X_transformed,
                feature_names=feature_names,
                show=False, max_display=15
            )
            st.pyplot(fig)
            plt.close()

            st.markdown("##### متوسط التأثير المطلق لكل متغير")
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            importance_df = pd.DataFrame({
                "المتغير": feature_names,
                "متوسط التأثير (دقائق)": mean_abs_shap,
            }).sort_values("متوسط التأثير (دقائق)", ascending=False).head(15)
            st.dataframe(importance_df, hide_index=True, use_container_width=True)

        except Exception as e:
            st.info("جاري استخدام طريقة بديلة لحساب SHAP...")
            try:
                import matplotlib.pyplot as plt
                preprocessor = model.named_steps["preprocessor"]
                X_transformed = preprocessor.transform(X_sample)
                feature_names = preprocessor.get_feature_names_out()
                feature_names = [
                    n.replace("numeric__", "").replace("categorical__", "")
                    for n in feature_names
                ]
                X_transformed_df = pd.DataFrame(
                    X_transformed if not hasattr(X_transformed, 'toarray') else X_transformed.toarray(),
                    columns=feature_names
                )
                bg = shap.sample(X_transformed_df, 50)
                inner_model = model.named_steps["model"]
                explainer = shap.KernelExplainer(inner_model.predict, bg)
                shap_values = explainer.shap_values(X_transformed_df.iloc[:100])

                st.markdown("##### أهم المتغيرات المؤثرة (ملخص SHAP)")
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                importance_df = pd.DataFrame({
                    "المتغير": feature_names,
                    "متوسط التأثير (دقائق)": mean_abs_shap,
                }).sort_values("متوسط التأثير (دقائق)", ascending=False).head(15)
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(importance_df["المتغير"][::-1], importance_df["متوسط التأثير (دقائق)"][::-1])
                ax.set_xlabel("Mean |SHAP value| (minutes)")
                ax.set_title(f"Feature Importance - {model_choice}")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                st.dataframe(importance_df, hide_index=True, use_container_width=True)
            except Exception as e2:
                st.error(f"تعذر حساب SHAP: {e2}")

    st.divider()
    st.markdown("##### رسوم الاعتماد الجزئي (Partial Dependence)")
    st.caption("يوضح كيف يتغير التوقع عند تغيير متغير واحد مع تثبيت البقية.")
    try:
        from sklearn.inspection import PartialDependenceDisplay
        import matplotlib.pyplot as plt

        numeric_pdp_features = [f for f in ["Hour", "DayOfWeek", "Month", "INITIAL_SEVERITY_LEVEL_CODE"] if f in X.columns]
        if numeric_pdp_features:
            pdp_feature = st.selectbox(
                "اختر المتغير",
                numeric_pdp_features,
                key="pdp_feature_select",
            )
            fig, ax = plt.subplots(figsize=(8, 4))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                PartialDependenceDisplay.from_estimator(
                    model, X_sample, features=[pdp_feature],
                    kind="average", ax=ax
                )
            ax.set_title(f"Partial Dependence: {pdp_feature}")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    except Exception as e:
        st.warning(f"تعذر رسم الاعتماد الجزئي: {e}")


def render_root_cause_dashboard(training_df: pd.DataFrame) -> None:
    st.subheader("لوحة تحليل الأسباب الجذرية")
    st.caption(
        "تحليل شامل لأنماط التأخير حسب المنطقة والوقت ونوع الحادث والخطورة."
    )

    target_col = "Response_Time_Minutes"
    if target_col not in training_df.columns:
        st.warning("بيانات زمن الاستجابة غير متوفرة.")
        return

    st.markdown("##### 1. متوسط زمن الاستجابة حسب المنطقة والساعة")
    if "BOROUGH" in training_df.columns and "Hour" in training_df.columns:
        heatmap_data = training_df.pivot_table(
            values=target_col, index="BOROUGH", columns="Hour", aggfunc="mean"
        ).round(2)
        st.dataframe(
            heatmap_data.style.background_gradient(cmap="RdYlGn_r", axis=None),
            use_container_width=True,
        )

    st.markdown("##### 2. أسوأ 10 مناطق إرسال من حيث التأخير")
    if "INCIDENT_DISPATCH_AREA" in training_df.columns:
        worst_areas = (
            training_df.groupby("INCIDENT_DISPATCH_AREA")[target_col]
            .agg(["mean", "count", "std"])
            .rename(columns={"mean": "المتوسط", "count": "العدد", "std": "الانحراف"})
            .query("العدد >= 20")
            .sort_values("المتوسط", ascending=False)
            .head(10)
            .round(2)
        )
        st.dataframe(worst_areas, use_container_width=True)

    st.markdown("##### 3. متوسط زمن الاستجابة حسب مستوى الخطورة")
    if "INITIAL_SEVERITY_LEVEL_CODE" in training_df.columns:
        severity_analysis = (
            training_df.groupby("INITIAL_SEVERITY_LEVEL_CODE")[target_col]
            .agg(["mean", "count"])
            .rename(columns={"mean": "المتوسط (دقائق)", "count": "عدد الحالات"})
            .sort_values("المتوسط (دقائق)", ascending=False)
            .round(2)
        )
        st.dataframe(severity_analysis, use_container_width=True)

    st.markdown("##### 4. أنماط التأخير حسب اليوم والفترة")
    if "DayOfWeek" in training_df.columns and "Hour" in training_df.columns:
        training_df_copy = training_df.copy()
        training_df_copy["الفترة"] = pd.cut(
            training_df_copy["Hour"],
            bins=[0, 6, 12, 18, 24],
            labels=["ليل (0-6)", "صباح (6-12)", "ظهر (12-18)", "مساء (18-24)"],
            include_lowest=True,
        )
        period_day = (
            training_df_copy.pivot_table(
                values=target_col, index="DayOfWeek", columns="الفترة", aggfunc="mean"
            )
            .round(2)
        )
        period_day.index = [DAY_NAMES.get(i, str(i)) for i in period_day.index]
        st.dataframe(
            period_day.style.background_gradient(cmap="RdYlGn_r", axis=None),
            use_container_width=True,
        )

    st.markdown("##### 5. أعلى 10 تصنيفات حوادث تأخيرا")
    if "INCIDENT_CLASSIFICATION" in training_df.columns:
        top_incidents = (
            training_df.groupby("INCIDENT_CLASSIFICATION")[target_col]
            .agg(["mean", "count"])
            .rename(columns={"mean": "المتوسط (دقائق)", "count": "عدد الحالات"})
            .query("`عدد الحالات` >= 50")
            .sort_values("المتوسط (دقائق)", ascending=False)
            .head(10)
            .round(2)
        )
        st.dataframe(top_incidents, use_container_width=True)

    st.divider()
    st.markdown("##### 6. تجميع أنماط التأخير (Clustering)")
    st.caption(
        "تجميع الحوادث حسب خصائصها للكشف عن مجموعات طبيعية ذات أنماط تأخير مشتركة."
    )
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler as _SS

        cluster_features = [c for c in ["Hour", "DayOfWeek", "Month", "INITIAL_SEVERITY_LEVEL_CODE"] if c in training_df.columns]
        if cluster_features and target_col in training_df.columns:
            cluster_df = training_df[cluster_features + [target_col]].dropna()
            if len(cluster_df) > 100:
                sample_cluster = cluster_df.sample(n=min(10000, len(cluster_df)), random_state=42)
                scaler = _SS()
                X_cluster = scaler.fit_transform(sample_cluster[cluster_features + [target_col]])
                n_clusters = st.slider("عدد المجموعات", 3, 8, 5, key="n_clusters_slider")
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                sample_cluster = sample_cluster.copy()
                sample_cluster["المجموعة"] = kmeans.fit_predict(X_cluster)

                agg_dict = {
                    "متوسط_الاستجابة": (target_col, "mean"),
                    "عدد_الحالات": (target_col, "count"),
                    "متوسط_الساعة": ("Hour", "mean"),
                }
                if "INITIAL_SEVERITY_LEVEL_CODE" in cluster_features:
                    agg_dict["متوسط_الخطورة"] = ("INITIAL_SEVERITY_LEVEL_CODE", "mean")
                cluster_summary = (
                    sample_cluster.groupby("المجموعة")
                    .agg(**agg_dict)
                    .round(2)
                )
                st.dataframe(cluster_summary, use_container_width=True)
                st.caption(
                    "كل مجموعة تمثل نمط تأخير مختلف. المجموعات ذات المتوسط المرتفع "
                    "تحتاج لتحقيق ومعالجة أسبابها."
                )
    except Exception as e:
        st.warning(f"تعذر إجراء التجميع: {e}")


def render_residual_analysis(training_df: pd.DataFrame, models: dict[str, object]) -> None:
    st.subheader("تحليل الأخطاء والبقايا")
    st.caption(
        "يوضح أين يفشل النموذج في التوقع بدقة ويكشف عن الأنماط المخفية."
    )

    target_col = "Response_Time_Minutes"
    X, y, num_feats, cat_feats = make_enhanced_model_frame(training_df)

    model_choice = st.selectbox(
        "اختر النموذج",
        list(models.keys()),
        format_func=arabic_model_name,
        key="residual_model_select",
    )
    model = models[model_choice]

    with st.spinner("جاري حساب التوقعات..."):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictions = model.predict(X)

    analysis = residual_analysis(y, predictions, X)

    from sklearn.metrics import r2_score
    r2 = r2_score(y, predictions)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("MAE", f"{analysis['Abs_Error'].mean():.2f} دقيقة")
    col2.metric("R²", f"{r2:.4f}")
    col3.metric("أكبر خطأ", f"{analysis['Abs_Error'].max():.2f} دقيقة")
    col4.metric("حالات خطأ > 15 دقيقة", f"{(analysis['Abs_Error'] > 15).sum():,}")
    col5.metric("نسبة الخطأ > 15 دقيقة", f"{(analysis['Abs_Error'] > 15).mean()*100:.1f}%")

    st.markdown("##### توزيع الأخطاء")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(analysis["Residual"], bins=50, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Residual (Actual - Predicted)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Residual Distribution")
    axes[0].axvline(x=0, color="red", linestyle="--")

    axes[1].scatter(analysis["Predicted"], analysis["Residual"], alpha=0.1, s=5)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residual vs Predicted")
    axes[1].axhline(y=0, color="red", linestyle="--")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("##### الحالات الأكثر خطأ (أعلى 20)")
    worst_cases = analysis.nlargest(20, "Abs_Error")[
        ["Actual", "Predicted", "Residual", "Abs_Error"]
        + [c for c in ["BOROUGH", "INCIDENT_CLASSIFICATION", "Hour"] if c in analysis.columns]
    ].round(2)
    st.dataframe(worst_cases, use_container_width=True)

    if "BOROUGH" in analysis.columns:
        st.markdown("##### متوسط الخطأ المطلق حسب المنطقة")
        borough_errors = (
            analysis.groupby("BOROUGH")["Abs_Error"]
            .mean()
            .sort_values(ascending=False)
            .round(2)
        )
        st.bar_chart(borough_errors)

    if "Hour" in analysis.columns:
        st.markdown("##### متوسط الخطأ المطلق حسب الساعة")
        hour_errors = (
            analysis.groupby("Hour")["Abs_Error"]
            .mean()
            .sort_values(ascending=False)
            .round(2)
        )
        st.line_chart(hour_errors.sort_index())

    st.divider()
    st.markdown("##### تحليل مجزأ (Segmented Analysis)")
    st.caption("تقسيم الأداء حسب شرائح البيانات لكشف نقاط الضعف المحددة.")

    if "BOROUGH" in analysis.columns and "Hour" in analysis.columns:
        segment_col = st.selectbox(
            "اختر التقسيم",
            ["BOROUGH", "INCIDENT_CLASSIFICATION", "Hour", "DayOfWeek"],
            key="segment_select",
            format_func=lambda x: {
                "BOROUGH": "المنطقة",
                "INCIDENT_CLASSIFICATION": "نوع الحادث",
                "Hour": "الساعة",
                "DayOfWeek": "اليوم",
            }.get(x, x),
        )
        if segment_col in analysis.columns:
            segmented = (
                analysis.groupby(segment_col)
                .agg(
                    عدد_الحالات=("Abs_Error", "count"),
                    متوسط_الخطأ=("Abs_Error", "mean"),
                    أكبر_خطأ=("Abs_Error", "max"),
                    متوسط_الفعلي=("Actual", "mean"),
                    متوسط_التوقع=("Predicted", "mean"),
                )
                .sort_values("متوسط_الخطأ", ascending=False)
                .round(2)
            )
            st.dataframe(segmented, use_container_width=True)


def render_causal_analysis(training_df: pd.DataFrame) -> None:
    st.subheader("تحليل سببي أولي")
    st.caption(
        "تحليل يحاول تحديد العلاقات السببية وليس الارتباطات فقط."
    )

    target_col = "Response_Time_Minutes"
    if target_col not in training_df.columns:
        st.warning("بيانات زمن الاستجابة غير متوفرة.")
        return

    st.markdown("##### 1. تأثير الخطورة على زمن الاستجابة (مع ضبط المنطقة)")
    if all(c in training_df.columns for c in ["INITIAL_SEVERITY_LEVEL_CODE", "BOROUGH"]):
        st.write(
            "نقارن متوسط زمن الاستجابة لكل مستوى خطورة داخل كل منطقة على حدة، "
            "مما يعزل تأثير الخطورة عن تأثير الموقع الجغرافي."
        )
        controlled = (
            training_df.groupby(["BOROUGH", "INITIAL_SEVERITY_LEVEL_CODE"])[target_col]
            .mean()
            .unstack(fill_value=0)
            .round(2)
        )
        st.dataframe(
            controlled.style.background_gradient(cmap="RdYlGn_r", axis=None),
            use_container_width=True,
        )

    st.markdown("##### 2. فرق-الفروق: مقارنة فترات الذروة وخارجها")
    if "Hour" in training_df.columns and "BOROUGH" in training_df.columns:
        st.write(
            "نقارن الفرق في زمن الاستجابة بين ساعات الذروة (7-9 صباحا، 5-7 مساء) "
            "والساعات العادية لكل منطقة."
        )
        df_copy = training_df.copy()
        df_copy["ذروة"] = df_copy["Hour"].isin([7, 8, 9, 17, 18, 19])
        diff_in_diff = (
            df_copy.groupby(["BOROUGH", "ذروة"])[target_col]
            .mean()
            .unstack(fill_value=0)
            .round(2)
        )
        diff_in_diff.columns = ["خارج الذروة", "ذروة"]
        diff_in_diff["الفرق (دقائق)"] = (
            diff_in_diff["ذروة"] - diff_in_diff["خارج الذروة"]
        ).round(2)
        st.dataframe(diff_in_diff, use_container_width=True)
        st.info(
            "إذا كان الفرق موجبا، فإن ساعات الذروة تزيد زمن الاستجابة "
            "في تلك المنطقة — وهذا قد يشير إلى تأثير الازدحام المروري."
        )

    st.markdown("##### 3. تحليل الاتجاه الزمني")
    if "Month" in training_df.columns:
        monthly_trend = (
            training_df.groupby("Month")[target_col]
            .agg(["mean", "count"])
            .rename(columns={"mean": "المتوسط", "count": "العدد"})
            .round(2)
        )
        monthly_trend.index = [MONTH_NAMES.get(m, str(m)) for m in monthly_trend.index]
        st.line_chart(monthly_trend["المتوسط"])
        st.dataframe(monthly_trend, use_container_width=True)

    st.markdown("##### 4. توصيات سببية")
    st.markdown("""
    بناء على التحليل:
    - **تأثير الذروة المرورية**: المناطق التي تظهر فرقا كبيرا بين الذروة وخارجها
      تحتاج إلى نقاط تمركز إضافية خلال ساعات الذروة.
    - **تأثير الخطورة**: إذا كانت الحالات عالية الخطورة لا تحصل على استجابة أسرع،
      فهناك مشكلة في بروتوكول الأولوية.
    - **التغير الموسمي**: الأشهر ذات المتوسط المرتفع تحتاج لمراجعة
      (طقس سيء، أحداث خاصة، نقص موظفين).
    """)


def percent_display(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    display_df = df.copy()
    for column in columns:
        if column in display_df.columns:
            display_df[column] = (display_df[column] * 100).round(1)
    return display_df


def render_executive_overview(
    training_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    case_df: pd.DataFrame,
) -> None:
    summary = response_summary(training_df)
    best_name, best_mae = best_model_label(comparison_df)
    quality = model_quality_table(case_df)
    best_quality = quality.iloc[0] if not quality.empty else None

    st.subheader("ملخص الأداء")
    st.markdown(
        """
        <div class="section-note">
        اقرأ هذه الصفحة بهذا الترتيب: كم يستغرق الوصول عادة، أين تظهر الحالات
        الأبطأ، ثم هل التوقعات قريبة من الواقع بما يكفي للمقارنة بين السيناريوهات.
        </div>
        """,
        unsafe_allow_html=True,
    )

    kpi_cols = st.columns(5)
    kpi_cols[0].metric("الحالات المستخدمة", f"{int(summary['count']):,}")
    kpi_cols[1].metric("متوسط الاستجابة", format_minutes(summary["mean"]))
    kpi_cols[2].metric("الوسيط", format_minutes(summary["median"]))
    kpi_cols[3].metric("زمن 90% من الحالات", format_minutes(summary["p90"]))
    kpi_cols[4].metric("أكثر من 15 دقيقة", format_percent(summary["slow_share"]))

    st.markdown(
        f"""
        <p class="plain-explain">
        الوسيط يعني أن نصف الحالات وصلت أسرع من هذا الرقم. زمن 90% من الحالات
        يعني أن 90 من كل 100 حالة وصلت خلال هذا الوقت أو أقل، بينما أبطأ 10 حالات
        أخذت وقتا أطول. أفضل نموذج حاليا هو <b>{best_name}</b> بمتوسط خطأ تقريبي
        <b>{format_minutes(best_mae)}</b>.
        </p>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("##### الأداء حسب المنطقة")
        borough_df = borough_summary(training_df)
        display_df = percent_display(borough_df, ["نسبة_أعلى_من_15"])
        st.dataframe(
            display_df.rename(
                columns={
                    "الحالات": "عدد الحالات",
                    "المتوسط": "متوسط الاستجابة",
                    "الوسيط": "الوسيط",
                    "P90": "زمن 90% من الحالات",
                    "نسبة_أعلى_من_15": "نسبة أعلى من 15 دقيقة %",
                    "فرصة_التحسين": "دقائق قابلة للتحسين",
                }
            ).head(8),
            hide_index=True,
            use_container_width=True,
        )
    with right:
        st.markdown("##### ساعات الضغط الأعلى")
        hourly = (
            training_df.groupby("Hour")[TARGET_COL]
            .mean()
            .sort_index()
            .round(2)
        )
        st.line_chart(hourly, height=260)
        peak_hour = int(hourly.idxmax())
        st.info(
            f"أعلى متوسط استجابة يظهر حول الساعة {peak_hour}:00. "
            "هذا يساعد على مراجعة التوزيع المناوبي ونقاط التمركز."
        )

    st.markdown("##### الخلاصة")
    if best_quality is not None:
        st.markdown(
            f"""
            <div class="recommendation-box">
            النموذج مناسب للمقارنة بين السيناريوهات وفهم المخاطر، وليس وعدا بزمن
            وصول دقيق لكل حالة منفردة. أفضل نتيجة تحقق متوسط خطأ
            <b>{format_minutes(float(best_quality["متوسط الخطأ"]))}</b>، و
            <b>{format_percent(float(best_quality["ضمن 10 دقائق"]))}</b>
            من حالات الاختبار كانت ضمن 10 دقائق من الزمن الحقيقي.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_improvement_opportunities(training_df: pd.DataFrame) -> None:
    st.subheader("أين يمكن تحسين الاستجابة؟")
    st.markdown(
        """
        <div class="section-note">
        هذا القسم يرتب الأماكن والأوقات وأنواع البلاغات حسب زمن الاستجابة. الهدف
        هو معرفة أين تظهر المشكلة أولا، ثم فتح التفاصيل عند الحاجة.
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_area, tab_incident, tab_time = st.tabs(
        ["مناطق الإرسال", "أنواع البلاغات", "الوقت والذروة"]
    )
    with tab_area:
        area_df = dispatch_area_summary(training_df).head(15)
        display_df = percent_display(area_df, ["نسبة أعلى من 15 دقيقة"])
        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True,
        )
        if not area_df.empty:
            top = area_df.iloc[0]
            st.markdown(
                f"""
                <div class="recommendation-box">
                أولوية عملية: راجع منطقة الإرسال <b>{top["منطقة الإرسال"]}</b>
                لأنها تسجل متوسط استجابة <b>{format_minutes(float(top["متوسط الاستجابة"]))}</b>.
                راجع التمركز، التغطية وقت الذروة، وتأخير الإرسال في هذه المنطقة.
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_incident:
        incident_df = incident_summary(training_df).head(12)
        display_df = percent_display(incident_df, ["نسبة أعلى من 15 دقيقة"])
        st.dataframe(
            display_df.rename(columns={"P90": "زمن 90% من الحالات"}),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "الجدول يعرض أنواع البلاغات ذات متوسط استجابة أعلى، بشرط وجود عدد كاف من الحالات."
        )

    with tab_time:
        by_hour = (
            training_df.groupby("Hour")
            .agg(
                عدد_الحالات=(TARGET_COL, "count"),
                متوسط_الاستجابة=(TARGET_COL, "mean"),
                نسبة_أعلى_من_15=(TARGET_COL, lambda s: (s > SLOW_RESPONSE_LIMIT).mean()),
            )
            .reset_index()
            .sort_values("متوسط_الاستجابة", ascending=False)
            .head(10)
            .round(2)
        )
        st.dataframe(
            percent_display(by_hour, ["نسبة_أعلى_من_15"]).rename(
                columns={
                    "Hour": "الساعة",
                    "نسبة_أعلى_من_15": "نسبة أعلى من 15 دقيقة %",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.markdown(
            """
            <p class="plain-explain">
            طريقة القراءة: اختر أعلى ساعتين أو ثلاث ساعات، ثم قارنها مع جدول
            المناوبات ونقاط تمركز الوحدات. إذا تكرر التأخير في نفس الفترة، فهذا
            يعني أن الوقت نفسه قد يكون جزءا من المشكلة.
            </p>
            """,
            unsafe_allow_html=True,
        )


def render_prediction_experience(
    training_df: pd.DataFrame,
    models: dict[str, object],
) -> None:
    st.subheader("توقع حالة جديدة")
    st.markdown(
        """
        <div class="section-note">
        اختر وصف الحالة، ثم شاهد توقع كل نموذج متاح. المقارنة بين النماذج تساعد
        على معرفة هل التوقع ثابت تقريبا أم أن النماذج تختلف كثيرا.
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_new_prediction(training_df, models)


def render_model_confidence(
    case_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
) -> None:
    st.subheader("هل التوقعات قريبة من الواقع؟")
    st.markdown(
        """
        <div class="section-note">
        هذا القسم يشرح جودة النموذج بطريقة مفهومة: كم دقيقة يخطئ في المتوسط؟
        وكم مرة يكون قريباً من الزمن الحقيقي؟
        </div>
        """,
        unsafe_allow_html=True,
    )

    quality = model_quality_table(case_df)
    if quality.empty:
        st.warning("لا توجد بيانات اختبار كافية لعرض جودة النموذج.")
        return

    display_quality = percent_display(quality, ["ضمن 5 دقائق", "ضمن 10 دقائق"])
    st.dataframe(
        display_quality.rename(
            columns={
                "ضمن 5 دقائق": "ضمن 5 دقائق %",
                "ضمن 10 دقائق": "ضمن 10 دقائق %",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    best = quality.iloc[0]
    st.markdown(
        f"""
        <div class="confidence-box">
        أفضل نموذج في بيانات الاختبار هو <b>{best["النموذج"]}</b>. متوسط الخطأ
        لديه <b>{format_minutes(float(best["متوسط الخطأ"]))}</b>. معنى ذلك:
        إذا توقع النموذج 10 دقائق، فالنتيجة الفعلية قد تختلف بعدة دقائق، لذلك
        يستخدم لفهم النمط العام وتحديد المخاطر، وليس لتحديد زمن وصول مضمون لحالة فردية.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("##### مقارنة MAE و RMSE")
    st.dataframe(
        comparison_for_display(comparison_df),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "MAE هو متوسط الخطأ بالدقائق. RMSE يعاقب الأخطاء الكبيرة، لذلك ارتفاعه يعني وجود حالات يصعب توقعها."
    )


def render_data_context(training_df: pd.DataFrame) -> None:
    st.subheader("البيانات المستخدمة: القوة، المشاكل، والحدود")
    parsed_dates = pd.to_datetime(training_df["INCIDENT_DATETIME"], errors="coerce")
    valid_dates = parsed_dates.dropna()
    missing = (
        training_df.isna()
        .mean()
        .sort_values(ascending=False)
        .head(8)
        .reset_index()
    )
    missing.columns = ["الحقل", "نسبة القيم الناقصة"]

    top_cols = st.columns(4)
    top_cols[0].metric("عدد السجلات", f"{len(training_df):,}")
    top_cols[1].metric("عدد الحقول", str(len(training_df.columns)))
    top_cols[2].metric("المناطق", str(training_df["BOROUGH"].nunique()))
    top_cols[3].metric("مناطق الإرسال", str(training_df["INCIDENT_DISPATCH_AREA"].nunique()))

    if not valid_dates.empty:
        st.caption(
            f"الفترة الزمنية في ملف العرض: من {valid_dates.min().date()} إلى {valid_dates.max().date()}."
        )

    st.markdown("##### ما هي البيانات؟")
    st.markdown(
        f"""
        مصدر البيانات الأصلي:
        <a href="{ORIGINAL_DATA_URL}" target="_blank">NYC Open Data - EMS Incident Dispatch Data</a>
        &nbsp;|&nbsp;
        <a href="{ORIGINAL_CSV_URL}" target="_blank">تحميل CSV الأصلي</a>
        """,
        unsafe_allow_html=True,
    )
    st.write(
        "استخدمنا سجلات حوادث EMS بعد تنظيفها وتجهيزها للنموذج. كل سجل يمثل بلاغا "
        "ويتضمن وقت البلاغ، وقت الوصول الأول للموقع، المنطقة، منطقة الإرسال، نوع "
        "البلاغ، مستوى الخطورة، وتأخير الإرسال. المتغير المستهدف هو زمن الاستجابة "
        "بالدقائق من وقت البلاغ حتى أول وصول للموقع."
    )

    strengths, issues = st.columns(2)
    with strengths:
        st.markdown("##### الجوانب القوية")
        st.markdown(
            """
            - حجم بيانات مناسب للعرض والتحليل، وليس مجرد عينة صغيرة.
            - يحتوي على وقت ومكان ونوع بلاغ وخطورة، وهي عوامل عملية سهلة الفهم.
            - يحتوي على الزمن الحقيقي للاستجابة، لذلك يمكن قياس النموذج مقابل الواقع.
            - يسمح بتحديد مناطق وساعات وأنواع بلاغات تحتاج مراجعة تشغيلية.
            """
        )
    with issues:
        st.markdown("##### المشاكل والقيود")
        st.markdown(
            """
            - البيانات تاريخية؛ إذا تغيرت الموارد أو السياسات فقد يتغير الأداء الفعلي.
            - بعض الحقول قد تكون ناقصة أو غير متسقة، مثل رموز التصنيف أو المنطقة.
            - النموذج لا يعرف كل العوامل الخارجية مثل الطقس، الازدحام المباشر، توفر الوحدات، أو إغلاق الطرق.
            - النتائج احتمالية وداعمة للقرار، وليست وعدا بزمن وصول لحالة فردية.
            """
        )

    st.markdown("##### جودة البيانات")
    st.dataframe(
        percent_display(missing, ["نسبة القيم الناقصة"]).rename(
            columns={"نسبة القيم الناقصة": "نسبة القيم الناقصة %"}
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("##### ماذا نحتاج لتحسين النسخة القادمة؟")
    st.markdown(
        """
        - إضافة حالة توفر الوحدات وقت البلاغ.
        - إضافة بيانات الطقس والازدحام وأحداث المدينة.
        - ربط الموقع بإحداثيات أو مناطق خدمة أدق من المنطقة العامة.
        - تحديث النموذج بشكل دوري حتى يعكس الأداء الحالي وليس التاريخي فقط.
        """
    )


def render_technical_details(
    training_df: pd.DataFrame,
    case_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    models: dict[str, object],
) -> None:
    st.subheader("التحليل التقني")
    st.caption(
        "هذا التبويب يجمع التحليلات التقنية المتقدمة في مكان واحد حتى تبقى الصفحة الرئيسية بسيطة وواضحة."
    )
    tech_tabs = st.tabs(
        [
            "حالة محفوظة",
            "الأسباب الجذرية",
            "SHAP",
            "تحليل الأخطاء",
            "تحليل سببي",
            "النماذج والبيانات",
        ]
    )
    with tech_tabs[0]:
        render_case_checker(case_df)
    with tech_tabs[1]:
        render_root_cause_dashboard(training_df)
    with tech_tabs[2]:
        render_shap_analysis(training_df, models)
    with tech_tabs[3]:
        render_residual_analysis(training_df, models)
    with tech_tabs[4]:
        render_causal_analysis(training_df)
    with tech_tabs[5]:
        st.markdown("##### مقارنة النماذج")
        st.dataframe(
            comparison_for_display(comparison_df),
            hide_index=True,
            use_container_width=True,
        )
        st.markdown("##### النماذج المتوفرة")
        for name in models.keys():
            st.write(f"- {arabic_model_name(name)} ({name})")
        st.markdown("##### عينة من البيانات")
        st.dataframe(training_df.head(20), use_container_width=True)


def main() -> None:
    require_files()
    training_df = load_training_data()
    case_df = load_case_predictions()
    comparison_df = load_model_comparison()
    models, model_source = load_models()

    render_sidebar(training_df, comparison_df, model_source)

    render_hero()

    tab_exec, tab_ops, tab_predict, tab_confidence, tab_data, tab_technical = st.tabs(
        [
            "ملخص الأداء",
            "فرص التحسين",
            "توقع حالة جديدة",
            "دقة التوقع",
            "البيانات والحدود",
            "التحليل التقني",
        ]
    )
    with tab_exec:
        render_executive_overview(training_df, comparison_df, case_df)
    with tab_ops:
        render_improvement_opportunities(training_df)
    with tab_predict:
        render_prediction_experience(training_df, models)
    with tab_confidence:
        render_model_confidence(case_df, comparison_df)
    with tab_data:
        render_data_context(training_df)
    with tab_technical:
        render_technical_details(training_df, case_df, comparison_df, models)


if __name__ == "__main__":
    main()
