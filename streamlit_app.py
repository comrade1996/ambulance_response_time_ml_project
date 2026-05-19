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
TRAINING_DATA_PATH = ROOT / "data" / "processed" / "ems_training_dataset_deploy.csv"
CASE_PREDICTIONS_PATH = ROOT / "outputs" / "reports" / "case_level_model_predictions.csv"
MODEL_COMPARISON_PATH = ROOT / "outputs" / "reports" / "model_comparison.csv"
LINEAR_MODEL_PATH = ROOT / "outputs" / "models" / "linear_regression.joblib"
RANDOM_FOREST_MODEL_PATH = ROOT / "outputs" / "models" / "random_forest_regressor.joblib"
XGBOOST_MODEL_PATH = ROOT / "outputs" / "models" / "xgboost.joblib"
LIGHTGBM_MODEL_PATH = ROOT / "outputs" / "models" / "lightgbm.joblib"
STACKING_MODEL_PATH = ROOT / "outputs" / "models" / "stacking_ensemble.joblib"

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
        background: linear-gradient(180deg, #f7faf9 0%, #eef5f2 100%);
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
            warnings.simplefilter("error", InconsistentVersionWarning)
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
        training_df = pd.read_csv(TRAINING_DATA_PATH, low_memory=False)
        X, y, numeric_features, categorical_features = make_enhanced_model_frame(training_df)
        fallback_models = make_models(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            random_state=42,
        )
        models = {
            name: fallback_models[name]
            for name in ["Linear Regression", "Random Forest Regressor"]
            if name in fallback_models
        }
        for model in models.values():
            model.fit(X, y)
        return models, "إعادة تدريب داخل بيئة Streamlit"


@st.cache_data(show_spinner=False)
def load_enhanced_data() -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    training_df = pd.read_csv(TRAINING_DATA_PATH, low_memory=False)
    return make_enhanced_model_frame(training_df)


@st.cache_data(show_spinner=False)
def get_interaction_lookup() -> dict:
    training_df = pd.read_csv(TRAINING_DATA_PATH, low_memory=False)
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
            "--sample-rows 100000 --chunksize 50000 --check-cases 20",
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
        return float(model.predict(input_frame)[0])


def comparison_chart(values: dict[str, float]) -> None:
    chart_data = pd.DataFrame(
        {"الدقائق": [round(value, 2) for value in values.values()]},
        index=list(values.keys()),
    )
    st.bar_chart(chart_data, height=260)


def render_sidebar(
    training_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    model_source: str,
) -> None:
    st.sidebar.header("حالة المشروع")
    st.sidebar.metric("بيانات التدريب", f"{len(training_df):,} سجل")
    st.sidebar.metric("عدد النماذج", str(len(comparison_df)))
    st.sidebar.metric("مصدر النماذج", model_source)
    st.sidebar.caption(
        "MAE يعني متوسط الخطأ بالدقائق. RMSE يعاقب الأخطاء الكبيرة أكثر. "
        "كلما كانت القيم أقل كان النموذج أفضل."
    )
    st.sidebar.dataframe(
        comparison_for_display(comparison_df),
        hide_index=True,
        use_container_width=True,
    )
    st.sidebar.divider()
    st.sidebar.caption("الملفات المستخدمة في تطبيق الويب")
    st.sidebar.code(
        "data/processed/ems_training_dataset_500000.csv\n"
        "outputs/models/linear_regression.joblib\n"
        "outputs/models/random_forest_regressor.joblib\n"
        "outputs/models/xgboost.joblib\n"
        "outputs/models/lightgbm.joblib\n"
        "outputs/models/stacking_ensemble.joblib\n"
        "outputs/reports/case_level_model_predictions.csv",
        language="text",
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
        f"اختر وصف الحالة من القوائم. التطبيق يشغل {len(models)} نماذج تلقائيا ويعرض "
        "زمن الاستجابة المتوقع بالدقائق."
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


def main() -> None:
    require_files()
    training_df = load_training_data()
    case_df = load_case_predictions()
    comparison_df = load_model_comparison()
    models, model_source = load_models()

    render_sidebar(training_df, comparison_df, model_source)

    st.title("توقع زمن استجابة الإسعاف")
    st.caption(
        "تطبيق ويب متقدم لتوقع زمن الاستجابة وتحليل الأسباب الجذرية "
        "باستخدام خمس خوارزميات تعلم آلي."
    )

    tab_new, tab_case, tab_root, tab_shap, tab_residual, tab_causal, tab_data = st.tabs(
        [
            "توقع جديد",
            "فحص حالة محفوظة",
            "الأسباب الجذرية",
            "تحليل SHAP",
            "تحليل الأخطاء",
            "تحليل سببي",
            "ملخص البيانات",
        ]
    )
    with tab_new:
        render_new_prediction(training_df, models)
    with tab_case:
        render_case_checker(case_df)
    with tab_root:
        render_root_cause_dashboard(training_df)
    with tab_shap:
        render_shap_analysis(training_df, models)
    with tab_residual:
        render_residual_analysis(training_df, models)
    with tab_causal:
        render_causal_analysis(training_df)
    with tab_data:
        st.subheader("ملخص البيانات والنماذج")
        st.write(
            f"يستخدم التطبيق مجموعة بيانات محفوظة تحتوي على {len(training_df):,} سجل، "
            f"مع {len(models)} نماذج مدربة."
        )
        st.dataframe(
            comparison_for_display(comparison_df),
            hide_index=True,
            use_container_width=True,
        )
        st.markdown("##### النماذج المتوفرة")
        for name in models.keys():
            st.write(f"- {arabic_model_name(name)} ({name})")
        st.write("معاينة بيانات التدريب")
        st.dataframe(training_df.head(20), use_container_width=True)


if __name__ == "__main__":
    main()
