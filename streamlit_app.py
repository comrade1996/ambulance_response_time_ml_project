from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.ambulance_response_time_ml.data import make_model_frame
from src.ambulance_response_time_ml.modeling import make_models


ROOT = Path(__file__).resolve().parent
TRAINING_DATA_PATH = ROOT / "data" / "processed" / "ems_training_dataset_100000.csv"
CASE_PREDICTIONS_PATH = ROOT / "outputs" / "reports" / "case_level_model_predictions.csv"
MODEL_COMPARISON_PATH = ROOT / "outputs" / "reports" / "model_comparison.csv"
LINEAR_MODEL_PATH = ROOT / "outputs" / "models" / "linear_regression.joblib"
RANDOM_FOREST_MODEL_PATH = ROOT / "outputs" / "models" / "random_forest_regressor.joblib"

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
        return (
            {
                "Linear Regression": joblib.load(LINEAR_MODEL_PATH),
                "Random Forest Regressor": joblib.load(RANDOM_FOREST_MODEL_PATH),
            },
            "ملفات النماذج المحفوظة",
        )
    except Exception:
        training_df = pd.read_csv(TRAINING_DATA_PATH, low_memory=False)
        X, y, numeric_features, categorical_features = make_model_frame(training_df)
        models = make_models(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            random_state=42,
        )
        for model in models.values():
            model.fit(X, y)
        return models, "إعادة تدريب داخل بيئة Streamlit"


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
            }
        ]
    )


def prediction_input_from_payload(payload: dict, model: object) -> pd.DataFrame:
    frame = pd.DataFrame([payload])
    feature_names = list(getattr(model, "feature_names_in_", FEATURE_COLUMNS))
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
    st.sidebar.metric("عدد النماذج", "2")
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
        "data/processed/ems_training_dataset_100000.csv\n"
        "outputs/models/linear_regression.joblib\n"
        "outputs/models/random_forest_regressor.joblib\n"
        "outputs/reports/case_level_model_predictions.csv",
        language="text",
    )


def render_case_checker(case_df: pd.DataFrame) -> None:
    st.subheader("فحص حالة حقيقية محفوظة")
    st.caption(
        "اختر رقم حالة من نتائج الاختبار المحفوظة، ثم قارن زمن الاستجابة الحقيقي "
        "مع مخرجات النموذجين."
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
    linear_prediction = float(row["Linear_Regression_Predicted_Minutes"])
    rf_prediction = float(row["Random_Forest_Predicted_Minutes"])

    metric_cols = st.columns(3)
    metric_cols[0].metric("زمن الاستجابة الحقيقي", format_minutes(actual))
    metric_cols[1].metric(
        "الانحدار الخطي",
        format_minutes(linear_prediction),
        delta=format_minutes(linear_prediction - actual),
        delta_color="inverse",
    )
    metric_cols[2].metric(
        "الغابة العشوائية",
        format_minutes(rf_prediction),
        delta=format_minutes(rf_prediction - actual),
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
        comparison_chart(
            {
                "الفعلي": actual,
                "الخطي": linear_prediction,
                "الغابة العشوائية": rf_prediction,
            }
        )


def render_new_prediction(training_df: pd.DataFrame, models: dict[str, object]) -> None:
    st.subheader("توقع حالة جديدة")
    st.caption(
        "اختر وصف الحالة من القوائم. التطبيق يشغل النموذجين تلقائيا ويعرض "
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
    location_cols = st.columns(2)
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

    detail_cols = st.columns(3)
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

    payload = {
        "Hour": hour,
        "DayOfWeek": day_of_week,
        "Month": month,
        "INITIAL_SEVERITY_LEVEL_CODE": initial_severity,
        "FINAL_SEVERITY_LEVEL_CODE": final_severity,
        "BOROUGH": borough,
        "INCIDENT_CLASSIFICATION": incident,
        "INCIDENT_DISPATCH_AREA": dispatch_area,
    }

    with st.container(border=True):
        st.write("ملخص القيم المختارة")
        st.dataframe(readable_payload(payload), hide_index=True, use_container_width=True)

    st.markdown("#### 3. النتيجة المتوقعة")
    linear_prediction = predict_minutes(models["Linear Regression"], payload)
    rf_prediction = predict_minutes(models["Random Forest Regressor"], payload)

    result_cols = st.columns(2)
    result_cols[0].metric("توقع الانحدار الخطي", format_minutes(linear_prediction))
    result_cols[1].metric("توقع الغابة العشوائية", format_minutes(rf_prediction))
    comparison_chart(
        {
            "الخطي": linear_prediction,
            "الغابة العشوائية": rf_prediction,
        }
    )
    st.markdown(
        '<p class="small-note"><span class="status-good">ملاحظة:</span> '
        "هذا توقع تقريبي وليس قرارا تشغيليا نهائيا. استخدم تبويب الحالة "
        "المحفوظة عندما تريد مقارنة التوقع مع زمن استجابة حقيقي معروف.</p>",
        unsafe_allow_html=True,
    )


def main() -> None:
    require_files()
    training_df = load_training_data()
    case_df = load_case_predictions()
    comparison_df = load_model_comparison()
    models, model_source = load_models()

    render_sidebar(training_df, comparison_df, model_source)

    st.title("توقع زمن استجابة الإسعاف")
    st.caption(
        "تطبيق ويب تجريبي لمقارنة توقعات الانحدار الخطي والغابة العشوائية "
        "لزمن الاستجابة اعتمادا على بيانات الإسعاف."
    )

    tab_new, tab_case, tab_data = st.tabs(
        ["توقع جديد", "فحص حالة محفوظة", "ملخص البيانات والنماذج"]
    )
    with tab_new:
        render_new_prediction(training_df, models)
    with tab_case:
        render_case_checker(case_df)
    with tab_data:
        st.subheader("ملخص البيانات والنماذج")
        st.write(
            "يستخدم التطبيق مجموعة بيانات محفوظة تحتوي على 100,000 سجل، مع ملفي "
            "النموذجين الناتجين من عملية التدريب."
        )
        st.dataframe(
            comparison_for_display(comparison_df),
            hide_index=True,
            use_container_width=True,
        )
        st.write("معاينة بيانات التدريب")
        st.dataframe(training_df.head(20), use_container_width=True)


if __name__ == "__main__":
    main()
