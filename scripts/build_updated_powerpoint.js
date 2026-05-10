const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const OUT = path.join(ROOT, "ambulance_response_time_updated_results_arabic.pptx");

function readCsv(file) {
  const text = fs.readFileSync(file, "utf8").trim();
  const lines = text.split(/\r?\n/);
  const headers = lines.shift().split(",");
  return lines.map((line) => {
    const values = line.split(",");
    return Object.fromEntries(headers.map((h, i) => [h, values[i]]));
  });
}

const comparison = readCsv(path.join(ROOT, "outputs/reports/model_comparison.csv"));
const cases = readCsv(path.join(ROOT, "outputs/reports/sample_cases_for_manual_check.csv"));
const features = readCsv(path.join(ROOT, "outputs/reports/feature_importance.csv")).slice(0, 7);

const rf = comparison.find((r) => r.Algorithm === "Random Forest Regressor");
const lr = comparison.find((r) => r.Algorithm === "Linear Regression");
const example = cases[0];

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "عمير جبزيل عبد اللطيف";
pptx.subject = "Ambulance Response Time ML Results";
pptx.title = "تحديث نتائج مشروع توقع زمن استجابة الإسعاف";
pptx.company = "AI and Machine Learning Project";
pptx.lang = "ar-SA";
pptx.theme = {
  headFontFace: "Arial",
  bodyFontFace: "Arial",
  lang: "ar-SA",
};
pptx.defineLayout({ name: "WIDE", width: 13.333, height: 7.5 });
pptx.layout = "WIDE";

const C = {
  bg: "F8FAFC",
  ink: "172033",
  muted: "64748B",
  orange: "D97045",
  blue: "1F4E79",
  green: "2E7D61",
  line: "CBD5E1",
  white: "FFFFFF",
};

function addBg(slide, title) {
  slide.background = { color: C.bg };
  slide.addShape(pptx.ShapeType.rect, {
    x: 0,
    y: 0,
    w: 13.333,
    h: 0.18,
    fill: { color: C.orange },
    line: { color: C.orange },
  });
  slide.addText(title, {
    x: 0.65,
    y: 0.38,
    w: 12,
    h: 0.45,
    fontFace: "Arial",
    fontSize: 22,
    bold: true,
    color: C.ink,
    align: "right",
    rtlMode: true,
    margin: 0,
  });
}

function addFooter(slide, n) {
  slide.addText(`مشروع تعلم الآلة لتوقع زمن استجابة الإسعاف  |  ${n}`, {
    x: 0.5,
    y: 7.12,
    w: 12.3,
    h: 0.25,
    fontSize: 8,
    color: C.muted,
    align: "left",
    fontFace: "Arial",
  });
}

function textbox(slide, text, x, y, w, h, opts = {}) {
  slide.addText(text, {
    x,
    y,
    w,
    h,
    fontFace: "Arial",
    fontSize: opts.size || 14,
    color: opts.color || C.ink,
    bold: opts.bold || false,
    align: opts.align || "right",
    valign: opts.valign || "top",
    rtlMode: true,
    margin: opts.margin ?? 0.06,
    breakLine: false,
    fit: "shrink",
  });
}

function card(slide, x, y, w, h, title, body, color = C.blue) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w,
    h,
    rectRadius: 0.06,
    fill: { color: C.white },
    line: { color: C.line, transparency: 15 },
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: x + w - 0.12,
    y,
    w: 0.12,
    h,
    fill: { color },
    line: { color },
  });
  textbox(slide, title, x + 0.18, y + 0.16, w - 0.38, 0.35, {
    size: 14,
    bold: true,
    color,
  });
  textbox(slide, body, x + 0.18, y + 0.55, w - 0.38, h - 0.72, {
    size: 12,
    color: C.ink,
  });
}

function metricCard(slide, x, y, w, h, label, value, sub, color) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w,
    h,
    rectRadius: 0.06,
    fill: { color: C.white },
    line: { color: C.line },
  });
  textbox(slide, label, x + 0.15, y + 0.15, w - 0.3, 0.28, {
    size: 11,
    color: C.muted,
    bold: true,
  });
  textbox(slide, value, x + 0.15, y + 0.47, w - 0.3, 0.45, {
    size: 24,
    color,
    bold: true,
  });
  textbox(slide, sub, x + 0.15, y + 0.98, w - 0.3, 0.32, {
    size: 10,
    color: C.muted,
  });
}

function addImageIfExists(slide, file, x, y, w, h) {
  const full = path.join(ROOT, file);
  if (fs.existsSync(full)) {
    slide.addImage({ path: full, x, y, w, h });
  }
}

// Slide 1
{
  const slide = pptx.addSlide();
  slide.background = { color: C.bg };
  slide.addShape(pptx.ShapeType.rect, { x: 0, y: 0, w: 13.333, h: 0.2, fill: { color: C.orange }, line: { color: C.orange } });
  textbox(slide, "توقع زمن استجابة الإسعاف", 0.7, 1.25, 12, 0.7, {
    size: 34,
    bold: true,
    color: C.ink,
  });
  textbox(slide, "تحديث النتائج والخصائص ومثال المخرجات", 0.7, 2.1, 12, 0.5, {
    size: 20,
    color: C.blue,
    bold: true,
  });
  textbox(slide, "إعداد: عمير جبزيل عبد اللطيف", 0.7, 3.0, 12, 0.35, {
    size: 15,
    color: C.muted,
  });
  card(slide, 8.1, 4.05, 4.35, 1.25, "الفكرة الرئيسية", "تدريب نموذجين للتنبؤ بزمن الاستجابة بالدقائق، ثم مقارنة النتائج على حالات حقيقية من البيانات.", C.orange);
  card(slide, 3.25, 4.05, 4.35, 1.25, "النموذجان", "Linear Regression و Random Forest Regressor", C.blue);
  card(slide, 0.7, 4.05, 2.1, 1.25, "حجم التدريب", "100,000 سجل نظيف", C.green);
  addFooter(slide, 1);
}

// Slide 2
{
  const slide = pptx.addSlide();
  addBg(slide, "مسار العمل في المشروع");
  const steps = [
    ["1", "قراءة البيانات", "تحميل ملف EMS الكبير على شكل أجزاء."],
    ["2", "حفظ Dataset", "حفظ 100,000 سجل نظيف في CSV."],
    ["3", "تدريب النموذجين", "Linear Regression و Random Forest."],
    ["4", "مقارنة النتائج", "استخدام MAE و RMSE فقط."],
    ["5", "فحص الحالات", "حفظ CSV فيه رقم الحالة والنتيجة الفعلية والتوقعات."],
  ];
  steps.forEach((s, i) => {
    const x = 0.7 + i * 2.5;
    slide.addShape(pptx.ShapeType.ellipse, { x, y: 1.7, w: 0.65, h: 0.65, fill: { color: C.orange }, line: { color: C.orange } });
    textbox(slide, s[0], x + 0.13, 1.83, 0.4, 0.22, { align: "center", size: 13, bold: true, color: C.white });
    textbox(slide, s[1], x - 0.25, 2.55, 1.2, 0.35, { align: "center", size: 13, bold: true, color: C.blue });
    textbox(slide, s[2], x - 0.55, 3.05, 1.8, 0.9, { align: "center", size: 10, color: C.ink });
    if (i < steps.length - 1) slide.addShape(pptx.ShapeType.line, { x: x + 0.72, y: 2.02, w: 1.72, h: 0, line: { color: C.line, width: 2 } });
  });
  card(slide, 1.05, 5.1, 11.2, 1.0, "المخرجات الجديدة", "بعد التدريب يمكن فتح case_level_model_predictions.csv ومقارنة Case_Number مع Actual_Response_Time_Minutes وتوقعات النموذجين.", C.green);
  addFooter(slide, 2);
}

// Slide 3
{
  const slide = pptx.addSlide();
  addBg(slide, "الخصائص المستخدمة في التدريب");
  const featureCards = [
    ["خصائص زمنية", "Hour\nDayOfWeek\nMonth"],
    ["خصائص الشدة", "INITIAL_SEVERITY_LEVEL_CODE\nFINAL_SEVERITY_LEVEL_CODE"],
    ["خصائص مكانية", "BOROUGH\nINCIDENT_DISPATCH_AREA"],
    ["نوع البلاغ", "INCIDENT_CLASSIFICATION\nمأخوذة من FINAL_CALL_TYPE عند الحاجة"],
  ];
  featureCards.forEach((f, i) => {
    const x = i % 2 === 0 ? 6.85 : 0.85;
    const y = i < 2 ? 1.45 : 4.15;
    card(slide, x, y, 5.55, 1.85, f[0], f[1], i === 0 ? C.orange : i === 1 ? C.blue : i === 2 ? C.green : "7C3AED");
  });
  textbox(slide, "الهدف الذي يتعلمه النموذجان هو Response_Time_Minutes، أي زمن الاستجابة الفعلي بالدقائق.", 1.0, 6.35, 11.4, 0.35, { size: 13, color: C.muted, bold: true });
  addFooter(slide, 3);
}

// Slide 4
{
  const slide = pptx.addSlide();
  addBg(slide, "نتائج الأداء: MAE و RMSE");
  textbox(slide, "لأن المشكلة انحدار وليست تصنيف، لا نستخدم Accuracy. نستخدم MAE و RMSE لقياس خطأ التوقع بالدقائق.", 0.9, 1.1, 11.7, 0.5, { size: 14, color: C.muted, bold: true });
  metricCard(slide, 7.1, 2.0, 2.55, 1.55, "Random Forest MAE", Number(rf.MAE).toFixed(2), "أقل أفضل", C.green);
  metricCard(slide, 9.85, 2.0, 2.55, 1.55, "Random Forest RMSE", Number(rf.RMSE).toFixed(2), "يعاقب الأخطاء الكبيرة", C.green);
  metricCard(slide, 1.0, 2.0, 2.55, 1.55, "Linear MAE", Number(lr.MAE).toFixed(2), "أقل أفضل", C.blue);
  metricCard(slide, 3.75, 2.0, 2.55, 1.55, "Linear RMSE", Number(lr.RMSE).toFixed(2), "يعاقب الأخطاء الكبيرة", C.blue);
  card(slide, 1.0, 4.45, 11.4, 1.3, "النتيجة", "Random Forest أعطى خطأ أقل من Linear Regression في MAE و RMSE، لذلك يظهر أداء أفضل على عينة الاختبار الحالية.", C.orange);
  addFooter(slide, 4);
}

// Slide 5
{
  const slide = pptx.addSlide();
  addBg(slide, "مثال حقيقي من المخرجات");
  const rows = [
    ["Case_Number", example.Case_Number],
    ["BOROUGH", example.BOROUGH],
    ["INCIDENT_CLASSIFICATION", example.INCIDENT_CLASSIFICATION],
    ["Actual Response", `${Number(example.Actual_Response_Time_Minutes).toFixed(2)} دقيقة`],
    ["Linear Prediction", `${Number(example.Linear_Regression_Predicted_Minutes).toFixed(2)} دقيقة`],
    ["Random Forest Prediction", `${Number(example.Random_Forest_Predicted_Minutes).toFixed(2)} دقيقة`],
  ];
  rows.forEach((r, i) => {
    const y = 1.35 + i * 0.68;
    slide.addShape(pptx.ShapeType.roundRect, { x: 1.0, y, w: 11.3, h: 0.52, rectRadius: 0.04, fill: { color: i % 2 === 0 ? "FFFFFF" : "EEF2F7" }, line: { color: C.line } });
    textbox(slide, r[0], 6.7, y + 0.12, 5.25, 0.25, { size: 11, color: C.muted, bold: true });
    textbox(slide, r[1], 1.3, y + 0.12, 5.1, 0.25, { size: 12, color: C.ink, bold: i >= 3 });
  });
  textbox(slide, "يمكن فتح sample_cases_for_manual_check.csv واستخدام Case_Number لمقارنة زمن الاستجابة الحقيقي مع توقعات النموذجين.", 1.0, 6.0, 11.3, 0.5, { size: 13, color: C.blue, bold: true });
  addFooter(slide, 5);
}

// Slide 6
{
  const slide = pptx.addSlide();
  addBg(slide, "أهم العوامل المؤثرة في زمن الاستجابة");
  const maxVal = Math.max(...features.map((f) => Number(f.Importance)));
  features.forEach((f, i) => {
    const y = 1.35 + i * 0.58;
    const barW = (Number(f.Importance) / maxVal) * 6.6;
    textbox(slide, f.Feature, 7.8, y + 0.05, 4.4, 0.28, { size: 10.5, color: C.ink, bold: i < 3 });
    slide.addShape(pptx.ShapeType.rect, { x: 1.15, y: y + 0.05, w: barW, h: 0.28, fill: { color: i < 3 ? C.orange : C.blue }, line: { color: i < 3 ? C.orange : C.blue } });
    textbox(slide, Number(f.Importance).toFixed(3), 0.55, y + 0.05, 0.5, 0.22, { size: 8.5, color: C.muted, align: "left" });
  });
  card(slide, 1.0, 6.0, 11.4, 0.65, "قراءة النتيجة", "كلما زادت قيمة Importance كان العامل أكثر استخداما داخل Random Forest لتقليل خطأ التوقع.", C.green);
  addFooter(slide, 6);
}

// Slide 7
{
  const slide = pptx.addSlide();
  addBg(slide, "رسوم مساعدة من التحليل");
  addImageIfExists(slide, "outputs/figures/average_response_time_by_hour.png", 0.75, 1.2, 5.9, 4.0);
  addImageIfExists(slide, "outputs/figures/top_incident_classifications_by_response_time.png", 6.75, 1.2, 5.9, 4.0);
  textbox(slide, "الرسوم تساعد على فهم تأثير الوقت ونوع البلاغ قبل النظر إلى النموذج.", 0.95, 5.75, 11.5, 0.45, { size: 14, color: C.muted, bold: true });
  addFooter(slide, 7);
}

// Slide 8
{
  const slide = pptx.addSlide();
  addBg(slide, "الخلاصة وطريقة التحقق");
  card(slide, 7.1, 1.4, 5.3, 1.35, "ماذا تم تحديثه؟", "تم حفظ Dataset من 100,000 سجل، وتدريب النموذجين، وحفظ ملف مقارنة حالة بحالة.", C.orange);
  card(slide, 1.0, 1.4, 5.3, 1.35, "كيف نتحقق؟", "نفتح CSV، نأخذ Case_Number، ثم نقارن Actual Response مع توقعات Linear و Random Forest.", C.blue);
  card(slide, 7.1, 3.35, 5.3, 1.35, "أفضل أداء", "Random Forest حقق MAE و RMSE أقل في العينة الحالية.", C.green);
  card(slide, 1.0, 3.35, 5.3, 1.35, "أمر الفحص", "python main.py predict-case --case-id 230022496", "7C3AED");
  textbox(slide, "المشروع الآن لا يعتمد على R2، ويعرض MAE و RMSE فقط لأن المطلوب قياس خطأ التوقع بالدقائق.", 1.0, 5.75, 11.4, 0.5, { size: 14, color: C.ink, bold: true });
  addFooter(slide, 8);
}

pptx.writeFile({ fileName: OUT });
console.log(OUT);
