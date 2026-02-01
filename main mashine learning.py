# ============================================================
# תרגיל למידת מכונה - פתרון מלא
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# הגדרת סגנון כללי לגרפים
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 11

# ============================================================
# חלק א: אנליזה וניבוי מחירי דיור - רגרסיה לינארית
# ============================================================

print("=" * 60)
print("חלק א: ניבוי מחירי דיור - רגרסיה לינארית")
print("=" * 60)

# --- שלב 1: טעינת הנתונים ---
# קוראים את קובץ הנתונים של מחירי הדיור
df_housing = pd.read_csv("Project housing data .csv")

print("\n--- מידע על הנתונים ---")
print(f"מספר שורות: {df_housing.shape[0]}, מספר עמודות: {df_housing.shape[1]}")
print(f"\nעמודות: {list(df_housing.columns)}")
print(f"\n5 שורות ראשונות:")
print(df_housing.head())
print(f"\nסטטיסטיקות תיאוריות:")
print(df_housing.describe())

# --- שלב 2: עיבוד מקדים ---
# בדיקה אם יש ערכים חסרים
print(f"\nערכים חסרים בכל עמודה:")
print(df_housing.isnull().sum())

# הסבר על קידוד המשתנים:
# - waterfront: משתנה בינארי (0/1) - כבר מקודד, אין צורך בשינוי
# - view (0-4): משתנה סדור (ordinal) - הערכים מייצגים סדר מהגרוע לטוב, אפשר להשאיר כמספר
# - condition (1-5): משתנה סדור (ordinal) - הערכים מייצגים סדר מהגרוע לטוב, אפשר להשאיר כמספר
# - שאר המשתנים: מספריים רציפים - אין צורך בקידוד
#
# הערה: view ו-condition הם משתנים סדורים (יש משמעות לסדר בין הערכים),
# ולכן ניתן להשאיר אותם כערכים מספריים ברגרסיה לינארית.

# --- שלב 3: הגדרת משתנים והפרדת נתונים ---
# X = כל המשתנים המסבירים (features), y = המשתנה התלוי (מחיר)
X_housing = df_housing.drop("price", axis=1)
y_housing = df_housing["price"]

# חלוקה לסט אימון (80%) וסט מבחן (20%)
# random_state=42 מבטיח תוצאות עקביות בכל הרצה
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_housing, y_housing, test_size=0.2, random_state=42
)

print(f"\nגודל סט אימון: {X_train_h.shape[0]} דירות")
print(f"גודל סט מבחן: {X_test_h.shape[0]} דירות")

# --- שלב 4: בניית מודל רגרסיה לינארית ---
# רגרסיה לינארית מנסה למצוא קו (או מישור) שממזער את סכום ריבועי השגיאות
model_housing = LinearRegression()
model_housing.fit(X_train_h, y_train_h)

# הצגת המקדמים (coefficients) של המודל
print("\n--- מקדמי המודל ---")
for feature, coef in zip(X_housing.columns, model_housing.coef_):
    print(f"  {feature}: {coef:,.2f}")
print(f"  חותך (intercept): {model_housing.intercept_:,.2f}")

# --- שלב 5: הערכת ביצועי המודל ---
# החיזוי נעשה על נתוני המבחן (שהמודל לא ראה בזמן האימון)
y_pred_h = model_housing.predict(X_test_h)

mae = mean_absolute_error(y_test_h, y_pred_h)
mse = mean_squared_error(y_test_h, y_pred_h)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_h, y_pred_h)

print("\n--- הערכת ביצועי המודל (על נתוני המבחן) ---")
print(f"  MAE (שגיאה מוחלטת ממוצעת): {mae:,.2f} $")
print(f"    -> בממוצע, המודל טועה ב-{mae:,.0f} דולר")
print(f"  MSE (שגיאה ריבועית ממוצעת): {mse:,.2f}")
print(f"  RMSE (שורש שגיאה ריבועית ממוצעת): {rmse:,.2f} $")
print(f"    -> מדד שגיאה ביחידות של דולר, מעניש טעויות גדולות יותר")
print(f"  R² (מקדם הקביעה): {r2:.4f}")
print(f"    -> המודל מסביר {r2*100:.1f}% מהשונות במחירי הדירות")

# --- שלב 6: פרדיקציה לדירה חדשה ---
print("\n--- חיזוי מחיר לדירה חדשה ---")

# יצירת דירה חדשה עם מאפיינים ריאליסטיים:
# דירה בת 3 חדרי שינה, 2 חדרי אמבטיה, 1,800 רגל רבוע שטח מגורים,
# מגרש של 6,000 רגל רבוע, קומה אחת, ללא חזית מים, נוף בינוני,
# מצב טוב, שטח עילי 1,800, ללא מרתף, נבנה ב-1990
new_apartment = pd.DataFrame({
    "bedrooms": [3],
    "bathrooms": [2],
    "sqft_living": [1800],
    "sqft_lot": [6000],
    "floors": [1],
    "waterfront": [0],
    "view": [2],
    "condition": [4],
    "sqft_above": [1800],
    "sqft_basement": [0],
    "yr_built": [1990]
})

predicted_price = model_housing.predict(new_apartment)[0]

print("מאפייני הדירה החדשה:")
print(f"  חדרי שינה: 3, חדרי אמבטיה: 2")
print(f"  שטח מגורים: 1,800 רגל רבוע")
print(f"  שטח מגרש: 6,000 רגל רבוע")
print(f"  קומות: 1, חזית מים: לא")
print(f"  נוף: 2 (בינוני), מצב: 4 (טוב)")
print(f"  שטח עילי: 1,800, מרתף: 0")
print(f"  שנת בנייה: 1990")
print(f"\n  מחיר חזוי: ${predicted_price:,.2f}")

# בדיקת סבירות: נשווה למחיר הממוצע בנתונים
avg_price = df_housing["price"].mean()
median_price = df_housing["price"].median()
print(f"\n  לשם השוואה:")
print(f"    מחיר ממוצע בנתונים: ${avg_price:,.2f}")
print(f"    מחיר חציוני בנתונים: ${median_price:,.2f}")
print(f"\n  הסבר: המחיר החזוי נראה סביר כי מדובר בדירה בגודל בינוני,")
print(f"  במצב טוב, עם נוף בינוני, וללא חזית מים - מאפיינים שמצדיקים")
print(f"  מחיר באזור הממוצע/חציון של השוק.")

# ============================================================
# ויזואליזציה - חלק א: דשבורד מחירי דיור
# ============================================================
print("\n--- מייצר גרפים לחלק א (דיור)... ---")

fig_housing, axes_h = plt.subplots(2, 3, figsize=(20, 12))
fig_housing.suptitle("Dashboard - Housing Price Analysis", fontsize=18, fontweight="bold", y=1.02)

# גרף 1: התפלגות מחירי הדירות (היסטוגרמה)
axes_h[0, 0].hist(df_housing["price"], bins=40, color="steelblue", edgecolor="white", alpha=0.8)
axes_h[0, 0].axvline(avg_price, color="red", linestyle="--", linewidth=2, label=f"Mean: ${avg_price:,.0f}")
axes_h[0, 0].axvline(median_price, color="orange", linestyle="--", linewidth=2, label=f"Median: ${median_price:,.0f}")
axes_h[0, 0].set_title("Price Distribution", fontsize=13, fontweight="bold")
axes_h[0, 0].set_xlabel("Price ($)")
axes_h[0, 0].set_ylabel("Count")
axes_h[0, 0].legend()

# גרף 2: מטריצת קורלציה (Heatmap)
corr_matrix = df_housing.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            ax=axes_h[0, 1], cbar_kws={"shrink": 0.8}, linewidths=0.5, square=True,
            annot_kws={"size": 7})
axes_h[0, 1].set_title("Correlation Matrix", fontsize=13, fontweight="bold")
axes_h[0, 1].tick_params(axis="both", labelsize=7)

# גרף 3: מחיר בפועל מול מחיר חזוי (Scatter)
axes_h[0, 2].scatter(y_test_h, y_pred_h, alpha=0.5, color="steelblue", edgecolor="white", s=30)
max_val = max(y_test_h.max(), y_pred_h.max())
axes_h[0, 2].plot([0, max_val], [0, max_val], "r--", linewidth=2, label="Perfect Prediction")
axes_h[0, 2].set_title(f"Actual vs Predicted (R²={r2:.3f})", fontsize=13, fontweight="bold")
axes_h[0, 2].set_xlabel("Actual Price ($)")
axes_h[0, 2].set_ylabel("Predicted Price ($)")
axes_h[0, 2].legend()

# גרף 4: מקדמי המודל (חשיבות משתנים)
coef_df = pd.DataFrame({
    "Feature": X_housing.columns,
    "Coefficient": model_housing.coef_
}).sort_values("Coefficient", key=abs, ascending=True)
colors = ["#e74c3c" if c < 0 else "#2ecc71" for c in coef_df["Coefficient"]]
axes_h[1, 0].barh(coef_df["Feature"], coef_df["Coefficient"], color=colors, edgecolor="white")
axes_h[1, 0].set_title("Model Coefficients", fontsize=13, fontweight="bold")
axes_h[1, 0].set_xlabel("Coefficient Value")
axes_h[1, 0].axvline(0, color="black", linewidth=0.8)

# גרף 5: התפלגות השגיאות (Residuals)
residuals = y_test_h - y_pred_h
axes_h[1, 1].hist(residuals, bins=40, color="coral", edgecolor="white", alpha=0.8)
axes_h[1, 1].axvline(0, color="black", linestyle="--", linewidth=2)
axes_h[1, 1].set_title("Residuals Distribution", fontsize=13, fontweight="bold")
axes_h[1, 1].set_xlabel("Prediction Error ($)")
axes_h[1, 1].set_ylabel("Count")

# גרף 6: שטח מגורים מול מחיר
axes_h[1, 2].scatter(df_housing["sqft_living"], df_housing["price"],
                     alpha=0.4, color="steelblue", edgecolor="white", s=20)
axes_h[1, 2].set_title("Sqft Living vs Price", fontsize=13, fontweight="bold")
axes_h[1, 2].set_xlabel("Sqft Living")
axes_h[1, 2].set_ylabel("Price ($)")

plt.tight_layout()
plt.savefig("housing_dashboard.png", dpi=150, bbox_inches="tight")
plt.show()
print("  -> הגרף נשמר בקובץ housing_dashboard.png")


# ============================================================
# חלק ב: אנליזה וניבוי של התקפי לב - רגרסיה לוגיסטית
# ============================================================

print("\n\n" + "=" * 60)
print("חלק ב: ניבוי התקפי לב - רגרסיה לוגיסטית")
print("=" * 60)

# --- שלב 1: טעינת הנתונים ---
df_heart = pd.read_csv("Project Heart attack data.csv")

print("\n--- מידע על הנתונים ---")
print(f"מספר שורות: {df_heart.shape[0]}, מספר עמודות: {df_heart.shape[1]}")
print(f"\nעמודות: {list(df_heart.columns)}")
print(f"\n5 שורות ראשונות:")
print(df_heart.head())
print(f"\nהתפלגות משתנה המטרה (target):")
print(df_heart["target"].value_counts())
print("  0 = סיכוי נמוך להתקף לב")
print("  1 = סיכוי גבוה להתקף לב")

# בדיקת ערכים חסרים
print(f"\nערכים חסרים:")
print(df_heart.isnull().sum())

# --- שלב 2: קידוד משתנים קטגוריאליים ---
# הסבר על המשתנים שדורשים קידוד:
#
# cp (סוג כאב חזה) - משתנה קטגוריאלי נומינלי (0,1,2,3):
#   0 = אסימפטומטי, 1 = אנגינה טיפוסית, 2 = אנגינה לא טיפוסית, 3 = כאב לא-אנגינלי
#   -> אין סדר טבעי בין הקטגוריות, לכן נבצע One-Hot Encoding
#
# restecg (תוצאות ECG במנוחה) - משתנה קטגוריאלי נומינלי (0,1,2):
#   0 = תקין, 1 = חריגת גל ST-T, 2 = היפרטרופיה של חדר שמאל
#   -> אין סדר טבעי, לכן נבצע One-Hot Encoding
#
# משתנים בינאריים שכבר מקודדים ואינם דורשים שינוי:
#   sex (0/1), fbs (0/1), exng (0/1)

# ביצוע One-Hot Encoding עם drop_first=True כדי למנוע בעיית מולטיקולינאריות
# (כלומר, אם יש 4 קטגוריות, ניצור 3 עמודות דמה - הקטגוריה הרביעית מיוצגת
# כאשר כל שאר העמודות הן 0)
df_heart_encoded = pd.get_dummies(df_heart, columns=["cp", "restecg"], drop_first=True)

print("\n--- לאחר קידוד One-Hot ---")
print(f"עמודות חדשות: {list(df_heart_encoded.columns)}")
print(f"מספר עמודות: {df_heart_encoded.shape[1]} (לפני: {df_heart.shape[1]})")

# --- שלב 3: הגדרת משתנים והפרדת נתונים ---
X_heart = df_heart_encoded.drop("target", axis=1)
y_heart = df_heart_encoded["target"]

# חלוקה לסט אימון (80%) וסט מבחן (20%)
X_train_hr, X_test_hr, y_train_hr, y_test_hr = train_test_split(
    X_heart, y_heart, test_size=0.2, random_state=42
)

print(f"\nגודל סט אימון: {X_train_hr.shape[0]} מטופלים")
print(f"גודל סט מבחן: {X_test_hr.shape[0]} מטופלים")

# --- שלב 4: בניית מודל רגרסיה לוגיסטית ---
# רגרסיה לוגיסטית מחשבת הסתברות לשייכות לקטגוריה (0 או 1)
# max_iter=1000 כדי להבטיח שהאלגוריתם יתכנס
model_heart = LogisticRegression(max_iter=1000, random_state=42)
model_heart.fit(X_train_hr, y_train_hr)

# --- שלב 5: הערכת ביצועי המודל ---
y_pred_hr = model_heart.predict(X_test_hr)

accuracy = accuracy_score(y_test_hr, y_pred_hr)
precision = precision_score(y_test_hr, y_pred_hr)
recall = recall_score(y_test_hr, y_pred_hr)
f1 = f1_score(y_test_hr, y_pred_hr)

print("\n--- הערכת ביצועי המודל (על נתוני המבחן) ---")
print(f"  Accuracy (דיוק כללי): {accuracy:.4f}")
print(f"    -> {accuracy*100:.1f}% מהתצפיות סווגו נכון")
print(f"  Precision (דיוק חיובי): {precision:.4f}")
print(f"    -> מתוך כל מי שהמודל חזה שבסיכון, {precision*100:.1f}% באמת בסיכון")
print(f"  Recall (רגישות): {recall:.4f}")
print(f"    -> המודל זיהה {recall*100:.1f}% מכלל החולים האמיתיים")
print(f"  F1 Score: {f1:.4f}")
print(f"    -> ממוצע הרמוני של Precision ו-Recall")

print(f"\n  מטריצת בלבול (Confusion Matrix):")
cm = confusion_matrix(y_test_hr, y_pred_hr)
print(f"    {cm}")
print(f"    [TN={cm[0,0]}, FP={cm[0,1]}]")
print(f"    [FN={cm[1,0]}, TP={cm[1,1]}]")

print(f"\n  דוח סיווג מלא:")
print(classification_report(y_test_hr, y_pred_hr, target_names=["סיכוי נמוך", "סיכוי גבוה"]))

# --- הסבר: איזו מטריקה חשובה יותר בהקשר הרפואי? ---
print("--- איזו מטריקה חשובה יותר בהקשר הרפואי? ---")
print("""
  בהקשר רפואי, המטריקה החשובה ביותר היא Recall (רגישות).

  הסיבה: בחיזוי התקפי לב, השגיאה החמורה ביותר היא False Negative -
  כלומר מטופל שנמצא בסיכון גבוה להתקף לב אך המודל חזה שהוא בסדר.
  טעות כזו עלולה לעלות בחיי אדם, כי המטופל לא יקבל טיפול בזמן.

  לעומת זאת, False Positive (המודל חזה סיכון אך המטופל בריא) פחות חמור -
  המטופל יעבור בדיקות נוספות מיותרות, אך לא יהיה מסוכן.

  Recall מודד בדיוק את זה: מתוך כל החולים האמיתיים, כמה המודל הצליח לזהות?
  ככל ש-Recall גבוה יותר, פחות חולים "נפלו בין הכיסאות".

  לכן, בתחום הרפואה מעדיפים מודל עם Recall גבוה, גם אם זה בא על חשבון
  Precision נמוך יותר (כלומר, יותר התראות שווא, אך פחות חולים שלא אותרו).
""")

# --- שלב 6: פרדיקציה למטופל חדש ---
print("--- חיזוי למטופל חדש ---")

# יצירת מטופל חדש עם מאפיינים ריאליסטיים:
# גבר בן 55, כאב חזה אסימפטומטי (cp=0), לחץ דם 140,
# כולסטרול 250, סוכר בצום תקין, ECG תקין (restecg=0),
# דופק מקסימלי 150, ללא אנגינה במאמץ, 1 כלי דם ראשי
new_patient_data = {
    "age": [55],
    "sex": [1],            # גבר
    "trtbps": [140],       # לחץ דם במנוחה
    "chol": [250],         # כולסטרול
    "fbs": [0],            # סוכר בצום תקין
    "thalachh": [150],     # דופק מקסימלי
    "exng": [0],           # ללא אנגינה במאמץ
    "ca": [1],             # כלי דם ראשי אחד
}

# הוספת עמודות One-Hot לפי הקידוד שבוצע
# cp=0 (אסימפטומטי) -> כל עמודות cp הן 0 (כי זו קטגוריית הבסיס)
# restecg=0 (תקין) -> כל עמודות restecg הן 0 (כי זו קטגוריית הבסיס)
for col in X_heart.columns:
    if col not in new_patient_data:
        new_patient_data[col] = [0]

new_patient = pd.DataFrame(new_patient_data)
# סידור העמודות בסדר הנכון
new_patient = new_patient[X_heart.columns]

prediction = model_heart.predict(new_patient)[0]
prediction_proba = model_heart.predict_proba(new_patient)[0]

print("מאפייני המטופל החדש:")
print(f"  גיל: 55, מין: גבר")
print(f"  סוג כאב חזה: אסימפטומטי (cp=0)")
print(f"  לחץ דם במנוחה: 140 mmHg")
print(f"  כולסטרול: 250 mg/dl")
print(f"  סוכר בצום: תקין (< 120 mg/dl)")
print(f"  תוצאות ECG: תקינות")
print(f"  דופק מקסימלי: 150")
print(f"  אנגינה במאמץ: לא")
print(f"  מספר כלי דם ראשיים: 1")

print(f"\n  תוצאת החיזוי: {'סיכוי גבוה להתקף לב' if prediction == 1 else 'סיכוי נמוך להתקף לב'}")
print(f"  הסתברות לסיכוי נמוך: {prediction_proba[0]:.4f} ({prediction_proba[0]*100:.1f}%)")
print(f"  הסתברות לסיכוי גבוה: {prediction_proba[1]:.4f} ({prediction_proba[1]*100:.1f}%)")

print(f"""
  פירוש קליני:
  המודל חזה שלמטופל זה יש {'סיכוי גבוה' if prediction == 1 else 'סיכוי נמוך'} להתקף לב.
  {'המלצה: יש לבצע בדיקות נוספות מעמיקות ולשקול טיפול מונע.' if prediction == 1 else 'עם זאת, מומלץ להמשיך במעקב רפואי שוטף.'}

  הסבר: המטופל הוא גבר בן 55 עם לחץ דם מוגבה (140) וכולסטרול גבוה (250),
  שני גורמי סיכון מרכזיים למחלות לב. עם זאת, הדופק המקסימלי שלו תקין (150),
  אין לו אנגינה במאמץ, ותוצאות ה-ECG תקינות - סימנים חיוביים.
  יש לו כלי דם ראשי אחד שנמצא מוצר, מה שמהווה גורם סיכון נוסף.
""")

# ============================================================
# ויזואליזציה - חלק ב: דשבורד התקפי לב
# ============================================================
print("--- מייצר גרפים לחלק ב (התקפי לב)... ---")

fig_heart, axes_hr = plt.subplots(2, 3, figsize=(20, 12))
fig_heart.suptitle("Dashboard - Heart Attack Prediction Analysis", fontsize=18, fontweight="bold", y=1.02)

# גרף 1: מטריצת בלבול (Confusion Matrix Heatmap)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes_hr[0, 0],
            xticklabels=["Low Risk", "High Risk"],
            yticklabels=["Low Risk", "High Risk"],
            annot_kws={"size": 16}, linewidths=2)
axes_hr[0, 0].set_title("Confusion Matrix", fontsize=13, fontweight="bold")
axes_hr[0, 0].set_xlabel("Predicted", fontsize=11)
axes_hr[0, 0].set_ylabel("Actual", fontsize=11)

# גרף 2: מדדי ביצועים (Metrics Bar Chart)
metrics_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
metrics_values = [accuracy, precision, recall, f1]
bar_colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]
bars = axes_hr[0, 1].bar(metrics_names, metrics_values, color=bar_colors, edgecolor="white", width=0.6)
for bar, val in zip(bars, metrics_values):
    axes_hr[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f"{val:.3f}", ha="center", fontweight="bold", fontsize=11)
axes_hr[0, 1].set_ylim(0, 1.1)
axes_hr[0, 1].set_title("Model Performance Metrics", fontsize=13, fontweight="bold")
axes_hr[0, 1].set_ylabel("Score")

# גרף 3: עקומת ROC
y_proba_hr = model_heart.predict_proba(X_test_hr)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test_hr, y_proba_hr)
roc_auc = auc(fpr, tpr)
axes_hr[0, 2].plot(fpr, tpr, color="#e74c3c", linewidth=2.5, label=f"ROC Curve (AUC = {roc_auc:.3f})")
axes_hr[0, 2].plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random Classifier")
axes_hr[0, 2].fill_between(fpr, tpr, alpha=0.15, color="#e74c3c")
axes_hr[0, 2].set_title("ROC Curve", fontsize=13, fontweight="bold")
axes_hr[0, 2].set_xlabel("False Positive Rate")
axes_hr[0, 2].set_ylabel("True Positive Rate")
axes_hr[0, 2].legend(loc="lower right")

# גרף 4: התפלגות גילאים לפי סיכון להתקף לב
for target_val, label, color in [(0, "Low Risk", "#2ecc71"), (1, "High Risk", "#e74c3c")]:
    subset = df_heart[df_heart["target"] == target_val]
    axes_hr[1, 0].hist(subset["age"], bins=15, alpha=0.6, color=color, label=label, edgecolor="white")
axes_hr[1, 0].set_title("Age Distribution by Risk", fontsize=13, fontweight="bold")
axes_hr[1, 0].set_xlabel("Age")
axes_hr[1, 0].set_ylabel("Count")
axes_hr[1, 0].legend()

# גרף 5: מקדמי המודל (חשיבות משתנים)
heart_coef_df = pd.DataFrame({
    "Feature": X_heart.columns,
    "Coefficient": model_heart.coef_[0]
}).sort_values("Coefficient", key=abs, ascending=True)
colors_hr = ["#e74c3c" if c < 0 else "#2ecc71" for c in heart_coef_df["Coefficient"]]
axes_hr[1, 1].barh(heart_coef_df["Feature"], heart_coef_df["Coefficient"], color=colors_hr, edgecolor="white")
axes_hr[1, 1].set_title("Model Coefficients", fontsize=13, fontweight="bold")
axes_hr[1, 1].set_xlabel("Coefficient Value")
axes_hr[1, 1].axvline(0, color="black", linewidth=0.8)

# גרף 6: התפלגות משתנה המטרה (Target) - Pie Chart
target_counts = df_heart["target"].value_counts()
axes_hr[1, 2].pie(target_counts, labels=["High Risk (1)", "Low Risk (0)"],
                  colors=["#e74c3c", "#2ecc71"], autopct="%1.1f%%",
                  startangle=90, textprops={"fontsize": 12},
                  explode=(0.05, 0), shadow=True)
axes_hr[1, 2].set_title("Target Distribution", fontsize=13, fontweight="bold")

plt.tight_layout()
plt.savefig("heart_attack_dashboard.png", dpi=150, bbox_inches="tight")
plt.show()
print("  -> הגרף נשמר בקובץ heart_attack_dashboard.png")


# ============================================================
# חלק ג: Clustering – K-Means | פילוח לקוחות כרטיסי אשראי
# ============================================================
# רקע עסקי:
# חברת "CardWise" רוצה להבין את התנהגות הלקוחות השונים.
# נשתמש באלגוריתם K-Means כדי לחלק את הלקוחות לקבוצות (segments)
# על בסיס דפוסי ההוצאה, המשיכות במזומן, מסגרת האשראי ועוד.

print("\n\n" + "=" * 60)
print("חלק ג: Clustering – K-Means | פילוח לקוחות כרטיסי אשראי")
print("=" * 60)

# ============================================================
# חלק א' של התרגיל – טעינת הנתונים ובדיקה ראשונית
# ============================================================
# הסבר למידה:
# לפני כל מודל, חשוב להכיר את הנתונים: מבנה, סוגי משתנים, ערכים חסרים.
# זה שלב קריטי שמונע טעויות בהמשך.

print("\n--- חלק א': טעינת נתונים ובדיקה ראשונית ---")

# שלב 1: טעינת הנתונים
df_customers = pd.read_csv("Customer Data.csv")

# שלב 2: הצגת מידע בסיסי
print(f"\n  מספר שורות (לקוחות): {df_customers.shape[0]}")
print(f"  מספר עמודות (תכונות): {df_customers.shape[1]}")
print(f"\n  5 שורות ראשונות:")
print(df_customers.head())
print(f"\n  סוגי עמודות ומידע כללי:")
print(df_customers.info())
print(f"\n  סטטיסטיקות תיאוריות:")
print(df_customers.describe())

# שלב 3: בדיקת ערכים חסרים
print("\n--- בדיקת ערכים חסרים ---")
missing = df_customers.isnull().sum()
missing_cols = missing[missing > 0]
print(missing_cols)

# הסבר למידה:
# טיפול בערכים חסרים – יש כמה שיטות:
#   1. מחיקת שורות – פשוט, אבל מאבדים מידע
#   2. מילוי בממוצע (mean) – רגיש לערכים קיצוניים (outliers)
#   3. מילוי בחציון (median) – עמיד יותר בפני outliers
#
# בנתוני כרטיסי אשראי יש הרבה outliers (לקוחות עם הוצאות חריגות),
# ולכן נבחר מילוי בחציון (median) – זה הערך האמצעי של הנתונים,
# שלא מושפע מערכים קיצוניים.

print("\n  שיטת טיפול: מילוי בחציון (median)")
print("  סיבה: הנתונים מכילים outliers (ערכים קיצוניים),")
print("  ולכן median עדיף על mean כי הוא לא מושפע מהם.")

for col in missing_cols.index:
    median_val = df_customers[col].median()
    df_customers[col] = df_customers[col].fillna(median_val)
    print(f"    {col}: מולא ב-median = {median_val:,.2f}")

print(f"\n  אימות - ערכים חסרים לאחר טיפול: {df_customers.isnull().sum().sum()}")


# ============================================================
# חלק ב' של התרגיל – ניתוח חקר נתונים (EDA)
# ============================================================
# הסבר למידה:
# EDA (Exploratory Data Analysis) עוזר לנו להבין את מבנה הנתונים,
# לזהות דפוסים, קשרים בין משתנים, ונקודות חריגות.
# זה שלב חיוני לפני בניית כל מודל.

print("\n\n--- חלק ב': ניתוח חקר נתונים (EDA) ---")

# 1. מטריצת קורלציה
# הסבר למידה:
# קורלציה מודדת את עוצמת הקשר הליניארי בין שני משתנים (בין -1 ל-1).
# ערך קרוב ל-1: קשר חיובי חזק (כשאחד עולה, גם השני עולה)
# ערך קרוב ל-(-1): קשר שלילי חזק
# ערך קרוב ל-0: אין קשר ליניארי
# חשוב ב-Clustering: תכונות עם קורלציה גבוהה מאוד עלולות "לספור פעמיים".

print("\n  1. מטריצת קורלציה בין תכונות עיקריות:")
key_features = ["BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS"]
corr_customers = df_customers[key_features].corr()
print(corr_customers.round(3))

print("\n  פרשנות מטריצת קורלציה:")
print("  - PURCHASES ↔ PAYMENTS: קורלציה חיובית – לקוחות שקונים יותר גם משלמים יותר")
print("  - BALANCE ↔ CASH_ADVANCE: קורלציה חיובית – משיכות מזומן מעלות את היתרה")
print("  - PURCHASES ↔ CASH_ADVANCE: קורלציה נמוכה – לקוחות נוטים לאחד מהשניים, לא לשניהם")

# 2. ויזואליזציות EDA
print("\n  2. מייצר גרפי EDA...")

fig_eda, axes_eda = plt.subplots(2, 3, figsize=(20, 12))
fig_eda.suptitle("EDA - Customer Credit Card Data", fontsize=18, fontweight="bold", y=1.02)

# גרף 1: היסטוגרמה של BALANCE
# הסבר למידה: היסטוגרמה מראה את התפלגות הערכים.
# אם ההתפלגות מוטה ימינה (right-skewed), יש הרבה לקוחות עם ערכים נמוכים
# ומעט לקוחות עם ערכים גבוהים מאוד (outliers).
axes_eda[0, 0].hist(df_customers["BALANCE"], bins=50, color="steelblue", edgecolor="white", alpha=0.8)
axes_eda[0, 0].set_title("Distribution of BALANCE", fontsize=13, fontweight="bold")
axes_eda[0, 0].set_xlabel("Balance")
axes_eda[0, 0].set_ylabel("Count")

# גרף 2: היסטוגרמה של PURCHASES
axes_eda[0, 1].hist(df_customers["PURCHASES"], bins=50, color="coral", edgecolor="white", alpha=0.8)
axes_eda[0, 1].set_title("Distribution of PURCHASES", fontsize=13, fontweight="bold")
axes_eda[0, 1].set_xlabel("Purchases")
axes_eda[0, 1].set_ylabel("Count")

# גרף 3: Scatter plot – PURCHASES vs CASH_ADVANCE
# הסבר למידה: Scatter plot מראה את הקשר בין שני משתנים.
# אם הנקודות מתפזרות לאורך הצירים (ולא באלכסון), זה מצביע
# על כך שלקוחות נוטים להשתמש ברכישות או במשיכות מזומן, אך לא בשניהם.
axes_eda[0, 2].scatter(df_customers["PURCHASES"], df_customers["CASH_ADVANCE"],
                       alpha=0.3, color="purple", edgecolor="white", s=15)
axes_eda[0, 2].set_title("Purchases vs Cash Advance", fontsize=13, fontweight="bold")
axes_eda[0, 2].set_xlabel("Purchases")
axes_eda[0, 2].set_ylabel("Cash Advance")

# גרף 4: מטריצת קורלציה (Heatmap)
sns.heatmap(corr_customers, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            ax=axes_eda[1, 0], cbar_kws={"shrink": 0.8}, linewidths=0.5, square=True)
axes_eda[1, 0].set_title("Correlation Matrix", fontsize=13, fontweight="bold")

# גרף 5: Boxplot של CASH_ADVANCE
# הסבר למידה: Boxplot מציג את התפלגות הנתונים:
# - הקו באמצע = חציון (median)
# - הקופסה = רבעון 1 עד רבעון 3 (50% מהנתונים)
# - ה-"שפמים" = טווח סביר
# - נקודות מעבר לשפמים = outliers (נקודות חריגות)
axes_eda[1, 1].boxplot(df_customers["CASH_ADVANCE"].dropna(), vert=True, patch_artist=True,
                       boxprops=dict(facecolor="lightyellow", color="orange"),
                       medianprops=dict(color="red", linewidth=2))
axes_eda[1, 1].set_title("Boxplot: CASH_ADVANCE", fontsize=13, fontweight="bold")
axes_eda[1, 1].set_ylabel("Cash Advance Amount")

# גרף 6: Boxplot של CREDIT_LIMIT
axes_eda[1, 2].boxplot(df_customers["CREDIT_LIMIT"].dropna(), vert=True, patch_artist=True,
                       boxprops=dict(facecolor="lightcyan", color="steelblue"),
                       medianprops=dict(color="red", linewidth=2))
axes_eda[1, 2].set_title("Boxplot: CREDIT_LIMIT", fontsize=13, fontweight="bold")
axes_eda[1, 2].set_ylabel("Credit Limit")

plt.tight_layout()
plt.savefig("clustering_eda_dashboard.png", dpi=150, bbox_inches="tight")
plt.show()
print("  -> הגרפים נשמרו בקובץ clustering_eda_dashboard.png")

# 3. זיהוי Outliers
print("\n  3. זיהוי נקודות קיצון (Outliers):")
for col_name in ["CASH_ADVANCE", "CREDIT_LIMIT"]:
    Q1 = df_customers[col_name].quantile(0.25)
    Q3 = df_customers[col_name].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((df_customers[col_name] < lower) | (df_customers[col_name] > upper)).sum()
    print(f"    {col_name}: Q1={Q1:,.0f}, Q3={Q3:,.0f}, IQR={IQR:,.0f}")
    print(f"      גבול עליון: {upper:,.0f}, מספר outliers: {n_outliers}")

print("\n  הסבר: לא נסיר את ה-outliers כי ב-Clustering הם עשויים לייצג")
print("  סגמנט לגיטימי של לקוחות (למשל, לקוחות VIP עם הוצאות גבוהות).")


# ============================================================
# חלק ג' של התרגיל – קדם-עיבוד והתאמה למודל
# ============================================================
# הסבר למידה:
# K-Means עובד על בסיס מרחקים אוקלידיים בין נקודות.
# אם תכונה אחת נמדדת באלפים (כמו BALANCE) ואחרת בין 0 ל-1
# (כמו PURCHASES_FREQUENCY), התכונה עם הערכים הגדולים תשלוט
# על חישוב המרחק. לכן חובה לבצע Standardization (נרמול)
# כדי שכל התכונות יהיו באותה סקאלה (ממוצע=0, סטיית תקן=1).

print("\n\n--- חלק ג': קדם-עיבוד והתאמה למודל ---")

# בחירת תכונות רלוונטיות
# הסרנו:
#   CUST_ID – מזהה ייחודי, לא מידע מספרי רלוונטי
#   TENURE – כמעט אחיד אצל כל הלקוחות (רובם 12 חודשים), לא תורם לפילוח
cols_to_drop = ["CUST_ID", "TENURE"]
df_clustering = df_customers.drop(columns=cols_to_drop)

print(f"  תכונות שנבחרו למודל ({df_clustering.shape[1]}):")
for i, col in enumerate(df_clustering.columns, 1):
    print(f"    {i}. {col}")

print(f"\n  תכונות שהוסרו: {cols_to_drop}")
print("    CUST_ID – מזהה, לא תכונה מספרית")
print("    TENURE – כמעט אחיד (רוב הלקוחות 12 חודשים)")

# Standardization
# הסבר למידה:
# StandardScaler מבצע: X_scaled = (X - mean) / std
# כלומר כל תכונה מקבלת ממוצע 0 וסטיית תקן 1.
# זה קריטי ב-K-Means כי האלגוריתם מחשב מרחקים –
# ובלי scaling, תכונות עם ערכים גדולים ישלטו בתוצאה.

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clustering)

print(f"\n  בוצע StandardScaler:")
print(f"    לפני – טווח BALANCE: [{df_clustering['BALANCE'].min():,.0f}, {df_clustering['BALANCE'].max():,.0f}]")
print(f"    לפני – טווח PURCHASES_FREQUENCY: [{df_clustering['PURCHASES_FREQUENCY'].min():.2f}, {df_clustering['PURCHASES_FREQUENCY'].max():.2f}]")
print(f"    אחרי – כל התכונות: ממוצע ≈ 0, סטיית תקן ≈ 1")


# ============================================================
# חלק ד' של התרגיל – מציאת מספר הקלאסטרים (Elbow Method)
# ============================================================
# הסבר למידה:
# שיטת ה-Elbow עוזרת לנו לבחור את מספר הקלאסטרים (k) האופטימלי.
# הרעיון: נריץ K-Means עם ערכי k שונים (2 עד 10) ונמדוד את ה-inertia
# (סכום ריבועי המרחקים של כל נקודה מהמרכז של הקלאסטר שלה).
#
# ככל ש-k גדל, ה-inertia יורד (כי יש יותר קלאסטרים קטנים).
# אנחנו מחפשים את "המרפק" – הנקודה שבה קצב הירידה מואט משמעותית.
# מעבר לנקודה זו, הוספת קלאסטרים נוספים לא משפרת הרבה.

print("\n\n--- חלק ד': מציאת מספר הקלאסטרים – Elbow Method ---")

inertias = []
K_range = range(2, 11)

for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(X_scaled)
    inertias.append(kmeans_temp.inertia_)
    print(f"  k={k}: Inertia = {kmeans_temp.inertia_:,.0f}")

# גרף Elbow
fig_elbow, ax_elbow = plt.subplots(figsize=(10, 6))
ax_elbow.plot(list(K_range), inertias, "bo-", linewidth=2, markersize=8)
ax_elbow.set_title("Elbow Method – Optimal Number of Clusters", fontsize=15, fontweight="bold")
ax_elbow.set_xlabel("Number of Clusters (k)", fontsize=12)
ax_elbow.set_ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=12)
ax_elbow.set_xticks(list(K_range))
ax_elbow.grid(True, alpha=0.3)

# סימון הנקודה הנבחרת
optimal_k = 4
ax_elbow.axvline(x=optimal_k, color="red", linestyle="--", linewidth=2, label=f"Chosen k={optimal_k}")
ax_elbow.legend(fontsize=12)

plt.tight_layout()
plt.savefig("clustering_elbow.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"\n  -> הגרף נשמר בקובץ clustering_elbow.png")

print(f"\n  בחירה: k = {optimal_k}")
print("  הסבר: בגרף ה-Elbow רואים שהירידה ב-Inertia מואטת משמעותית")
print("  אחרי k=4. כלומר, מ-4 קלאסטרים ומעלה, ההוספה של קלאסטר נוסף")
print("  לא משפרת הרבה את ההפרדה. לכן k=4 הוא האיזון הטוב ביותר")
print("  בין פשטות המודל לבין איכות הפילוח.")


# ============================================================
# חלק ה' של התרגיל – אימון KMeans ופרופילינג של קלאסטרים
# ============================================================
# הסבר למידה:
# K-Means הוא אלגוריתם Unsupervised Learning (למידה לא מפוקחת).
# השלבים של האלגוריתם:
#   1. בחר k נקודות מרכז (centroids) אקראיות
#   2. שייך כל נקודת נתונים למרכז הקרוב אליה
#   3. חשב מחדש את המרכזים (ממוצע כל הנקודות בקלאסטר)
#   4. חזור על 2-3 עד שהמרכזים לא משתנים (התכנסות)
#
# random_state=42 מבטיח תוצאות זהות בכל הרצה.

print("\n\n--- חלק ה': אימון KMeans ופרופילינג ---")

# שלב 1: אימון המודל
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# שלב 2: הוספת עמודת Cluster ל-DataFrame
df_customers["Cluster"] = clusters

print(f"\n  מודל KMeans אומן עם k={optimal_k}")
print(f"  כל לקוח קיבל תווית קלאסטר (0 עד {optimal_k - 1})")

# שלב 3: מאפיינים ממוצעים לכל קלאסטר
print("\n  --- גודל כל קלאסטר ---")
cluster_sizes = df_customers["Cluster"].value_counts().sort_index()
for cluster_id, size in cluster_sizes.items():
    pct = size / len(df_customers) * 100
    print(f"    Cluster {cluster_id}: {size} לקוחות ({pct:.1f}%)")

print("\n  --- ממוצעים לכל קלאסטר (תכונות עיקריות) ---")
profile_cols = ["BALANCE", "PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES",
                "CASH_ADVANCE", "CASH_ADVANCE_TRX", "PURCHASES_TRX",
                "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS", "PRC_FULL_PAYMENT",
                "PURCHASES_FREQUENCY"]
cluster_profiles = df_customers.groupby("Cluster")[profile_cols].mean()
print(cluster_profiles.round(2).to_string())

# שלב 4: פרשנות הקלאסטרים
# הסבר למידה:
# לאחר שה-KMeans חילק את הלקוחות לקבוצות, התפקיד שלנו כאנליסטים
# הוא לתת משמעות עסקית לכל קבוצה. נבדוק את הממוצעים ונמצא דפוסים.

print("\n\n  --- פרשנות הקלאסטרים ---")

# נגדיר פרופילים על סמך הממוצעים
cluster_names = {}
cluster_descriptions = {}

for c in range(optimal_k):
    profile = cluster_profiles.loc[c]
    bal = profile["BALANCE"]
    purch = profile["PURCHASES"]
    cash = profile["CASH_ADVANCE"]
    credit = profile["CREDIT_LIMIT"]
    payments = profile["PAYMENTS"]
    freq = profile["PURCHASES_FREQUENCY"]
    full_pay = profile["PRC_FULL_PAYMENT"]

    # לוגיקת סיווג – מבוססת על הממוצעים הכלליים של כל הקלאסטרים
    overall_avg_purch = cluster_profiles["PURCHASES"].mean()
    overall_avg_cash = cluster_profiles["CASH_ADVANCE"].mean()

    if cash > purch and cash > overall_avg_cash:
        name = "Cash-Advance Seekers"
        desc = ("לקוחות שמשתמשים בעיקר במשיכות מזומן מהכרטיס.\n"
                "    יתרה גבוהה, רכישות נמוכות. סיכון אשראי גבוה יותר.")
    elif purch > overall_avg_purch * 2 and freq > 0.7:
        name = "High-Value Customers"
        desc = ("לקוחות פרימיום עם רכישות בהיקפים גבוהים מאוד ומסגרת אשראי גבוהה.\n"
                "    לקוחות רווחיים עם פוטנציאל ל-upselling ותוכניות VIP.")
    elif freq > 0.5 and purch > overall_avg_purch * 0.5:
        name = "Frequent Buyers"
        desc = ("לקוחות פעילים שמשתמשים בכרטיס באופן קבוע לרכישות.\n"
                "    תדירות גבוהה, היקף בינוני – לקוחות נאמנים ויציבים.")
    else:
        name = "Low-Activity Customers"
        desc = ("לקוחות עם פעילות נמוכה – רכישות מעטות ובתדירות נמוכה.\n"
                "    פוטנציאל לשימוש מוגבר עם תמריצים נכונים.")

    cluster_names[c] = name
    cluster_descriptions[c] = desc
    size = cluster_sizes[c]

    print(f"\n  Cluster {c} – \"{name}\" ({size} לקוחות)")
    print(f"    {desc}")
    print(f"    BALANCE: {bal:,.0f} | PURCHASES: {purch:,.0f} | CASH_ADVANCE: {cash:,.0f}")
    print(f"    CREDIT_LIMIT: {credit:,.0f} | PAYMENTS: {payments:,.0f}")
    print(f"    תדירות רכישות: {freq:.2f} | אחוז תשלום מלא: {full_pay:.2f}")


# ויזואליזציה – דשבורד קלאסטרים
print("\n\n  --- מייצר גרפי Clustering... ---")

fig_clust, axes_cl = plt.subplots(2, 3, figsize=(20, 12))
fig_clust.suptitle("K-Means Clustering – Customer Segmentation Dashboard",
                   fontsize=18, fontweight="bold", y=1.02)

cluster_colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12"]

# גרף 1: גודל כל קלאסטר (Bar chart)
bars = axes_cl[0, 0].bar(
    [f"C{i}\n{cluster_names.get(i, '')}" for i in range(optimal_k)],
    cluster_sizes.values,
    color=cluster_colors[:optimal_k], edgecolor="white"
)
for bar, val in zip(bars, cluster_sizes.values):
    axes_cl[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                       str(val), ha="center", fontweight="bold", fontsize=10)
axes_cl[0, 0].set_title("Cluster Sizes", fontsize=13, fontweight="bold")
axes_cl[0, 0].set_ylabel("Number of Customers")

# גרף 2: ממוצע BALANCE ו-PURCHASES לכל קלאסטר
x_pos = np.arange(optimal_k)
width = 0.35
axes_cl[0, 1].bar(x_pos - width/2, cluster_profiles["BALANCE"], width,
                  label="Balance", color="#3498db", alpha=0.8)
axes_cl[0, 1].bar(x_pos + width/2, cluster_profiles["PURCHASES"], width,
                  label="Purchases", color="#e74c3c", alpha=0.8)
axes_cl[0, 1].set_title("Avg Balance vs Purchases by Cluster", fontsize=13, fontweight="bold")
axes_cl[0, 1].set_xlabel("Cluster")
axes_cl[0, 1].set_ylabel("Amount")
axes_cl[0, 1].set_xticks(x_pos)
axes_cl[0, 1].set_xticklabels([f"C{i}" for i in range(optimal_k)])
axes_cl[0, 1].legend()

# גרף 3: ממוצע CASH_ADVANCE לכל קלאסטר
axes_cl[0, 2].bar([f"C{i}" for i in range(optimal_k)],
                  cluster_profiles["CASH_ADVANCE"],
                  color=cluster_colors[:optimal_k], edgecolor="white")
axes_cl[0, 2].set_title("Avg Cash Advance by Cluster", fontsize=13, fontweight="bold")
axes_cl[0, 2].set_ylabel("Cash Advance Amount")

# גרף 4: Scatter PURCHASES vs BALANCE צבוע לפי קלאסטר
for c in range(optimal_k):
    mask = df_customers["Cluster"] == c
    axes_cl[1, 0].scatter(df_customers.loc[mask, "PURCHASES"],
                          df_customers.loc[mask, "BALANCE"],
                          alpha=0.4, color=cluster_colors[c], label=f"C{c}: {cluster_names.get(c, '')}",
                          s=15, edgecolor="white", linewidth=0.3)
axes_cl[1, 0].set_title("Purchases vs Balance by Cluster", fontsize=13, fontweight="bold")
axes_cl[1, 0].set_xlabel("Purchases")
axes_cl[1, 0].set_ylabel("Balance")
axes_cl[1, 0].legend(fontsize=8)

# גרף 5: ממוצע CREDIT_LIMIT ו-PAYMENTS לכל קלאסטר
axes_cl[1, 1].bar(x_pos - width/2, cluster_profiles["CREDIT_LIMIT"], width,
                  label="Credit Limit", color="#2ecc71", alpha=0.8)
axes_cl[1, 1].bar(x_pos + width/2, cluster_profiles["PAYMENTS"], width,
                  label="Payments", color="#9b59b6", alpha=0.8)
axes_cl[1, 1].set_title("Avg Credit Limit vs Payments by Cluster", fontsize=13, fontweight="bold")
axes_cl[1, 1].set_xlabel("Cluster")
axes_cl[1, 1].set_ylabel("Amount")
axes_cl[1, 1].set_xticks(x_pos)
axes_cl[1, 1].set_xticklabels([f"C{i}" for i in range(optimal_k)])
axes_cl[1, 1].legend()

# גרף 6: Radar-style comparison (תדירויות ואחוזים ממוצעים)
freq_cols = ["PURCHASES_FREQUENCY", "ONEOFF_PURCHASES_FREQUENCY",
             "PURCHASES_INSTALLMENTS_FREQUENCY", "CASH_ADVANCE_FREQUENCY", "PRC_FULL_PAYMENT"]
freq_means = df_customers.groupby("Cluster")[freq_cols].mean()
freq_means.T.plot(kind="bar", ax=axes_cl[1, 2], color=cluster_colors[:optimal_k],
                  edgecolor="white", alpha=0.85)
axes_cl[1, 2].set_title("Avg Frequencies & Ratios by Cluster", fontsize=13, fontweight="bold")
axes_cl[1, 2].set_ylabel("Average Value (0-1)")
axes_cl[1, 2].legend([f"C{i}" for i in range(optimal_k)], fontsize=8)
axes_cl[1, 2].tick_params(axis="x", rotation=25)

plt.tight_layout()
plt.savefig("clustering_dashboard.png", dpi=150, bbox_inches="tight")
plt.show()
print("  -> הגרפים נשמרו בקובץ clustering_dashboard.png")


# ============================================================
# חלק ו' של התרגיל – סיכום ומסקנות עסקיות
# ============================================================

print("\n\n" + "=" * 60)
print("חלק ו': סיכום ומסקנות עסקיות")
print("=" * 60)

print("""
  1. תובנות עסקיות מהתפלגות הקלאסטרים:
  ─────────────────────────────────────
  המודל חילק את הלקוחות ל-4 סגמנטים מובחנים:
""")

for c in range(optimal_k):
    print(f"  * Cluster {c} – \"{cluster_names[c]}\" ({cluster_sizes[c]} לקוחות, {cluster_sizes[c]/len(df_customers)*100:.1f}%)")

print("""
  התובנה המרכזית: לקוחות כרטיסי אשראי אינם קבוצה הומוגנית.
  יש הבדלים מהותיים בין דפוסי השימוש – חלקם משתמשים לרכישות תכופות,
  חלקם למשיכות מזומן, חלקם הם לקוחות פרימיום, וחלקם כמעט לא פעילים.

  2. המלצות ליישום – שיווק ממוקד, ניהול סיכונים ומבצעים:
  ──────────────────────────────────────────────────────────
""")

recommendations = {
    "Cash-Advance Seekers": (
        "    שיווק: הצעת הלוואות אישיות בריבית נמוכה יותר כחלופה למשיכות מזומן.\n"
        "    סיכון: מעקב צמוד – משיכות מזומן תכופות מצביעות על בעיות תזרים.\n"
        "    מבצע: בונוס על מעבר לרכישות רגילות במקום משיכות מזומן."
    ),
    "Frequent Buyers": (
        "    שיווק: תוכנית נאמנות עם נקודות/cashback על רכישות.\n"
        "    סיכון: נמוך – לקוחות יציבים שמשלמים בזמן.\n"
        "    מבצע: הנחות בחנויות פופולריות, שדרוג כרטיס עם הטבות."
    ),
    "High-Value Customers": (
        "    שיווק: שירות VIP, כרטיס פרימיום עם הטבות בלעדיות.\n"
        "    סיכון: נמוך – אבל חשוב לשמור על שביעות רצון כדי שלא יעזבו.\n"
        "    מבצע: הזמנות לאירועים, גישה ללאונג'ים בשדות תעופה, ביטוח נסיעות."
    ),
    "Low-Activity Customers": (
        "    שיווק: קמפיין הפעלה – 'חזור להשתמש בכרטיס וקבל X'.\n"
        "    סיכון: סיכון לנטישה (churn) – צריך לפעול לפני שעוזבים.\n"
        "    מבצע: 0% עמלה על 3 חודשים ראשונים, cashback מוגבר."
    ),
}

for c in range(optimal_k):
    name = cluster_names[c]
    print(f"  Cluster {c} – \"{name}\":")
    if name in recommendations:
        print(recommendations[name])
    print()

print("  לסיכום: מודל K-Means מאפשר לחברת CardWise לעבור משיווק גורף")
print("  לשיווק ממוקד (Targeted Marketing), מה שצפוי לשפר את שימור")
print("  הלקוחות, להגדיל הכנסות, ולנהל סיכוני אשראי בצורה חכמה יותר.")
print("\n" + "=" * 60)
print("סוף התרגיל – Clustering K-Means")
print("=" * 60)
