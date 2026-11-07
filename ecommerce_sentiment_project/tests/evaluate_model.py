import os
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "reviews_clean.csv"
MODEL_DIR = PROJECT_ROOT / "models"

print(f"Reading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)


text_column = "text_clean" if "text_clean" in df.columns else "text"
label_column = "sentiment" if "sentiment" in df.columns else "label"

X = df[text_column].astype(str)
# Lấy nhãn thực tế từ dataset
y = df[label_column]

# Nếu nhãn đang là dạng chữ (String), tiến hành map sang dạng số để khớp với model (0, 1, 2)
if y.dtype == object or y.isin(['Negative', 'Neutral', 'Positive', 'Tiêu cực', 'Trung lập', 'Tích cực']).any():
    mapping_dict = {
        'Negative': 0, 'Neutral': 1, 'Positive': 2,
        'Tiêu cực': 0, 'Trung lập': 1, 'Tích cực': 2
    }
    y = y.map(mapping_dict)

# Ép kiểu sang số nguyên (int) để tránh lỗi kiểu dữ liệu
y = y.astype(int)

# Chia tap Test dung theo ty le va random_state luc train (80/20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -NAP MO HINH
with open(MODEL_DIR / "tfidf_vectorizer.pkl", "rb") as f:
  vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
with open(MODEL_DIR / "logistic_regression_model.pkl", "rb") as f:
  model_lr = joblib.load(MODEL_DIR / "logistic_regression_model.pkl")
with open(MODEL_DIR / "naive_bayes_model.pkl", "rb") as f:
  model_nb = joblib.load(MODEL_DIR / "naive_bayes_model.pkl")

# Bien doi tap Test thanh ma tran TF-IDF
X_test_vec = vectorizer.transform(X_test)

#DANH GIA LOGISTIC REGRESSION
print("\n" + "=" * 10 + " EVALUATION: LOGISTIC REGRESSION " + "=" * 10)
y_pred_lr = model_lr.predict(X_test_vec)
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}\n")
print(classification_report(y_test, y_pred_lr))

#DANH GIA NAIVE BAYES
print("\n" + "=" * 10 + " EVALUATION: NAIVE BAYES " + "=" * 10)
y_pred_nb = model_nb.predict(X_test_vec)
print(f"Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}\n")
print(classification_report(y_test, y_pred_nb))

#VE MA TRAN NHAM LAN (CONFUSION MATRIX)
plt.figure(figsize=(8, 6))
labels_unique = sorted(y_test.unique())
cm_lr = confusion_matrix(y_test, y_pred_lr, labels=labels_unique)
sns.heatmap(
    cm_lr,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels_unique,
    yticklabels=labels_unique,
)
plt.title(
    "Confusion Matrix - Logistic Regression", fontsize=12, fontweight="bold"
)
plt.xlabel("Predicted Label", fontsize=10)
plt.ylabel("Actual Label", fontsize=10)
plt.tight_layout()

plt.figure(figsize=(8, 6))
cm_nb = confusion_matrix(y_test, y_pred_nb, labels=labels_unique)
sns.heatmap(
    cm_nb,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=labels_unique,
    yticklabels=labels_unique,
)
plt.title("Confusion Matrix - Naive Bayes", fontsize=12, fontweight="bold")
plt.xlabel("Predicted Label", fontsize=10)
plt.ylabel("Actual Label", fontsize=10)
plt.tight_layout()

plt.show()