import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 1. Đọc dữ liệu đã làm sạch
df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "reviews_clean.csv")

# Kiểm tra xem cột văn bản và cột nhãn tên là gì (thường là text_seg/text_clean và sentiment/label)
text_column = "text_seg" if "text_seg" in df.columns else "text"
label_column = "sentiment" if "sentiment" in df.columns else "label"

X = df[text_column].astype(str)
y = df[label_column]

# Chia tập test (tương tự tỷ lệ lúc nhóm huấn luyện, ví dụ test_size = 0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Load Vectorizer và các Mô hình đã lưu (.pkl)
try:
    vectorizer = pickle.load(open(PROJECT_ROOT / "models" / "tfidf_vectorizer.pkl", "rb"))
    model_lr = pickle.load(open(PROJECT_ROOT / "models" / "logistic_regression_model.pkl", "rb"))
    model_nb = pickle.load(open(PROJECT_ROOT / "models" / "naive_bayes_model.pkl", "rb"))
except Exception as e:
    print(f"Lỗi khi load file mô hình: {e}")
    exit()

# Biến đổi tập test thành dạng vector TF-IDF
X_test_vec = vectorizer.transform(X_test)

# 3. Đánh giá Mô hình Logistic Regression
print("================ CHỈ SỐ LOGISTIC REGRESSION ================")
y_pred_lr = model_lr.predict(X_test_vec)
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}\n")
print(classification_report(y_test, y_pred_lr))

# 4. Đánh giá Mô hình Naive Bayes
print("\n================ CHỈ SỐ NAIVE BAYES ================")
y_pred_nb = model_nb.predict(X_test_vec)
print(f"Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}\n")
print(classification_report(y_test, y_pred_nb))