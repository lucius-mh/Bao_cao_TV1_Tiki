from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 1. Xác định đường dẫn chuẩn bằng Pathlib
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "reviews_clean.csv"
MODEL_DIR = PROJECT_ROOT / "models"  # Thư mục chứa các file .pkl

# Đọc dữ liệu sạch
df = pd.read_csv(DATA_PATH)

# Kiểm tra tên cột văn bản và nhãn
text_column = "text_seg" if "text_seg" in df.columns else "text"
label_column = "sentiment" if "sentiment" in df.columns else "label"

X = df[text_column].astype(str)
X = df[text_column].astype(str)
y = df[label_column]

# === THÊM ĐOẠN NÀY ĐỂ ĐỒNG BỘ NHÃN VỀ DẠNG SỐ ===
mapping_dict = {
    'Negative': 0, 'Neutral': 1, 'Positive': 2,
    'Tiêu cực': 0, 'Trung lập': 1, 'Tích cực': 2
}
if y.dtype == object or y.isin(list(mapping_dict.keys())).any():
    y = y.map(mapping_dict)
y = y.astype(int)
# ===============================================

# Chia tập test (tỷ lệ 80/20, random_state=42 giống lúc train)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chia tập test (tỷ lệ 80/20, random_state=42 giống lúc train)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Load Vectorizer và các Mô hình từ thư mục models/ bằng joblib
try:
    vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
    model_lr = joblib.load(MODEL_DIR / "logistic_regression_model.pkl")
    model_nb = joblib.load(MODEL_DIR / "naive_bayes_model.pkl")
    print("Nạp mô hình và vectorizer thành công!\n")
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