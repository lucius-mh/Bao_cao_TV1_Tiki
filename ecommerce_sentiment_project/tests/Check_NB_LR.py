from pathlib import Path
import joblib

# 1. Nạp (load) các file .pkl từ ổ cứng vào bộ nhớ chuẩn đường dẫn
print("Đang nạp bộ từ vựng và mô hình...")
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
MODEL_DIR = PROJECT_ROOT / "models"

vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
model_lr = joblib.load(MODEL_DIR / "logistic_regression_model.pkl")
model_nb = joblib.load(MODEL_DIR / "naive_bayes_model.pkl")
print("Nạp thành công!\n")

# 2. Tạo một vài bình luận mới chưa từng xuất hiện trong tập dữ liệu để kiểm tra thử
test_comments = [
    "Sách cực kỳ đẹp, nội dung rất hay và ý nghĩa 10 điểm.",      # Kỳ vọng: Tích cực (2)
    "Giao hàng quá chậm, bọc hàng sơ sài làm sách bị rách góc.",  # Kỳ vọng: Tiêu cực (0)
    "Sách đọc tạm được, chất lượng giấy bình thường."             # Kỳ vọng: Trung lập (1)
]

# 3. Biến đổi văn bản qua Vectorizer để máy tính hiểu được
X_test_new = vectorizer.transform(test_comments)

# 4. Yêu cầu mô hình dự đoán (0: Tiêu cực, 1: Trung lập, 2: Tích cực)
print("--- KẾT QUẢ DỰ ĐOÁN TỪ NAIVE BAYES ---")
nb_preds = model_nb.predict(X_test_new)
for text, pred in zip(test_comments, nb_preds):
    print(f"Nhãn: {pred} | Bình luận: '{text}'")

print("\n--- KẾT QUẢ DỰ ĐOÁN TỪ LOGISTIC REGRESSION ---")
lr_preds = model_lr.predict(X_test_new)
for text, pred in zip(test_comments, lr_preds):
    print(f"Nhãn: {pred} | Bình luận: '{text}'")