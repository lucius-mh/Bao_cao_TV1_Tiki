import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

print("==================================================")

# Đọc và chuẩn bị dữ liệu
print("Đọc dữ liệu từ 'reviews_clean.csv'...")
df = pd.read_csv('reviews_clean.csv')
df = df.dropna(subset=['text_clean', 'label'])
X = df['text_clean']
y = df['label']
print(f"-> Dữ liệu hợp lệ: {len(df)} dòng.\n")

# Chia tập Train/Test (80/20)
print("Chia tập dữ liệu Train/Test (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)
print(f"-> Dữ liệu Train: {X_train.shape[0]} mẫu.")
print(f"-> Dữ liệu Test: {X_test.shape[0]} mẫu.\n")

# Trích xuất đặc trưng bằng TfidfVectorizer
print("Xây dựng ma trận đặc trưng TF-IDF (có thể mất một chút thời gian)...")
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(f"-> Kích thước ma trận Train TF-IDF: {X_train_tfidf.shape}\n")

# Huấn luyện và đánh giá mô hình
print("Huấn luyện và đánh giá mô hình:\n")

# --- Mô hình Naive Bayes ---
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_preds = nb_model.predict(X_test_tfidf)
nb_acc = accuracy_score(y_test, nb_preds)

print("--- KẾT QUẢ NAIVE BAYES ---")
print(f"Độ chính xác (Accuracy): {nb_acc:.4f}")
print(classification_report(y_test, nb_preds))

# --- Mô hình Logistic Regression ---
lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr_model.fit(X_train_tfidf, y_train)
lr_preds = lr_model.predict(X_test_tfidf)
lr_acc = accuracy_score(y_test, lr_preds)

print("--- KẾT QUẢ LOGISTIC REGRESSION ---")
print(f"Độ chính xác (Accuracy): {lr_acc:.4f}")
print(classification_report(y_test, lr_preds))

# Xuất file mô hình
print("Đang lưu các file .pkl...")
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(nb_model, 'naive_bayes_model.pkl')
joblib.dump(lr_model, 'logistic_regression_model.pkl')

print("HOÀN TẤT! Đã xuất thành công 3 file .pkl ra thư mục hiện tại.")