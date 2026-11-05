import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]

print("--- KIỂM TRA MA TRẬN ĐẶC TRƯNG TF-IDF ---")

# 1. Chuẩn bị dữ liệu
df = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'reviews_clean.csv')
df = df.dropna(subset=['text_clean', 'label'])
X_train, _, _, _ = train_test_split(
    df['text_clean'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# 2. Xây dựng ma trận TF-IDF
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000)
X_train_tfidf = vectorizer.fit_transform(X_train)

print(f"\nKích thước ma trận Train TF-IDF: {X_train_tfidf.shape}")
print(f"-> Gồm: {X_train_tfidf.shape[0]} dòng (số câu bình luận) x {X_train_tfidf.shape[1]} cột (số từ/cụm từ).")

# 3. Trích xuất từ điển và xem dữ liệu
feature_names = vectorizer.get_feature_names_out()
print("\nXem thử 15 từ vựng/cụm từ ngẫu nhiên trong từ điển (từ cột 1000 đến 1015):")
print(feature_names[1000:1015])

print("\nTrực quan hóa một góc nhỏ của ma trận (5 câu bình luận đầu tiên x 8 từ vựng đầu tiên):")
# Chuyển một góc nhỏ của ma trận thưa (sparse) sang dạng dễ nhìn (dense)
dense_matrix = X_train_tfidf[:5, :8].todense()
matrix_df = pd.DataFrame(dense_matrix, columns=feature_names[:8])
print(matrix_df)