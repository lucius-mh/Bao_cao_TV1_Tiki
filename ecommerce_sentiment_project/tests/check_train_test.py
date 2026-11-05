import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]

print("--- KIỂM TRA CHIA TẬP DỮ LIỆU TRAIN/TEST ---")

# 1. Đọc dữ liệu
df = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'reviews_clean.csv')
df = df.dropna(subset=['text_clean', 'label'])
print(f"Tổng số mẫu hợp lệ: {len(df)}\n")

# 2. Chia tập Train/Test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    df['text_clean'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# 3. In kết quả
print(f"Số lượng tập Train (Huấn luyện): {X_train.shape[0]} mẫu")
print(f"Số lượng tập Test (Kiểm thử): {X_test.shape[0]} mẫu\n")

print("Phân bố nhãn (0: Tiêu cực, 1: Trung lập, 2: Tích cực) trong tập TRAIN:")
print(y_train.value_counts().sort_index())

print("\nPhân bố nhãn trong tập TEST:")
print(y_test.value_counts().sort_index())