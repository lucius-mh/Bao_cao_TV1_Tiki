import pandas as pd
from pathlib import Path

# Đọc file dữ liệu đã làm sạch của nhóm
project_root = Path(__file__).resolve().parents[1]
df = pd.read_csv(project_root / "data" / "processed" / "reviews_clean.csv")

# 1. Kiểm tra kích thước dữ liệu (Tổng số dòng và cột)
print(f"Tổng số lượng dòng dữ liệu: {df.shape[0]}")
print(f"Tổng số cột: {df.shape[1]}")

# 2. Kiểm tra xem có dòng nào bị trống (Null / Missing values) không
print("\nSố lượng giá trị bị thiếu (Null) ở mỗi cột:")
print(df.isnull().sum())

# 3. Kiểm tra các dòng bị trùng lặp (Duplicate)
print(f"\nSố lượng dòng bị trùng lặp: {df.duplicated().sum()}")

# 4. Kiểm tra phân phối nhãn cảm xúc (Tích cực / Tiêu cực)
if "sentiment" in df.columns or "label" in df.columns:
  col_name = "sentiment" if "sentiment" in df.columns else "label"
  print(f"\nPhân phối nhãn trong cột '{col_name}':")
  print(df[col_name].value_counts())