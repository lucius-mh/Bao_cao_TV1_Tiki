import pickle
from pathlib import Path

import streamlit as st
from underthesea import word_tokenize
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Cấu hình trang
st.set_page_config(
    page_title="Phân Tích Cảm Xúc Review TMĐT", page_icon="🛒", layout="wide"
)


# Load dữ liệu và mô hình
@st.cache_resource
def load_data_and_models():
  vectorizer = pickle.load(open(PROJECT_ROOT / "models" / "tfidf_vectorizer.pkl", "rb"))
  model_lr = pickle.load(open(PROJECT_ROOT / "models" / "logistic_regression_model.pkl", "rb"))
  model_nb = pickle.load(open(PROJECT_ROOT / "models" / "naive_bayes_model.pkl", "rb"))
  df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "reviews_clean.csv")
  return vectorizer, model_lr, model_nb, df


vectorizer, model_lr, model_nb, df = load_data_and_models()

# --- SIDEBAR ---
st.sidebar.title("⚙️ Menu Điều Hướng")
app_mode = st.sidebar.radio(
    "Chọn chế độ:",
    [
        "🔍 Dự Đoán Từng Câu & So Sánh",
        "📁 Phân Tích Hàng Loạt (Batch)",
        "📊 Khám Phá Dữ Liệu",
    ],
)
st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Đề tài 5:** Phân tích cảm xúc review TMĐT Tiki/Shopee & Hệ thống phân"
    " loại phản hồi."
)

# --- TAB 1: DỰ ĐOÁN & SO SÁNH ---
if app_mode == "🔍 Dự Đoán Từng Câu & So Sánh":
  st.title("🛒 Hệ Thống Phân Tích Cảm Xúc Review Thương Mại Điện Tử")
  st.markdown(
      "Nhập bình luận của khách hàng để hệ thống chạy song song và so sánh kết"
      " quả giữa **Logistic Regression** và **Naive Bayes**."
  )

  user_input = st.text_area(
      "Nhập câu bình luận (review) của khách hàng:",
      placeholder="Ví dụ: Sản phẩm dùng rất tốt, giao hàng nhanh, rất đáng tiền...",
      height=120,
  )

  if st.button(
      "🚀 Chạy Phân Tích & So Sánh Song Song",
      type="primary",
      use_container_width=True,
  ):
    if user_input.strip() == "":
      st.warning("Vui lòng nhập nội dung bình luận!")
    else:
      # Xử lý văn bản
      processed_text = word_tokenize(user_input.lower(), format="text")
      X_input = vectorizer.transform([processed_text])

      # Dự đoán Logistic Regression
      pred_lr = model_lr.predict(X_input)[0]
      prob_lr = max(model_lr.predict_proba(X_input)[0]) * 100

      # Dự đoán Naive Bayes
      pred_nb = model_nb.predict(X_input)[0]
      prob_nb = max(model_nb.predict_proba(X_input)[0]) * 100

      st.markdown("---")
      st.subheader("🎯 Kết quả phân tích từ 2 mô hình:")

      col1, col2 = st.columns(2)

      with col1:
        st.markdown("### 🔹 Logistic Regression")
        if pred_lr in ["Positive", "Tích cực", 1, "1"]:
          st.success(f"🌟 Nhãn: **Tích cực**\n\nĐộ tin cậy: **{prob_lr:.2f}%**")
        else:
          st.error(f"⚠️ Nhãn: **Tiêu cực**\n\nĐộ tin cậy: **{prob_lr:.2f}%**")

      with col2:
        st.markdown("### 🔹 Naive Bayes")
        if pred_nb in ["Positive", "Tích cực", 1, "1"]:
          st.success(f"🌟 Nhãn: **Tích cực**\n\nĐộ tin cậy: **{prob_nb:.2f}%**")
        else:
          st.error(f"⚠️ Nhãn: **Tiêu cực**\n\nĐộ tin cậy: **{prob_nb:.2f}%**")

      # Hiệu ứng pháo giấy nếu review tích cực và độ tin cậy cao
      if (
          pred_lr in ["Positive", "Tích cực", 1, "1"]
          and prob_lr > 90
          and prob_nb > 90
      ):
        st.balloons()

# --- TAB 2: PHÂN TÍCH HÀNG LOẠT ---
elif app_mode == "📁 Phân Tích Hàng Loạt (Batch)":
  st.title("📁 Phân Tích Danh Sách Đánh Giá Hàng Loạt")
  st.markdown(
      "Tải lên một file CSV chứa cột `text` để hệ thống tự động phân loại tất"
      " cả các dòng cùng lúc."
  )

  uploaded_file = st.file_uploader(
      "Tải file CSV đánh giá của bạn lên đây", type=["csv"]
  )

  if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.write("📄 **Xem trước dữ liệu tải lên:**", test_df.head())

    if "text" in test_df.columns:
      if st.button("⚡ Chạy Phân T Loại Toàn Bộ File", type="primary"):
        with st.spinner("Hệ thống đang xử lý và phân tích..."):
          processed_texts = [
              word_tokenize(str(t).lower(), format="text")
              for t in test_df["text"]
          ]
          X_batch = vectorizer.transform(processed_texts)
          test_df["Predicted_Sentiment"] = model_lr.predict(X_batch)

        st.success("✅ Phân tích hàng loạt thành công!")
        st.dataframe(test_df[["text", "Predicted_Sentiment"]].head(20))

        # Thống kê biểu đồ cột trực quan
        st.markdown("### 📊 Biểu đồ thống kê kết quả:")
        sentiment_counts = test_df["Predicted_Sentiment"].value_counts()
        st.bar_chart(sentiment_counts)
    else:
      st.error(
          "⚠️ File CSV của bạn bắt buộc phải có cột tên là `text` chứa nội"
          " dung bình luận!"
      )
  else:
    st.info(
        "💡 **Mẹo:** Bạn có thể tải lại chính file"
        " `data/processed/reviews_clean.csv` để test thử tính năng này!"
    )

# --- TAB 3: KHÁM PHÁ DỮ LIỆU & BÁO CÁO CHI TIẾT ---
else:
  st.title("📈 Khám Phá Dữ Liệu & Thống Kê Báo Cáo")
  st.markdown(
      "Tổng hợp toàn bộ bảng số liệu và biểu đồ phân tích từ quy trình làm"
      " sạch, trích xuất đặc trưng đến đánh giá hiệu năng mô hình."
  )

  # --- BẢNG 1 & HÌNH 1: QUY TRÌNH LÀM SẠCH VÀ PHÂN BỐ DỮ LIỆU ---
  st.markdown("---")
  st.subheader("1. Quy Trình Làm Sạch & Phân Bố Cảm Xúc Dữ Liệu")

  col_t1, col_t2 = st.columns(2)

  with col_t1:
    st.markdown("### 📋 Bảng 1: Kết quả quy trình làm sạch dữ liệu")
    cleaning_summary = {
        "Giai đoạn xử lý": [
            "Dữ liệu thô thu thập (Raw Data)",
            "Sau khi xử lý trùng lặp & thiếu dữ liệu (Null/Duplicate)",
            "Dữ liệu hoàn chỉnh đưa vào huấn luyện (Cleaned)",
        ],
        "Số lượng (dòng)": ["80,000+", "73,587", "73,587"],
        "Trạng thái": ["Thô", "Đã làm sạch", "Sẵn sàng"],
    }
    st.dataframe(pd.DataFrame(cleaning_summary), use_container_width=True)

  with col_t2:
    st.markdown("### 📊 Hình 1: Phân bố nhãn cảm xúc")
    sentiment_counts = df["sentiment"].value_counts()
    st.bar_chart(sentiment_counts)

  # --- BẢNG 2 & HÌNH 2: ĐỘ DÀI BÌNH LUẬN THEO CẢM XÚC ---
  st.markdown("---")
  st.subheader("2. Phân Tích Độ Dài Bình Luận (Text Length)")

  if "text_length" not in df.columns:
    df["text_length"] = df["text"].astype(str).apply(lambda x: len(x.split()))

  col_l1, col_l2 = st.columns(2)

  with col_l1:
    st.markdown("### 📋 Bảng 2: Thống kê độ dài bình luận (số từ)")
    len_stats = (
        df.groupby("sentiment")["text_length"]
        .agg(["mean", "min", "max", "median"])
        .reset_index()
    )
    len_stats.columns = [
      "Nhãn cảm xúc",
      "Trung bình (Từ)",
      "Ngắn nhất",
      "Dài nhất",
      "Trung vị",
    ]
    st.dataframe(len_stats, use_container_width=True)

  with col_l2:
    st.markdown("### 📊 Hình 2: Độ dài bình luận theo nhãn cảm xúc")
    avg_len = df.groupby("sentiment")["text_length"].mean()
    st.bar_chart(avg_len)

  # --- BẢNG 3: TỪ KHÓA ĐẶC TRƯNG TIÊU BIỂU (TF-IDF) ---
  st.markdown("---")
  st.subheader("3. Từ Khóa Đặc Trưng Theo Nhãn Cảm Xúc")
  st.markdown(
      "### 📋 Bảng 3: Top các từ khóa có trọng số TF-IDF cao nhất đại diện cho"
      " từng nhóm"
  )

  keywords_data = {
      "Nhãn cảm xúc": [
          "Tích cực (Positive)",
          "Tiêu cực (Negative)",
          "Trung tính (Neutral)",
      ],
      "Từ khóa đặc trưng nổi bật (Top Keywords)": [
          "ngon, đẹp, chất_lượng, nhanh, tuyệt_vời, đóng_gói_cẩn_thận, hài_lòng",
          "kém, chậm, tệ, vỡ, hỏng, thất_vọng, không_đúng, lừa_đảo",
          "bình_thường, tạm_được, ok, xem_thêm, chưa_dùng, chấp_nhận_được",
      ],
  }
  st.dataframe(pd.DataFrame(keywords_data), use_container_width=True)

  # --- BẢNG 4 & HÌNH 3: SO SÁNH HIỆU NĂNG MÔ HÌNH ---
  st.markdown("---")
  st.subheader("4. Đánh Giá & So Sánh Hiệu Năng Mô Hình")

  col_p1, col_p2 = st.columns([3, 2])

  with col_p1:
    st.markdown("### 📋 Bảng 4: So sánh hiệu năng mô hình (Tập Test)")
    perf_data = {
        "Mô hình": ["Logistic Regression", "Naive Bayes"],
        "Accuracy (%)": [89.20, 88.67],
        "F1 (Negative)": [0.55, 0.49],
        "F1 (Positive)": [0.95, 0.94],
        "Macro Avg F1": [0.75, 0.72],
    }
    st.dataframe(pd.DataFrame(perf_data), use_container_width=True)

  with col_p2:
    st.markdown("### 🏆 Hình 3: So sánh Accuracy (%)")
    acc_df = pd.DataFrame({
        "Logistic Regression": [89.20],
        "Naive Bayes": [88.67],
    }).T
    acc_df.columns = ["Accuracy (%)"]
    st.bar_chart(acc_df)

  st.info(
      "💡 **Kết luận đánh giá:** Mô hình **Logistic Regression** đạt tổng thể"
      " Accuracy cao hơn (89.20%) và có chỉ số F1-Score vượt trội hơn hẳn ở"
      " lớp Tiêu cực (0.55 so với 0.49), do đó được lựa chọn làm mô hình chính"
      " phục vụ phân loại trên hệ thống."
  )

  st.markdown("---")
  st.subheader("📄 Xem trước dữ liệu mẫu (15 dòng đầu):")
  st.dataframe(df.head(15), use_container_width=True)