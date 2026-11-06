import re
import unicodedata
from functools import lru_cache
from pathlib import Path
 
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
 
 
# ==========================================
# 1. CẤU HÌNH TRANG
# ==========================================
 
st.set_page_config(
    page_title="Hệ thống phân tích cảm xúc",
    page_icon="🛒",
    layout="wide"
)
 
st.markdown("""
<style>
.main {
    background-color: #0e1117;
    color: #ffffff;
}
 
.stMetric {
    background-color: #161b22;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #30363d;
}
 
div[data-testid="stMetricValue"] {
    color: #ffffff;
}
 
.stButton > button {
    border-radius: 8px;
}
 
section[data-testid="stSidebar"] {
    background-color: #11151c;
}
 
</style>
""", unsafe_allow_html=True)
 
 
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
RETRAINED_DIR = MODEL_DIR / "retrained"   # nếu có thì ưu tiên (khớp tiểu luận)
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "reviews_clean.csv"
STOPWORDS_FILE = PROJECT_ROOT / "data" / "raw" / "vietnamese-stopwords.txt"
MODEL_FILES = (
    "tfidf_vectorizer.pkl",
    "logistic_regression_model.pkl",
    "naive_bayes_model.pkl",
)
MAX_BATCH_DEFAULT = 3000
 
 
# ==========================================
# 2. TIỀN XỬ LÝ VĂN BẢN (GIỐNG LÚC HUẤN LUYỆN)
# ==========================================
# Các bước giống notebook Bao_cao_TV1_Tiki.ipynb:
# chuẩn hóa -> tách từ -> bỏ stopwords (giữ từ sắc thái) -> dự phòng text_seg.
 
ABBREVIATIONS = {
    "ko": "không", "k": "không", "kh": "không", "hok": "không", "hông": "không",
    "dc": "được", "đc": "được",
    "sp": "sản phẩm", "mn": "mọi người", "vs": "với",
}
 
# Từ mang sắc thái/phủ định: KHÔNG bị loại khỏi câu
KEEP_WORDS = {
    "không", "chưa", "chẳng", "chả", "đừng", "không_bao_giờ", "chưa_bao_giờ",
    "rất", "quá", "hơi", "khá", "thực_sự", "nhiều", "ít", "hơn", "thêm", "giảm", "tăng",
    "nên", "cần", "phải", "thiếu", "muốn",
    "tốt", "hay", "khó", "dễ", "đủ", "đúng", "rõ", "thích", "được", "nhanh", "lâu", "mới",
    "lớn", "nhỏ", "đẹp", "xấu", "kém", "tệ", "chậm", "rẻ", "đắt", "bền", "hỏng", "lỗi", "sai",
    "ưng", "hài_lòng", "thất_vọng", "tiếc", "ngon",
}
 
 
def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)      # xóa URL
    text = re.sub(r"([^\W\d_])\1{2,}", r"\1", text)         # "ngonnnn" -> "ngon"
    text = re.sub(r"[^\w\s]|_", " ", text)                  # bỏ dấu câu, emoji
    words = [ABBREVIATIONS.get(w, w) for w in text.split()]
    return " ".join(words)
 
 
@lru_cache(maxsize=1)
def load_stopwords():
    if not STOPWORDS_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {STOPWORDS_FILE}. Cần file này để tiền xử lý giống lúc huấn luyện."
        )
    with open(STOPWORDS_FILE, "r", encoding="utf-8") as f:
        sw = {
            unicodedata.normalize("NFC", line.strip().lower()).replace(" ", "_")
            for line in f if line.strip()
        }
    return frozenset(sw - KEEP_WORDS)
 
 
def clean_text(text):
    """Trả về text_clean; chuỗi rỗng nghĩa là không còn nội dung hợp lệ."""
    if not isinstance(text, str) or not text.strip():
        return ""
    try:
        from underthesea import word_tokenize
    except ImportError as exc:
        raise ImportError("Thiếu thư viện underthesea. Chạy: pip install underthesea") from exc
    seg = word_tokenize(normalize_text(text), format="text")
    if not seg.strip():
        return ""
    stop = load_stopwords()
    clean = " ".join(w for w in seg.split() if w not in stop)
    return clean if clean.strip() else seg
 
 
# ==========================================
# 3. ÁNH XẠ NHÃN CẢM XÚC
# ==========================================
# Quy ước: 0 = Tiêu cực, 1 = Trung lập, 2 = Tích cực
 
SENTIMENT_MAP = {
    0: {"label": "Tiêu cực", "icon": "⚠️", "hex": "#ff4b4b"},
    1: {"label": "Trung lập", "icon": "⚪", "hex": "#ffa154"},
    2: {"label": "Tích cực", "icon": "🌟", "hex": "#00cc96"},
}
LABEL_IDS = [0, 1, 2]
LABEL_NAMES = [SENTIMENT_MAP[i]["label"] for i in LABEL_IDS]
 
_NAME_TO_ID = {
    "negative": 0, "neutral": 1, "positive": 2,
    "tiêu cực": 0, "trung lập": 1, "tích cực": 2,
}
 
 
def to_label_id(value):
    """Nhận nhãn số (0/1/2, '1') hoặc chữ (Negative/Neutral/Positive/Tích cực...) -> 0/1/2."""
    s = str(value).strip().lower()
    if s in _NAME_TO_ID:
        return _NAME_TO_ID[s]
    try:
        v = int(float(s))
        if v in SENTIMENT_MAP:
            return v
    except ValueError:
        pass
    raise ValueError(f"Nhãn không hợp lệ: {value!r}")
 
 
def get_sentiment_info(pred):
    return SENTIMENT_MAP[to_label_id(pred)]
 
 
def get_label_column(data):
    if "label" in data.columns:
        return "label"
    if "sentiment" in data.columns:
        return "sentiment"
    return None
 
 
# ==========================================
# 4. TẢI MÔ HÌNH VÀ DỮ LIỆU
# ==========================================
 
@st.cache_resource
def load_data_and_models():
    model_dir = None
    for d in (RETRAINED_DIR, MODEL_DIR):
        if all((d / f).exists() for f in MODEL_FILES):
            model_dir = d
            break
    if model_dir is None:
        raise FileNotFoundError("Không tìm thấy đủ 3 file .pkl trong thư mục models/.")
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp: {DATA_FILE}")
 
    vectorizer = joblib.load(model_dir / MODEL_FILES[0])
    model_lr = joblib.load(model_dir / MODEL_FILES[1])
    model_nb = joblib.load(model_dir / MODEL_FILES[2])
    df = pd.read_csv(DATA_FILE)
    return vectorizer, model_lr, model_nb, df, model_dir
 
 
try:
    vectorizer, model_lr, model_nb, df, model_dir = load_data_and_models()
except Exception as e:
    st.error("Không thể khởi động hệ thống.")
    st.exception(e)
    st.stop()
 
MATCHES_REPORT = (
    vectorizer.max_features == 20000
    and model_lr.get_params().get("class_weight") == "balanced"
)
 
 
# ==========================================
# 5. HÀM DỰ ĐOÁN VÀ XÁC SUẤT
# ==========================================
 
def predict_proba_aligned(model, X):
    """Xác suất theo thứ tự cố định [Tiêu cực, Trung lập, Tích cực]."""
    raw = model.predict_proba(X)
    out = np.zeros((raw.shape[0], 3))
    for col, cls in enumerate(model.classes_):
        out[:, to_label_id(cls)] = raw[:, col]
    return out
 
 
def predict_many(model, texts_clean):
    """Trả về (mảng nhãn 0/1/2, ma trận xác suất n x 3)."""
    proba = predict_proba_aligned(model, vectorizer.transform(list(texts_clean)))
    return proba.argmax(axis=1), proba
 
 
def predict_sentiment(model, processed_text):
    ids, proba = predict_many(model, [processed_text])
    probs = proba[0]
    probability_data = {SENTIMENT_MAP[i]["label"]: float(probs[i]) * 100 for i in LABEL_IDS}
    return int(ids[0]), float(probs.max()) * 100, probability_data
 
 
def render_result(model_name, prediction, confidence, probabilities):
    info = get_sentiment_info(prediction)
 
    st.markdown(f"### {model_name}")
    st.markdown(f"## {info['icon']} {info['label']}")
    st.caption(f"Mã nhãn: {prediction}")
    st.markdown(f"**Xác suất dự đoán: {confidence:.2f}%**")
    st.progress(min(int(confidence), 100))
    st.markdown("**Xác suất theo từng nhãn:**")
    st.dataframe(
        pd.DataFrame({
            "Nhãn cảm xúc": list(probabilities.keys()),
            "Xác suất (%)": [round(v, 2) for v in probabilities.values()],
        }),
        hide_index=True,
        use_container_width=True,
    )
    st.caption("Đây là xác suất mô hình ước lượng, không phải độ chính xác thực tế.")
 
 
@st.cache_data
def evaluate_models(model_key):
    """Đánh giá trên tập kiểm thử (80/20, random_state=42, phân tầng) như lúc huấn luyện."""
    label_col = get_label_column(df)
    data = df.dropna(subset=["text_clean", label_col])
    y = data[label_col].map(to_label_id)
    _, X_test, _, y_test = train_test_split(
        data["text_clean"].astype(str), y, test_size=0.2, random_state=42, stratify=y
    )
    out = {"n_test": len(y_test)}
    for name, model in (("Naive Bayes", model_nb), ("Logistic Regression", model_lr)):
        pred, _ = predict_many(model, X_test)
        p, r, f, _s = precision_recall_fscore_support(y_test, pred, labels=LABEL_IDS, zero_division=0)
        out[name] = {
            "accuracy": accuracy_score(y_test, pred), "p": p, "r": r, "f": f,
            "cm": confusion_matrix(y_test, pred, labels=LABEL_IDS),
        }
    return out
 
 
# ==========================================
# 6. THANH ĐIỀU HƯỚNG
# ==========================================
 
st.sidebar.title("⚙️ Điều Hướng Hệ Thống")
 
app_mode = st.sidebar.radio(
    "Chọn khu vực chức năng:",
    [
        "📊 01. Trang Tổng Quan",
        "🔍 02. Phân Tích Cảm Xúc",
        "📁 03. Phân Tích Hàng Loạt",
        "🏆 04. Đánh Giá Mô Hình"
    ]
)
 
st.sidebar.markdown("---")
 
st.sidebar.info("""
**Đề tài 5**
 
Phân tích cảm xúc đánh giá sản phẩm thương mại điện tử Tiki/Shopee và hệ thống phân loại phản hồi.
""")
 
st.sidebar.markdown("---")
st.sidebar.caption(f"Bộ mô hình đang dùng: `{Path(model_dir).relative_to(PROJECT_ROOT)}`")
if not MATCHES_REPORT:
    st.sidebar.warning(
        "Bộ mô hình này khác cấu hình trong tiểu luận (TF-IDF 20.000, LR balanced), "
        "nên số liệu có thể không trùng báo cáo."
    )
st.sidebar.caption("Hệ thống phân tích cảm xúc bằng học máy")
 
 
# ==========================================
# 7. TRANG TỔNG QUAN
# ==========================================
 
if app_mode == "📊 01. Trang Tổng Quan":
 
    st.title("📊 Trang Tổng Quan")
    st.markdown("Tổng quan dữ liệu đánh giá sách trên Tiki.")
 
    total_comments = len(df)
    label_column = get_label_column(df)
 
    pos_count = neu_count = neg_count = 0
    if label_column is not None:
        ids = df[label_column].map(to_label_id)
        pos_count = int((ids == 2).sum())
        neu_count = int((ids == 1).sum())
        neg_count = int((ids == 0).sum())
 
    pos_pct = pos_count / total_comments * 100 if total_comments else 0
    neu_pct = neu_count / total_comments * 100 if total_comments else 0
    neg_pct = neg_count / total_comments * 100 if total_comments else 0
 
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📈 Tổng số bình luận", f"{total_comments:,}")
    with col2:
        st.metric("🌟 Tích cực", f"{pos_pct:.1f}%", f"{pos_count:,} bình luận", delta_color="off")
    with col3:
        st.metric("⚪ Trung lập", f"{neu_pct:.1f}%", f"{neu_count:,} bình luận", delta_color="off")
    with col4:
        st.metric("⚠️ Tiêu cực", f"{neg_pct:.1f}%", f"{neg_count:,} bình luận", delta_color="off")
 
    st.markdown("---")
    col_chart, col_info = st.columns([2, 1])
 
    with col_chart:
        st.subheader("📊 Biểu đồ phân bố cảm xúc")
        if label_column is not None:
            chart_data = pd.DataFrame({
                "Nhãn": ["Tiêu cực", "Trung lập", "Tích cực"],
                "Số lượng": [neg_count, neu_count, pos_count],
            })
            st.bar_chart(chart_data.set_index("Nhãn"))
            if min(neg_count, neu_count, pos_count) > 0:
                st.caption(
                    f"Lớp lớn nhất gấp khoảng {max(neg_count, neu_count, pos_count) / min(neg_count, neu_count, pos_count):.1f} "
                    "lần lớp nhỏ nhất: dữ liệu mất cân bằng nên cần xem F1-macro và Recall, không chỉ Accuracy."
                )
        else:
            st.warning("Không tìm thấy cột label hoặc sentiment trong dữ liệu.")
 
    with col_info:
        st.subheader("🛒 Thông tin dữ liệu")
        st.markdown("""
        - **Nguồn dữ liệu:** Tiki Books Dataset (đánh giá sách trên Tiki).
        - **Định dạng:** CSV.
        - **Mô hình:** Logistic Regression.
        - **Mô hình so sánh:** Naive Bayes.
        - **Phương pháp:** TF-IDF (unigram + bigram).
        """)
 
    if "rating" in df.columns:
        st.subheader("⭐ Phân bố số sao")
        st.bar_chart(df["rating"].value_counts().sort_index())
 
    st.markdown("---")
    st.subheader("📄 Xem trước dữ liệu")
    st.dataframe(df.head(10), use_container_width=True)
 
 
# ==========================================
# 8. PHÂN TÍCH CẢM XÚC TỪNG CÂU
# ==========================================
 
elif app_mode == "🔍 02. Phân Tích Cảm Xúc":
 
    st.title("🔍 Phân Tích Cảm Xúc Từng Câu")
    st.markdown("Nhập bình luận để so sánh kết quả phân loại của hai mô hình.")
 
    col_ex1, col_ex2, col_ex3 = st.columns(3)
 
    with col_ex1:
        if st.button("🌟 Mẫu Tích Cực", use_container_width=True):
            st.session_state.review_input = (
                "Sản phẩm dùng rất tốt, đóng gói cẩn thận, giao hàng cực kỳ nhanh!"
            )
    with col_ex2:
        if st.button("⚪ Mẫu Trung Lập", use_container_width=True):
            st.session_state.review_input = (
                "Sản phẩm tạm ổn, giá tiền đi đôi với chất lượng, bình thường."
            )
    with col_ex3:
        if st.button("⚠️ Mẫu Tiêu Cực", use_container_width=True):
            st.session_state.review_input = (
                "Hàng bị móp méo, chất lượng tệ, shop làm ăn tắc trách không đúng mô tả."
            )
 
    user_input = st.text_area(
        "Nhập nội dung bình luận:",
        key="review_input",
        placeholder="Ví dụ: Sản phẩm dùng tạm được, không có gì đặc sắc...",
        height=120
    )
 
    if st.button("🚀 Chạy Phân Tích Song Song", type="primary", use_container_width=True):
 
        if not user_input.strip():
            st.warning("Vui lòng nhập nội dung bình luận.")
        else:
            try:
                processed_text = clean_text(user_input)
                if not processed_text:
                    st.warning("Nội dung không còn dữ liệu sau khi xử lý (ví dụ chỉ có emoji/ký tự đặc biệt).")
                else:
                    pred_lr, prob_lr, probs_lr = predict_sentiment(model_lr, processed_text)
                    pred_nb, prob_nb, probs_nb = predict_sentiment(model_nb, processed_text)
                    st.markdown("### 🔎 Kiểm tra dữ liệu đầu vào")
                    st.write("Văn bản gốc:", user_input)
                    st.write("Sau tiền xử lý:", processed_text)

                    st.subheader("🏷️ Các nhãn cảm xúc của mô hình")

                    label_df = pd.DataFrame({
                    "Mã nhãn": [0, 1, 2],
                    "Tên cảm xúc": ["Tiêu cực", "Trung lập", "Tích cực"]
                    })

                    st.dataframe(label_df, hide_index=True, use_container_width=True)

                    st.caption("Logistic Regression và Naive Bayes đều sử dụng 3 nhãn cảm xúc trên.")

                    st.write("Xác suất Logistic Regression:", probs_lr)
                    st.write("Xác suất Naive Bayes:", probs_nb)
                    st.markdown("---")
                    st.subheader("🎯 Kết Quả Phân Tích")
                    col1, col2 = st.columns(2)
                    with col1:
                        render_result("🔹 Logistic Regression", pred_lr, prob_lr, probs_lr)
                    with col2:
                        render_result("🔹 Naive Bayes", pred_nb, prob_nb, probs_nb)
 
                    if pred_lr == pred_nb:
                        st.success("Hai mô hình đưa ra cùng một kết quả dự đoán.")
                    else:
                        st.warning(
                            "Hai mô hình đưa ra kết quả khác nhau. "
                            "Bạn nên xem xét nội dung bình luận và xác suất từng nhãn."
                        )
                    with st.expander("Văn bản sau tiền xử lý (đầu vào thực tế của mô hình)"):
                        st.code(processed_text)
            except Exception as e:
                st.error("Có lỗi trong quá trình phân tích.")
                st.exception(e)
 
 
# ==========================================
# 9. PHÂN TÍCH HÀNG LOẠT
# ==========================================
 
elif app_mode == "📁 03. Phân Tích Hàng Loạt":
 
    st.title("📁 Phân Tích Hàng Loạt")
    st.markdown("Tải lên file CSV có cột `text` để phân tích nhiều bình luận cùng lúc bằng cả hai mô hình.")
 
    uploaded_file = st.file_uploader("Tải file CSV lên đây", type=["csv"])
 
    if uploaded_file is None:
        st.info("Bạn có thể thử với file `data/processed/reviews_clean.csv`.")
    else:
        try:
            test_df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
        except Exception as e:
            st.error("Không thể đọc file CSV.")
            st.exception(e)
            test_df = None
 
        if test_df is not None:
            st.subheader("📄 Xem trước dữ liệu")
            st.dataframe(test_df.head(5), use_container_width=True)
 
            if test_df.empty:
                st.warning("File CSV không có dữ liệu.")
            elif "text" not in test_df.columns:
                st.error("File CSV bắt buộc phải có cột `text` chứa bình luận. "
                         f"Các cột hiện có: {', '.join(map(str, test_df.columns))}")
            else:
                max_rows = st.number_input(
                    "Số dòng tối đa cần phân tích (tách từ tiếng Việt khá chậm)",
                    min_value=1, max_value=len(test_df),
                    value=min(len(test_df), MAX_BATCH_DEFAULT), step=100,
                )
 
                if st.button("⚡ Chạy Phân Loại Toàn Bộ File", type="primary", use_container_width=True):
                    try:
                        work = test_df.head(int(max_rows)).copy()
                        work["text"] = work["text"].fillna("").astype(str)
                        progress = st.progress(0.0, text="Đang tiền xử lý...")
                        cleaned = []
                        for i, t in enumerate(work["text"]):
                            cleaned.append(clean_text(t))
                            if i % 50 == 0:
                                progress.progress((i + 1) / len(work), text=f"Đang tiền xử lý {i + 1}/{len(work)}")
                        progress.progress(1.0, text="Đang dự đoán...")
 
                        valid = [i for i, c in enumerate(cleaned) if c.strip()]
                        result_df = work[["text"]].copy()
                        for tag in ("LR", "NB"):
                            result_df[f"Dự đoán {tag}"] = "Không có nội dung"
                            result_df[f"Xác suất {tag} (%)"] = np.nan
                        if valid:
                            for tag, model in (("LR", model_lr), ("NB", model_nb)):
                                ids, proba = predict_many(model, [cleaned[i] for i in valid])
                                result_df.iloc[valid, result_df.columns.get_loc(f"Dự đoán {tag}")] = [
                                    SENTIMENT_MAP[i]["label"] for i in ids
                                ]
                                result_df.iloc[valid, result_df.columns.get_loc(f"Xác suất {tag} (%)")] = (
                                    proba.max(axis=1) * 100
                                ).round(2)
                        result_df["Hai mô hình khớp nhau"] = np.where(
                            result_df["Dự đoán LR"] == result_df["Dự đoán NB"], "Có", "Không"
                        )
                        progress.empty()
                        st.session_state["batch_result"] = (result_df, len(valid), len(work) - len(valid))
                    except Exception as e:
                        st.error("Có lỗi trong quá trình phân tích hàng loạt.")
                        st.exception(e)
 
                if "batch_result" in st.session_state:
                    result_df, n_ok, n_skip = st.session_state["batch_result"]
                    st.success(
                        f"Đã phân tích {n_ok:,} bình luận hợp lệ."
                        + (f" Bỏ qua {n_skip:,} dòng rỗng/không hợp lệ." if n_skip else "")
                    )
                    st.markdown("---")
                    st.subheader("📊 Thống kê kết quả")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Logistic Regression**")
                        st.bar_chart(result_df["Dự đoán LR"].value_counts().reindex(LABEL_NAMES, fill_value=0))
                    with col2:
                        st.markdown("**Naive Bayes**")
                        st.bar_chart(result_df["Dự đoán NB"].value_counts().reindex(LABEL_NAMES, fill_value=0))
                    valid_rows = result_df["Dự đoán LR"] != "Không có nội dung"
                    if valid_rows.any():
                        st.metric(
                            "Tỷ lệ hai mô hình khớp nhau",
                            f"{(result_df.loc[valid_rows, 'Hai mô hình khớp nhau'] == 'Có').mean():.1%}",
                        )
 
                    st.markdown("---")
                    st.subheader("📋 Kết quả phân tích")
                    st.dataframe(result_df, use_container_width=True)
                    st.download_button(
                        label="📥 Tải Kết Quả Phân Tích (CSV)",
                        data=result_df.to_csv(index=False).encode("utf-8-sig"),
                        file_name="ket_qua_phan_tich_cam_xuc.csv",
                        mime="text/csv",
                        type="primary",
                    )
 
 
# ==========================================
# 10. ĐÁNH GIÁ VÀ SO SÁNH MÔ HÌNH
# ==========================================
 
else:
 
    st.title("🏆 Đánh Giá Và So Sánh Mô Hình")
    st.markdown("So sánh hiệu năng của Logistic Regression và Naive Bayes.")
 
    with st.spinner("Đang đánh giá trên tập kiểm thử..."):
        ev = evaluate_models(str(model_dir))
 
    st.caption(
        f"Tập kiểm thử: {ev['n_test']:,} mẫu (chia 80/20, random_state=42, phân tầng). "
        "Số liệu được tính trực tiếp từ bộ mô hình đang nạp."
    )
    if not MATCHES_REPORT:
        st.warning("Bộ mô hình đang nạp khác cấu hình tiểu luận nên số liệu có thể không trùng với báo cáo.")
 
    names = ["Naive Bayes", "Logistic Regression"]
 
    st.subheader("📋 Bảng so sánh hiệu năng")
    summary = pd.DataFrame({
        n: {
            "Accuracy": ev[n]["accuracy"],
            "Precision (macro)": ev[n]["p"].mean(),
            "Recall (macro)": ev[n]["r"].mean(),
            "F1-macro": ev[n]["f"].mean(),
        }
        for n in names
    }).round(3)
    st.dataframe(summary, use_container_width=True)
 
    st.subheader("📈 Chi tiết theo từng nhãn")
    rows = []
    for n in names:
        for k, lab in enumerate(LABEL_NAMES):
            rows.append({
                "Mô hình": n, "Nhãn": lab,
                "Precision": ev[n]["p"][k], "Recall": ev[n]["r"][k], "F1-score": ev[n]["f"][k],
            })
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, use_container_width=True)
 
    st.subheader("📊 So sánh F1-score theo nhãn")
    st.bar_chart(pd.DataFrame(
        {lab: [ev[n]["f"][k] for n in names] for k, lab in enumerate(LABEL_NAMES)}, index=names
    ))
 
    st.subheader("🧩 Ma trận nhầm lẫn")
    c1, c2 = st.columns(2)
    for col, n, cmap in ((c1, "Naive Bayes", "Greens"), (c2, "Logistic Regression", "Blues")):
        with col:
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(ev[n]["cm"], annot=True, fmt="d", cmap=cmap, cbar=False,
                        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
            ax.set_title(n)
            ax.set_xlabel("Dự đoán")
            ax.set_ylabel("Thực tế")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
 
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("⚖️ Ưu và nhược điểm")
        st.markdown("""
        **Logistic Regression**
 
        - Ưu điểm: nhận diện lớp thiểu số tốt hơn khi dùng `class_weight='balanced'`.
        - Nhược điểm: Precision lớp Trung lập còn thấp; huấn luyện lâu hơn.
 
        **Naive Bayes**
 
        - Ưu điểm: huấn luyện nhanh, đơn giản, dùng làm mốc so sánh.
        - Nhược điểm: thiên về lớp Tích cực nên bỏ sót lớp Trung lập/Tiêu cực dù Accuracy cao.
        """)
    with col2:
        st.subheader("🎯 Nhận xét")
        st.info("""
        Dữ liệu mất cân bằng mạnh, nên F1-macro và Recall từng lớp phản ánh chất lượng
        tốt hơn Accuracy. Hãy đối chiếu kết luận với bảng và ma trận nhầm lẫn ở trên.
        """)