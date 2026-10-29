# -*- coding: utf-8 -*-
"""
Demo web chạy OFFLINE cho đồ án ViTHSD.

Chạy (trong thư mục local, đã kích hoạt môi trường ảo):
    streamlit run app.py
rồi mở trình duyệt tại http://localhost:8501. Dừng: Ctrl + C trong cửa sổ dòng lệnh.
"""
import json
import os

# Bật chế độ offline TRƯỚC khi import thư viện Hugging Face
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import common  # noqa: E402
from common import LEVELS, MODEL_DIR, OUT_DIR, TARGETS  # noqa: E402

st.set_page_config(page_title="ViTHSD – Phát hiện ngôn từ thù ghét", layout="wide")
TARGET_VI = {"individual": "Cá nhân", "groups": "Nhóm người", "religion/creed": "Tôn giáo",
             "race/ethnicity": "Dân tộc", "politics": "Chính trị"}
BADGE = {0: "⚪ không nhắc", 1: "🟢 clean", 2: "🟠 offensive", 3: "🔴 hate"}
MODELS = {"visobert": "ViSoBERT", "phobert": "PhoBERT-v2"}


@st.cache_resource(show_spinner="Đang huấn luyện lại baseline TF-IDF + LR (chỉ lần đầu, ~1 phút)...")
def get_baseline():
    """Tái tạo baseline trên máy với siêu tham số đã chọn trên Colab."""
    train, _, _, _ = common.load_and_clean(verbose=False)
    p = OUT_DIR / "baseline_info.json"
    info = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"best_C": 1.0, "best_class_weight": "balanced"}
    cw = None if info["best_class_weight"] in ("None", "none") else info["best_class_weight"]
    vec = common.build_vectorizer()
    models = common.fit_lr(vec.fit_transform(train["text"]), common.label_matrix(train), info["best_C"], cw)
    return vec, models


@st.cache_resource(show_spinner="Đang nạp mô hình Transformer (chỉ lần đầu)...")
def get_transformer(key):
    return common.load_finetuned(MODEL_DIR / key, "cpu")


@st.cache_resource
def get_segmenter():
    return common.make_segmenter()


def predict_all(text):
    """Trả về {tên mô hình: mảng 5 mức} cho 1 bình luận."""
    out = {}
    vec, lr_models = get_baseline()
    X = vec.transform([common.normalize_text(text)])
    out["TF-IDF + LR"] = common.predict_proba_lr(lr_models, X).argmax(-1)[0]
    for key, name in MODELS.items():
        if (MODEL_DIR / key / "model.safetensors").exists():
            model, tok, info = get_transformer(key)
            out[name] = common.predict_transformer(model, tok, info, [text], segmenter=get_segmenter())[0][0]
    return out


st.title("ViTHSD – Phát hiện ngôn từ thù ghét theo đối tượng")
st.caption("Đồ án môn CMP169 – Trí tuệ Nhân tạo · Chạy hoàn toàn offline trên máy · Mô hình có thể sai, chỉ dùng cho học tập")

missing = [k for k in MODELS if not (MODEL_DIR / k / "model.safetensors").exists()]
if missing:
    st.warning(f"Chưa có mô hình: {missing}. Giải nén ViTHSD_ket_qua.zip (từ Colab) vào thư mục local/.")

text = st.text_area("Nhập một bình luận tiếng Việt:", height=110,
                    placeholder="Ví dụ: bài hát này hay quá, nghe hoài không chán")
if st.button("Phân tích", type="primary") and text.strip():
    res = predict_all(text)
    table = pd.DataFrame({name: [BADGE[int(v)] for v in levels] for name, levels in res.items()},
                         index=[TARGET_VI[t] for t in TARGETS])
    st.dataframe(table, width="stretch")
    worst = {name: LEVELS[int(levels.max())] for name, levels in res.items()}
    st.write("**Mức nặng nhất theo từng mô hình:**", ", ".join(f"{k}: {v}" for k, v in worst.items()))

with st.expander("Kết quả trên tập test (tính lúc train)"):
    p = OUT_DIR / "comparison_test.csv"
    if p.exists():
        st.dataframe(pd.read_csv(p, index_col=0))
    img = OUT_DIR / "comparison_paper.png"
    if img.exists():
        st.image(str(img))
