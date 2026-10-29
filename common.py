# -*- coding: utf-8 -*-
"""
common.py – Các hàm dùng chung cho 2 notebook của đồ án ViTHSD.

    01_EDA_Baseline.ipynb   : khám phá dữ liệu + mô hình baseline
    02_Transformer.ipynb    : fine-tune ViSoBERT / PhoBERT

Để ở một file chung nhằm bảo đảm 2 notebook đọc dữ liệu, tiền xử lý và tính
độ đo GIỐNG HỆT nhau -> so sánh baseline và Transformer mới công bằng.
"""
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support, roc_auc_score)

# ---------------------------------------------------------------------------
# 1. HẰNG SỐ
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent          # thư mục chứa file này
DATA_DIR = ROOT / "data"                        # nơi để train / dev / test
OUT_DIR = ROOT / "outputs"                      # biểu đồ, kết quả
MODEL_DIR = ROOT / "models"                     # mô hình pre-trained + đã train
SEED = 42

TARGETS = ["individual", "groups", "religion/creed", "race/ethnicity", "politics"]
LEVELS = ["none", "clean", "offensive", "hate"]   # giá trị 0, 1, 2, 3 trong dữ liệu

# Kết quả trong paper (Bảng 12, tập test, %) – tính theo cách của tác giả
PAPER_RESULTS = {
    "Bi-GRU-LSTM-CNN": 46.65, "m-BERT": 47.15, "XLM-R": 50.84, "PhoBERT (v1)": 50.33,
    "VELECTRA": 46.58, "ViSoBERT": 51.12, "XLM-R + Bi-GRU-LSTM-CNN": 51.74,
    "ViSoBERT + Bi-GRU-LSTM-CNN": 51.16,
}


# ---------------------------------------------------------------------------
# 2. ĐỌC DỮ LIỆU
# ---------------------------------------------------------------------------
def read_split(name, data_dir=DATA_DIR):
    """Đọc 1 tập (train/dev/test) đã tải từ Hugging Face (sonlam1102/vithsd).

    Lưu ý: file trên Hugging Face tên train.xlsx.csv nhưng nội dung là CSV.
    Hàm này thử đọc CSV trước, nếu không được mới đọc như Excel thật.
    """
    data_dir = Path(data_dir)
    # .xlsx.csv = tên file trên Hugging Face (sonlam1102/vithsd)
    candidates = [data_dir / f"{name}.xlsx.csv", data_dir / f"{name}.csv"]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"Không thấy file {name}.xlsx.csv trong {data_dir}. Hãy tải dữ liệu từ Hugging Face "
                                "(sonlam1102/vithsd) vào thư mục data/")
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except (UnicodeDecodeError, pd.errors.ParserError):
        df = pd.read_excel(path)
    df = df.rename(columns={"Unnamed: 0": "orig_id"})
    df["content"] = df["content"].fillna("").astype(str)
    for t in TARGETS:
        df[t] = df[t].astype(int)
    return df.reset_index(drop=True)


def label_matrix(df):
    """Ma trận nhãn (số bình luận, 5)."""
    return df[TARGETS].to_numpy(dtype=np.int64)


# ---------------------------------------------------------------------------
# 3. TIỀN XỬ LÝ VĂN BẢN (quy tắc cố định -> áp dụng cho cả 3 tập, không rò rỉ)
# ---------------------------------------------------------------------------
_URL = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_REPEAT = re.compile(r"([^\d\s])\1{2,}")   # ký tự (không phải số) lặp >= 3 lần
_SPACE = re.compile(r"\s+")


def normalize_text(text):
    """Chuẩn hóa nhẹ: Unicode NFC, link -> 'link', rút gọn ký tự lặp, gộp khoảng trắng.
    Giữ nguyên emoji, dấu câu, chữ hoa vì đó là tín hiệu cảm xúc."""
    text = unicodedata.normalize("NFC", str(text))
    text = _URL.sub(" link ", text)
    text = _REPEAT.sub(r"\1\1", text)          # "đẹppppp" -> "đẹpp"
    return _SPACE.sub(" ", text).strip()


def dedup_key(text):
    """Khóa so khớp trùng lặp: đã chuẩn hóa + chữ thường."""
    return normalize_text(text).lower()


def load_and_clean(verbose=True):
    """Đọc 3 tập + làm sạch. Trả về (train, dev, test, report).

    - Thêm cột 'text' = văn bản đã chuẩn hóa (đưa vào mô hình).
    - CHỈ xóa dòng ở train: ô lỗi '#ERROR!'/rỗng, dòng trùng hoàn toàn,
      dòng trùng văn bản với dev/test (chống rò rỉ dữ liệu).
    - Dev/test giữ nguyên số dòng để so sánh công bằng với paper.
    """
    train, dev, test = read_split("train"), read_split("dev"), read_split("test")
    for df in (train, dev, test):
        df["text"] = df["content"].map(normalize_text)
        df["key"] = df["content"].map(dedup_key)
    n0 = len(train)
    bad = train["key"].isin(["#error!", ""])
    train = train[~bad]
    n1 = len(train)
    train = train.drop_duplicates(subset=["key"] + TARGETS)
    n2 = len(train)
    train = train[~train["key"].isin(set(dev["key"]) | set(test["key"]))].reset_index(drop=True)
    n3 = len(train)
    report = {"train_goc": n0, "bo_dong_loi": n0 - n1, "bo_trung_hoan_toan": n1 - n2,
              "bo_trung_dev_test": n2 - n3, "train_con_lai": n3, "dev": len(dev), "test": len(test)}
    if verbose:
        print(f"Train: {n0} -> {n3} dòng (bỏ {n0 - n1} dòng lỗi, {n1 - n2} dòng trùng, "
              f"{n2 - n3} dòng trùng với dev/test) | dev {len(dev)} | test {len(test)}")
    return train, dev, test, report


# ---------------------------------------------------------------------------
# 4. ĐỘ ĐO ĐÁNH GIÁ
# ---------------------------------------------------------------------------
def _f1(p, r):
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def micro_prf(pred, gold, mode="target_level"):
    """Micro P/R/F1 trên các cặp (bình luận, target) – CHỈ SỐ CHÍNH của đồ án.
    mode='target'       : chỉ cần đoán đúng CÓ nhắc target (mức > 0)
    mode='target_level' : phải đúng cả target lẫn mức (clean/offensive/hate)"""
    pred, gold = np.asarray(pred), np.asarray(gold)
    if mode == "target":
        pred, gold = (pred > 0).astype(int), (gold > 0).astype(int)
    tp = ((pred == gold) & (pred != 0)).sum()
    p = tp / max((pred != 0).sum(), 1)
    r = tp / max((gold != 0).sum(), 1)
    return {"precision": p, "recall": r, "f1": _f1(p, r)}


def paper_prf(pred, gold, mode="target_level"):
    """P/R/F1 theo ĐÚNG code của nhóm tác giả: tính cho từng bình luận rồi lấy trung bình.
    Bình luận không nhắc target nào luôn được 0 điểm -> điểm tối đa ~87,5% trên test.
    Chỉ dùng để SO SÁNH với con số trong paper."""
    pred, gold = np.asarray(pred), np.asarray(gold)
    if mode == "target":
        pred, gold = (pred > 0).astype(int), (gold > 0).astype(int)
    correct = ((pred == gold) & (pred != 0)).sum(1)
    n_pred, n_gold = (pred != 0).sum(1), (gold != 0).sum(1)
    p = np.where(n_pred > 0, correct / np.maximum(n_pred, 1), 0).mean()
    r = np.where(n_gold > 0, correct / np.maximum(n_gold, 1), 0).mean()
    return {"precision": p, "recall": r, "f1": _f1(p, r)}


def summarize(pred, gold, probs=None):
    """Bảng tóm tắt các độ đo (đơn vị %).

    probs (nếu có): xác suất (N, 5, 4) -> tính thêm ROC-AUC, PR-AUC cho bài toán
    'toxic' ở mức bình luận (có ít nhất 1 target ở mức offensive/hate).
    """
    pred, gold = np.asarray(pred), np.asarray(gold)
    s = {
        "F1 T+L micro": micro_prf(pred, gold)["f1"],
        "P T+L micro": micro_prf(pred, gold)["precision"],
        "R T+L micro": micro_prf(pred, gold)["recall"],
        "F1 Target micro": micro_prf(pred, gold, "target")["f1"],
        "F1 T+L (paper)": paper_prf(pred, gold)["f1"],
        "F1 Target (paper)": paper_prf(pred, gold, "target")["f1"],
        "Macro-F1 TB 5 target": np.mean([f1_score(gold[:, j], pred[:, j], labels=[0, 1, 2, 3],
                                                  average="macro", zero_division=0) for j in range(5)]),
    }
    y_tox = (gold >= 2).any(1).astype(int)
    s["Toxic F1"] = f1_score(y_tox, (pred >= 2).any(1).astype(int), zero_division=0)
    if probs is not None:
        score = np.asarray(probs)[:, :, 2:].sum(2).max(1)   # P(offensive)+P(hate) lớn nhất
        s["Toxic ROC-AUC"] = roc_auc_score(y_tox, score)
        s["Toxic PR-AUC"] = average_precision_score(y_tox, score)
    y_hate = (gold == 3).any(1)
    s["Recall hate"] = ((pred == 3).any(1) & y_hate).sum() / max(y_hate.sum(), 1)
    return {k: round(float(v) * 100, 2) for k, v in s.items()}


def per_target_report(pred, gold):
    """Bảng Precision/Recall/F1 theo từng target (macro trên 4 mức) + support."""
    rows = []
    for j, t in enumerate(TARGETS):
        p, r, f, _ = precision_recall_fscore_support(gold[:, j], pred[:, j], labels=[0, 1, 2, 3],
                                                     average="macro", zero_division=0)
        rows.append({"target": t, "macro_P": round(p * 100, 2), "macro_R": round(r * 100, 2),
                     "macro_F1": round(f * 100, 2), "số mẫu có nhắc": int((gold[:, j] > 0).sum())})
    return pd.DataFrame(rows)


def print_classification_reports(pred, gold):
    """In classification_report của scikit-learn cho từng target."""
    for j, t in enumerate(TARGETS):
        print(f"===== {t} =====")
        print(classification_report(gold[:, j], pred[:, j], labels=[0, 1, 2, 3], target_names=LEVELS,
                                    digits=3, zero_division=0))


def plot_confusion_matrices(pred, gold, title=""):
    """Vẽ 5 ma trận nhầm lẫn (hàng = thực tế, cột = dự đoán – quy ước của scikit-learn)."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    axes = axes.ravel()
    for ax, j in zip(axes, range(5)):
        cm = confusion_matrix(gold[:, j], pred[:, j], labels=[0, 1, 2, 3])
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for a in range(4):
            for b in range(4):
                ax.text(b, a, f"{cm[a, b]}\n{norm[a, b]:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if norm[a, b] > 0.6 else "black")
        ax.set_xticks(range(4), LEVELS, rotation=30)
        ax.set_yticks(range(4), LEVELS)
        ax.set_xlabel("Dự đoán")
        ax.set_ylabel("Thực tế")
        ax.set_title(TARGETS[j])
    axes[5].axis("off")
    axes[5].text(0.05, 0.5, "Hàng = nhãn thật\nCột = dự đoán\nĐường chéo = đúng\nSố trên = số bình luận,\nsố dưới = % của hàng",
                 fontsize=11, va="center")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def levels_to_text(row):
    """Một hàng nhãn (5 mức 0..3) -> chuỗi dễ đọc, ví dụ 'individual:hate, groups:offensive'."""
    parts = [f"{TARGETS[j]}:{LEVELS[v]}" for j, v in enumerate(row) if v > 0]
    return ", ".join(parts) if parts else "(không nhắc target)"


# ---------------------------------------------------------------------------
# 5. BASELINE TF-IDF + LOGISTIC REGRESSION (dùng ở Notebook 1 và bản demo local)
# ---------------------------------------------------------------------------
def build_vectorizer():
    """TF-IDF n-gram từ (1-2) + n-gram ký tự (2-5). Chỉ được fit trên train."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                           token_pattern=r"(?u)\b\w+\b")
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True,
                           max_features=200_000)
    return FeatureUnion([("word", word), ("char", char)])


def fit_lr(X, Y, C=1.0, class_weight=None):
    """5 Logistic Regression, mỗi target 1 mô hình 4 lớp."""
    from sklearn.linear_model import LogisticRegression
    return [LogisticRegression(C=C, class_weight=class_weight, max_iter=3000, random_state=SEED).fit(X, Y[:, j])
            for j in range(5)]


def predict_proba_lr(models, X):
    """Xác suất (N, 5, 4), đủ 4 cột kể cả khi train thiếu 1 mức nào đó."""
    out = np.zeros((X.shape[0], 5, 4))
    for j, m in enumerate(models):
        out[:, j, m.classes_] = m.predict_proba(X)
    return out


# ---------------------------------------------------------------------------
# 6. MÔ HÌNH TRANSFORMER MULTI-LABEL (khung run_classification.py)
#    15 nhãn dạng "target#mức"; mô hình cho 15 logit -> giải mã về (N, 5) mức 0..3
# ---------------------------------------------------------------------------
def decode(logits, label2id, thr=0.5):
    """logit (N, 15) -> (nhãn (N, 5) mức 0..3, xác suất (N, 5, 4)).

    Mỗi nhãn đi qua sigmoid. Với từng target: nếu nhãn cao nhất trong 3 mức
    (clean/offensive/hate) vượt ngưỡng thr thì chọn mức đó, ngược lại = 0 (không nhắc).
    thr = 0.5 tương đương logit > 0 như script gốc.
    """
    logits = np.asarray(logits, dtype=float)
    sig = 1 / (1 + np.exp(-logits))
    n = len(logits)
    pred = np.zeros((n, 5), dtype=int)
    probs = np.zeros((n, 5, 4))
    for j, t in enumerate(TARGETS):
        cols = [label2id.get(f"{t}#{LEVELS[lv]}") for lv in (1, 2, 3)]
        s = np.stack([sig[:, c] if c is not None else np.zeros(n) for c in cols], 1)   # (N, 3)
        top = s.max(1)
        pred[:, j] = np.where(top > thr, s.argmax(1) + 1, 0)
        probs[:, j, 1:] = s / np.maximum(s.sum(1, keepdims=True), 1e-9) * top[:, None]
        probs[:, j, 0] = 1 - top
    return pred, probs


def make_segmenter():
    """Tách từ cho PhoBERT bằng underthesea (mô hình tách từ đóng gói sẵn, chạy offline)."""
    from underthesea import word_tokenize
    return lambda s: word_tokenize(s, format="text") if s else s


def load_finetuned(model_dir, device="cpu"):
    """Nạp mô hình đã fine-tune trên Colab (thư mục chứa config.json, model.safetensors,
    tokenizer, run_info.json). Chỉ đọc ổ đĩa: local_files_only=True."""
    import json
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model_dir = Path(model_dir)
    info = json.loads((model_dir / "run_info.json").read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True).to(device).eval()
    return model, tok, info


def predict_transformer(model, tok, info, texts, batch_size=32, device="cpu", segmenter=None):
    """Dự đoán cho danh sách văn bản THÔ, tiền xử lý giống hệt lúc train. Trả về (nhãn, xác suất)."""
    import torch
    xs = [normalize_text(t) for t in texts]
    if info.get("segment"):
        seg = segmenter or make_segmenter()
        xs = [seg(x) for x in xs]
    order = np.argsort([len(x) for x in xs])          # gom câu cùng độ dài -> nhanh hơn
    logits = np.zeros((len(xs), model.config.num_labels))
    with torch.no_grad():
        for i in range(0, len(xs), batch_size):
            idx = order[i:i + batch_size]
            enc = tok([xs[k] for k in idx], truncation=True, max_length=info["max_seq_length"],
                      padding=True, return_tensors="pt").to(device)
            logits[idx] = model(**enc).logits.float().cpu().numpy()
    return decode(logits, model.config.label2id, info.get("threshold", 0.5))
