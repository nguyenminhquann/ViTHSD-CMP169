# ViTHSD – Phát hiện ngôn từ thù ghét theo đối tượng trên bình luận tiếng Việt

Tiểu luận nhóm môn **CMP169 – Trí tuệ nhân tạo** (Khoa CNTT – HUTECH), đề tài số 21.
Giảng viên: TS. Nguyễn Đắc Dzự Trình.

## 1. Bài toán

- **Đầu vào:** một bình luận mạng xã hội tiếng Việt.
- **Đầu ra:** với mỗi đối tượng (*target*) trong 5 đối tượng `individual` (cá nhân), `groups` (nhóm người), `religion/creed` (tôn giáo), `race/ethnicity` (chủng tộc), `politics` (chính trị), mô hình cho biết mức độ:
  `none` (không nhắc tới) · `clean` (nhắc tới, lời lẽ sạch) · `offensive` (xúc phạm) · `hate` (thù ghét).
- **Chỉ số chính:** F1 Target+Level (micro) trên tập test.

Bài báo gốc: *ViTHSD: Exploiting Hatred by Targets for Hate Speech Detection on Vietnamese Social Media Texts*.

## 2. Dữ liệu

- Bộ ViTHSD trên Hugging Face: https://huggingface.co/datasets/sonlam1102/vithsd (giấy phép MIT).
- Train 7.000 · Dev 1.201 · Test 1.800 bình luận. Sau khi làm sạch (bỏ dòng lỗi, bỏ trùng, bỏ bình luận train trùng với dev/test): train còn 6.785.
- **Dữ liệu không đưa lên repo.** Tải bằng `2_TAI_MO_HINH_GOC.bat` (cần tài khoản Hugging Face đã bấm đồng ý điều khoản của bộ dữ liệu).

## 3. Phương pháp

| Mô hình | Vai trò |
|---|---|
| Dummy (lớp đa số, ngẫu nhiên theo phân phối) | Mốc thấp nhất |
| TF-IDF (từ 1–2 gram + ký tự 2–5 gram) + Logistic Regression | Baseline |
| ViSoBERT (`uitnlp/visobert`) fine-tune | Mô hình chính |
| PhoBERT-v2 (`vinai/phobert-base-v2`) fine-tune | Mô hình so sánh |

Cấu hình fine-tune: 15 nhãn nhị phân (5 target × 3 mức), loss BCE, `max_len = 128`, `lr = 3e-5`, `batch = 32`, fp16, early stopping (patience 3), ngưỡng chọn trên tập dev, `seed = 42`. Phần cứng: NVIDIA RTX 4060.

## 4. Kết quả chính (tập test)

| Mô hình | F1 Target+Level micro (%) | F1 theo cách tính của bài báo (%) |
|---|---:|---:|
| Dummy (lớp đa số) | 29,49 | 28,70 |
| TF-IDF + LR | 51,96 | 48,51 |
| PhoBERT-v2 | 55,25 | 49,21 |
| **ViSoBERT** | **55,67** | **51,62** |
| Bài báo gốc – mô hình tốt nhất (XLM-R + Bi-GRU-LSTM-CNN) | – | 51,74 |

Hình và bảng chi tiết nằm trong thư mục `outputs/`.

## 5. Cấu trúc thư mục

```
├── 01_EDA_Baseline.ipynb     khám phá dữ liệu, baseline TF-IDF + LR
├── 02_Transformer.ipynb      fine-tune ViSoBERT / PhoBERT-v2
├── 03_Demo_Offline.ipynb     nạp mô hình đã train, dự đoán offline
├── app.py                    ứng dụng demo (Streamlit)
├── common.py                 hàm dùng chung: đọc, làm sạch dữ liệu, đánh giá, vẽ hình
├── tai_mo_hinh_goc.py        tải dữ liệu + 2 mô hình gốc từ Hugging Face
├── kiem_tra.py               kiểm tra môi trường trước khi chạy
├── 1_CAI_DAT.bat … 5_CHAY_APP.bat   các bước cài đặt và chạy trên Windows
├── requirements.txt          danh sách thư viện (phiên bản cố định)
├── outputs/                  hình và bảng kết quả
├── report/                   báo cáo, slide, biên bản họp, bảng phân công
├── data/                     (không có trên repo) dữ liệu tải về
└── models/                   (không có trên repo) mô hình gốc và mô hình đã train
```

## 6. Cài đặt và chạy (Windows 10/11, Python 3.12)

| Bước | Thao tác | Cần Internet? |
|---|---|---|
| 1 | Nhấp đúp `1_CAI_DAT.bat`: tạo môi trường ảo `.venv`, cài PyTorch (tự chọn bản GPU/CPU) và thư viện | Có |
| 2 | Nhấp đúp `2_TAI_MO_HINH_GOC.bat`: tải dữ liệu và 2 mô hình gốc | Có |
| 3 | Nhấp đúp `3_KIEM_TRA.bat`: kiểm tra mọi thứ đã sẵn sàng | Không |
| 4 | Nhấp đúp `4_CHAY_NOTEBOOK.bat`, chạy lần lượt notebook 01 → 02 → 03 (**Run → Run All Cells**) | Không |
| 5 | Nhấp đúp `5_CHAY_APP.bat`: mở demo tại http://localhost:8501 | Không |

Sau bước 2, toàn bộ dự án chạy **offline**. Notebook 02 cần GPU NVIDIA để train trong thời gian hợp lý (khoảng 10–20 phút mỗi mô hình trên RTX 4060).

## 7. Mã nguồn tham khảo

- Notebook chương 2 sách *Natural Language Processing with Transformers* (Apache-2.0): khung EDA và baseline.
- Ví dụ *Classification of text documents using sparse features* của scikit-learn 1.9.1 (BSD-3): khung TF-IDF và so sánh bộ phân lớp.
- `run_classification.py` của Hugging Face Transformers v5.19.0 (Apache-2.0): khung huấn luyện Transformer.

## 8. Công cụ AI đã sử dụng

Nhóm sử dụng Claude (Anthropic) để được hướng dẫn cài đặt môi trường. Nhóm đã tự chạy toàn bộ code trên máy, kiểm tra kết quả và chịu trách nhiệm về nội dung nộp.

