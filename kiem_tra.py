# -*- coding: utf-8 -*-
"""
Kiểm tra máy local đã sẵn sàng chạy 2 mô hình chưa (chạy offline).

Chạy:  python kiem_tra.py      (hoặc nhấp đúp 3_KIEM_TRA.bat)
"""
import os
import platform
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
problems = []


def ok(msg):
    print("  [OK]  ", msg)


def bad(msg, fix):
    print("  [LỖI] ", msg, "\n         -> Cách sửa:", fix)
    problems.append(msg)


print("\n1) Python")
v = sys.version_info
if v[:2] == (3, 12):
    ok(f"Python {platform.python_version()}")
else:
    bad(f"Đang dùng Python {platform.python_version()}", "cài Python 3.12 rồi chạy lại 1_CAI_DAT.bat")

print("\n2) Thư viện")
for name in ["torch", "transformers", "sentencepiece", "safetensors", "sklearn", "pandas",
             "matplotlib", "underthesea", "streamlit", "notebook"]:
    try:
        mod = __import__(name)
        ok(f"{name:13s} {getattr(mod, '__version__', '')}")
    except Exception as e:
        bad(f"Chưa cài được {name} ({type(e).__name__})", "chạy lại 1_CAI_DAT.bat khi có Internet")

print("\n3) Dữ liệu (tải từ Hugging Face sonlam1102/vithsd)")
for split in ["train", "dev", "test"]:
    if (HERE / "data" / f"{split}.xlsx.csv").exists():
        ok(f"data/{split}.xlsx.csv")
    else:
        bad(f"Thiếu dữ liệu {split}", "khi còn mạng, nhấp đúp 2_TAI_MO_HINH_GOC.bat (cần token Hugging Face)")

print("\n4) Mô hình gốc (cần để train ở Notebook 2)")
for folder in ["visobert", "phobert-base-v2"]:
    if (HERE / "models" / "pretrained" / folder / "config.json").exists():
        ok(f"models/pretrained/{folder}")
    else:
        bad(f"Chưa có models/pretrained/{folder}", "khi còn mạng, nhấp đúp 2_TAI_MO_HINH_GOC.bat")
try:
    import torch
    print("  [GPU] ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "không có GPU – Notebook 2 train sẽ rất chậm")
except Exception:
    pass

print("\n5) Hai mô hình đã train (có sau khi chạy Notebook 2)")
for key in ["visobert", "phobert"]:
    d = HERE / "models" / key
    need = ["config.json", "model.safetensors", "run_info.json"]
    missing = [f for f in need if not (d / f).exists()]
    if missing:
        print(f"  [CHƯA CÓ] models/{key} – bình thường nếu chưa chạy Notebook 2 (hoặc giải nén ViTHSD_ket_qua.zip từ Colab)")
        continue
    try:
        sys.path.insert(0, str(HERE))
        import common
        model, tok, info = common.load_finetuned(d, "cpu")
        labels, _ = common.predict_transformer(model, tok, info, ["bài hát này hay quá"])
        n = sum(p.numel() for p in model.parameters()) / 1e6
        ok(f"models/{key}: nạp được ({n:.0f} triệu tham số), dự đoán thử chạy tốt")
    except Exception as e:
        bad(f"models/{key} không nạp được: {type(e).__name__}: {e}", "giải nén lại ViTHSD_ket_qua.zip")

print("\n" + "=" * 60)
if problems:
    print(f"CÒN {len(problems)} VẤN ĐỀ – sửa theo hướng dẫn ở trên rồi chạy lại.")
else:
    print("SẴN SÀNG! Có thể tắt Wi-Fi và chạy 4_CHAY_NOTEBOOK.bat hoặc 5_CHAY_APP.bat")
print("=" * 60)
