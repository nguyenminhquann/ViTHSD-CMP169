# -*- coding: utf-8 -*-
"""
Tải từ Hugging Face – CẦN INTERNET, chỉ chạy 1 lần:
  1) Dữ liệu ViTHSD (sonlam1102/vithsd, dataset gated -> cần đăng nhập) vào data/
  2) Hai mô hình gốc uitnlp/visobert, vinai/phobert-base-v2 (public) vào models/pretrained/
Sau đó mọi notebook chạy hoàn toàn offline.

Chạy:  python tai_mo_hinh_goc.py      (hoặc nhấp đúp 2_TAI_MO_HINH_GOC.bat)
"""
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
DATASET = "sonlam1102/vithsd"
DATA_FILES = ["train.xlsx.csv", "dev.xlsx.csv", "test.xlsx.csv"]
MODELS = {"uitnlp/visobert": "visobert", "vinai/phobert-base-v2": "phobert-base-v2"}

from huggingface_hub import get_token, hf_hub_download, login, snapshot_download  # noqa: E402

# ---------------------------------------------------------------- 1) Dữ liệu (gated)
data_dir = HERE / "data"
data_dir.mkdir(exist_ok=True)
if all((data_dir / f).exists() for f in DATA_FILES):
    print("[đã có] dữ liệu trong", data_dir)
else:
    if get_token() is None:
        print("Dữ liệu ViTHSD trên Hugging Face cần đăng nhập.")
        print("Dán token (loại Read) của tài khoản ĐÃ bấm Agree trên trang https://huggingface.co/datasets/sonlam1102/vithsd")
        login()                                   # hỏi token ngay trong cửa sổ cmd
    try:
        for f in DATA_FILES:
            hf_hub_download(repo_id=DATASET, filename=f, repo_type="dataset", local_dir=str(data_dir))
            print(f"[OK] data/{f}")
    except Exception as e:
        print(f"[LỖI] Không tải được dữ liệu ({type(e).__name__}).")
        print("  -> Kiểm tra: tài khoản đã bấm Agree trên trang dataset chưa, token có đúng tài khoản đó không.")
        print("  -> Đăng nhập lại bằng lệnh:  hf auth login --force")
        sys.exit(1)

# ---------------------------------------------------------------- 2) Hai mô hình gốc (public)
for hf_id, folder in MODELS.items():
    dest = HERE / "models" / "pretrained" / folder
    if (dest / "config.json").exists():
        print(f"[đã có] {dest}")
        continue
    print(f"[tải] {hf_id} -> {dest}  (~400-550 MB, chờ vài phút)")
    snapshot_download(repo_id=hf_id, local_dir=str(dest),
                      ignore_patterns=["*.h5", "*.msgpack", "*.onnx", "flax_model*", "tf_model*"])

# Nạp thử chỉ từ ổ đĩa để chắc chắn sau này dùng offline được
from transformers import AutoModel, AutoTokenizer  # noqa: E402

for folder in MODELS.values():
    path = HERE / "models" / "pretrained" / folder
    tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    model = AutoModel.from_pretrained(str(path), local_files_only=True)
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[OK] {folder}: {n:.0f} triệu tham số, tokenizer {type(tok).__name__}")
print("\nXONG. Từ giờ có thể tắt Internet.")
