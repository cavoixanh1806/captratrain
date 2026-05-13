"""
dataset.py
==========
Module định nghĩa class CaptchaDataset để tiền xử lý dữ liệu CAPTCHA
và chuẩn bị đưa vào mô hình TrOCR (VisionEncoderDecoderModel).

Class này hỗ trợ cả Synthetic Data (data/synthetic/) lẫn Real Data (data/)
thông qua cùng một interface — chỉ khác nhau ở đường dẫn đầu vào.
"""

import os
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import TrOCRProcessor


class CaptchaDataset(Dataset):
    """Dataset cho bài toán nhận dạng CAPTCHA với mô hình TrOCR.

    Đọc danh sách ảnh và nhãn từ file metadata.csv, sau đó dùng
    TrOCRProcessor để:
      - Xử lý ảnh: resize về kích thước chuẩn của ViT encoder, normalize.
      - Tokenize text: chuyển chuỗi ký tự thành token IDs cho decoder.

    Args:
        image_dir: Đường dẫn thư mục chứa ảnh CAPTCHA.
        metadata_path: Đường dẫn file metadata.csv (cột: filename, text).
        processor: Instance của TrOCRProcessor đã được load.
        max_target_length: Độ dài tối đa của chuỗi nhãn sau khi tokenize.
    """

    def __init__(
        self,
        image_dir: str | Path,
        metadata_path: str | Path,
        processor: TrOCRProcessor,
        max_target_length: int = 16,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.max_target_length = max_target_length

        # Đọc metadata.csv — mỗi dòng là một cặp (filename, text)
        self.df = pd.read_csv(metadata_path, dtype=str)

        # Đảm bảo đúng cột
        required_cols = {"filename", "text"}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(
                f"metadata.csv phải có đủ các cột: {required_cols}. "
                f"Hiện tại chỉ có: {set(self.df.columns)}"
            )

        # Loại bỏ các dòng có giá trị null hoặc text rỗng
        self.df = self.df.dropna(subset=["filename", "text"]).reset_index(drop=True)
        self.df = self.df[self.df["text"].str.strip().astype(bool)].reset_index(drop=True)

    def __len__(self) -> int:
        """Trả về tổng số mẫu trong dataset."""
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Lấy một mẫu dữ liệu đã được tiền xử lý.

        Pipeline:
            1. Đọc ảnh từ disk, chuyển sang RGB.
            2. Dùng processor để encode ảnh → pixel_values.
            3. Dùng processor để tokenize text → labels (token IDs).
            4. Thay padding token ID bằng -100 để loss function bỏ qua.

        Args:
            idx: Chỉ số của mẫu cần lấy.

        Returns:
            Dict gồm:
                - "pixel_values": Tensor ảnh đã chuẩn hóa, shape (3, H, W).
                - "labels": Tensor token IDs của nhãn, shape (seq_len,).

        Raises:
            FileNotFoundError: Nếu file ảnh không tồn tại.
        """
        row = self.df.iloc[idx]
        filename: str = row["filename"]
        text: str = row["text"]

        # ── Bước 1: Tải ảnh ──────────────────────────────────────────────────
        image_path = self.image_dir / filename
        if not image_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file ảnh: '{image_path}'. "
                f"Kiểm tra lại metadata.csv và thư mục ảnh."
            )

        # TrOCR yêu cầu ảnh RGB (3 kênh màu)
        image = Image.open(image_path).convert("RGB")

        # ── Bước 2: Encode ảnh bằng TrOCRProcessor ───────────────────────────
        # processor xử lý ảnh qua ViTFeatureExtractor:
        #   - Resize về 384x384 (hoặc kích thước chuẩn của model)
        #   - Normalize với mean/std của ImageNet
        pixel_values = self.processor(
            images=image,
            return_tensors="pt",
        ).pixel_values.squeeze(0)  # Bỏ batch dim: (1, 3, H, W) → (3, H, W)

        # ── Bước 3: Tokenize text (labels) ───────────────────────────────────
        # processor.tokenizer chuyển chuỗi ký tự thành token IDs
        # padding="max_length" đảm bảo tất cả labels có cùng độ dài
        # truncation=True cắt bớt nếu text quá dài
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)  # (1, seq_len) → (seq_len,)

        # ── Bước 4: Thay padding token bằng -100 ─────────────────────────────
        # CrossEntropyLoss sẽ bỏ qua các vị trí có label = -100,
        # tránh tính loss trên padding tokens (không có ý nghĩa ngữ nghĩa)
        pad_token_id = self.processor.tokenizer.pad_token_id
        labels[labels == pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }


def create_datasets(
    processor: TrOCRProcessor,
    use_real_data: bool = False,
    real_data_dir: str | Path = "data",
    synthetic_train_dir: str | Path = "data/synthetic/train",
    synthetic_val_dir: str | Path = "data/synthetic/val",
    max_target_length: int = 16,
    val_split_ratio: float = 0.2,
) -> tuple[CaptchaDataset, CaptchaDataset]:
    """Factory function tạo cặp (train_dataset, val_dataset).

    Hỗ trợ 3 chế độ:
        - Synthetic only: dùng data/synthetic/train và data/synthetic/val.
        - Real only: dùng data/ với metadata.csv, tự chia train/val 80/20.
        - Combined: merge cả synthetic và real.

    Args:
        processor: TrOCRProcessor đã được load.
        use_real_data: True để dùng Real Data từ real_data_dir.
        real_data_dir: Thư mục chứa ảnh thực và metadata.csv.
        synthetic_train_dir: Thư mục synthetic train.
        synthetic_val_dir: Thư mục synthetic val.
        max_target_length: Độ dài tối đa của labels.
        val_split_ratio: Tỷ lệ chia val khi dùng real data (mặc định 0.2).

    Returns:
        Tuple (train_dataset, val_dataset).
    """
    from torch.utils.data import ConcatDataset

    if use_real_data:
        real_data_dir = Path(real_data_dir)
        metadata_path = real_data_dir / "metadata.csv"

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy '{metadata_path}'. "
                f"Hãy tạo file metadata.csv với 2 cột: filename, text."
            )

        # Đọc toàn bộ metadata rồi chia train/val
        df_all = pd.read_csv(metadata_path, dtype=str).dropna()
        n_val = int(len(df_all) * val_split_ratio)
        df_val = df_all.sample(n=n_val, random_state=42)
        df_train = df_all.drop(df_val.index)

        # Lưu tạm metadata đã chia để CaptchaDataset đọc
        train_meta = real_data_dir / "_train_meta.csv"
        val_meta = real_data_dir / "_val_meta.csv"
        df_train.to_csv(train_meta, index=False)
        df_val.to_csv(val_meta, index=False)

        train_dataset = CaptchaDataset(real_data_dir, train_meta, processor, max_target_length)
        val_dataset = CaptchaDataset(real_data_dir, val_meta, processor, max_target_length)

    else:
        # Dùng Synthetic Data
        synthetic_train_dir = Path(synthetic_train_dir)
        synthetic_val_dir = Path(synthetic_val_dir)

        train_dataset = CaptchaDataset(
            synthetic_train_dir,
            synthetic_train_dir / "metadata.csv",
            processor,
            max_target_length,
        )
        val_dataset = CaptchaDataset(
            synthetic_val_dir,
            synthetic_val_dir / "metadata.csv",
            processor,
            max_target_length,
        )

    return train_dataset, val_dataset
