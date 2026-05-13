"""
create_metadata_template.py
============================
Script tiện ích — tạo file metadata.csv mẫu cho thư mục data/.

Tự động quét tất cả file .png trong data/ và tạo metadata.csv
với cột `text` để trống, sẵn sàng cho bạn điền nhãn vào.

Cách chạy:
    python create_metadata_template.py
"""

import csv
from pathlib import Path


def create_template(data_dir: str = "data", output_csv: str = "data/metadata.csv") -> None:
    """Tạo file metadata.csv mẫu từ danh sách ảnh trong thư mục.

    Args:
        data_dir: Thư mục chứa ảnh CAPTCHA.
        output_csv: Đường dẫn file CSV đầu ra.
    """
    data_path = Path(data_dir)
    output_path = Path(output_csv)

    # Lấy danh sách tất cả file .png, sắp xếp theo tên
    png_files = sorted(data_path.glob("*.png"))

    if not png_files:
        print(f"❌ Không tìm thấy file .png nào trong '{data_dir}'")
        return

    # Kiểm tra nếu metadata.csv đã tồn tại
    if output_path.exists():
        confirm = input(
            f"⚠️  File '{output_csv}' đã tồn tại. Ghi đè? (y/N): "
        ).strip().lower()
        if confirm != "y":
            print("Hủy bỏ.")
            return

    # Tạo file CSV với cột text để trống
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "text"])
        writer.writeheader()
        for png_file in png_files:
            writer.writerow({"filename": png_file.name, "text": ""})

    print(f"✅ Đã tạo '{output_csv}' với {len(png_files)} dòng.")
    print()
    print("📝 Hướng dẫn tiếp theo:")
    print(f"   1. Mở file '{output_csv}' bằng Excel hoặc Notepad")
    print("   2. Điền chuỗi ký tự CAPTCHA vào cột 'text' cho từng ảnh")
    print("   3. Lưu file và chạy: python train.py --use-real-data")


if __name__ == "__main__":
    create_template()
