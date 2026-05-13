"""
fix_rename.py
=============
Script sửa lại tên file đã bị đổi thành nhãn.
- Đổi các file có tên dạng nhãn (VD: 4KTN9.png) về lại format map_XXXXX.png
- Ghi nhãn vào metadata.csv

Cách chạy:
    python fix_rename.py
"""

import os
import csv
from pathlib import Path


def main() -> None:
    data_dir = Path(r"C:\Users\Administrator\Desktop\captratrain\data")

    # Lấy tất cả file .png
    all_png = sorted(data_dir.glob("*.png"))

    # Phân loại: file đã đổi tên (nhãn) vs file giữ nguyên (map_XXXXX)
    renamed_files: list[Path] = []  # File đã bị đổi tên thành nhãn
    map_files: list[Path] = []      # File vẫn giữ format map_XXXXX

    for f in all_png:
        if f.stem.startswith("map_") and f.stem[4:].isdigit():
            map_files.append(f)
        else:
            renamed_files.append(f)

    print(f"Tổng file .png: {len(all_png)}")
    print(f"File giữ nguyên (map_XXXXX): {len(map_files)}")
    print(f"File đã đổi tên (nhãn): {len(renamed_files)}")
    print()

    # Tìm các số thứ tự bị thiếu (0-499)
    existing_indices = set()
    for f in map_files:
        idx = int(f.stem.replace("map_", ""))
        existing_indices.add(idx)

    missing_indices = sorted(set(range(500)) - existing_indices)

    print(f"Số thứ tự bị thiếu: {missing_indices}")
    print()

    if len(renamed_files) != len(missing_indices):
        print(f"⚠️  Số file đổi tên ({len(renamed_files)}) != số thứ tự thiếu ({len(missing_indices)})")
        print("    Sẽ gán theo thứ tự có sẵn.")

    # Sắp xếp file đổi tên theo tên (alphabetical)
    renamed_files.sort(key=lambda f: f.name)

    # Dict lưu mapping: new_name -> label
    label_map: dict[str, str] = {}

    # Đổi tên file nhãn về map_XXXXX.png
    for i, (renamed_file, idx) in enumerate(zip(renamed_files, missing_indices)):
        label = renamed_file.stem  # Tên file chính là nhãn (VD: "4KTN9")
        new_name = f"map_{idx:05d}.png"
        new_path = data_dir / new_name

        print(f"  {renamed_file.name} -> {new_name}  (nhãn: {label})")
        renamed_file.rename(new_path)
        label_map[new_name] = label

    print()
    print(f"✅ Đã đổi tên {len(label_map)} file về format map_XXXXX.png")

    # Tạo metadata.csv với nhãn đã biết
    metadata_path = data_dir / "metadata.csv"

    # Lấy tất cả file map_XXXXX.png hiện tại
    all_map_files = sorted(data_dir.glob("map_*.png"))

    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "text"])
        writer.writeheader()
        for map_file in all_map_files:
            filename = map_file.name
            # Nếu file này có nhãn đã biết, ghi nhãn; nếu không, để trống
            text = label_map.get(filename, "")
            writer.writerow({"filename": filename, "text": text})

    labeled_count = sum(1 for v in label_map.values() if v)
    unlabeled_count = len(all_map_files) - labeled_count

    print(f"✅ Đã tạo '{metadata_path}'")
    print(f"   - Đã có nhãn: {labeled_count} file")
    print(f"   - Chưa có nhãn: {unlabeled_count} file (cần bạn điền thêm)")
    print()
    print("📝 Tiếp theo:")
    print(f"   1. Mở file '{metadata_path}' bằng Excel")
    print("   2. Điền cột 'text' cho các dòng còn trống")
    print("   3. Chạy: python train.py --use-real-data")


if __name__ == "__main__":
    main()
