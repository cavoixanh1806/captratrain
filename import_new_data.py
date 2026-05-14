"""
import_new_data.py
===================
Import anh moi tu thu muc dataset/ vao data/ + cap nhat metadata.csv.

Format file dataset: map_<LABEL>.png (label la 5 ky tu trong ten file)
Format file data: map_<NNNNN>.png + label trong metadata.csv

Usage:
    python import_new_data.py
"""

import re
import shutil
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SOURCE_DIR = Path("dataset")
TARGET_DIR = Path("data")
METADATA_CSV = TARGET_DIR / "metadata.csv"


def main():
    # Doc metadata hien tai
    if METADATA_CSV.exists():
        df = pd.read_csv(METADATA_CSV, dtype=str)
        df = df.dropna(subset=["filename", "text"]).reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=["filename", "text"])

    logger.info(f"Metadata hien tai: {len(df)} anh")

    # Tim so thu tu lon nhat trong filename hien tai (map_NNNNN.png)
    max_num = -1
    for fn in df["filename"]:
        m = re.match(r"map_(\d+)\.png", fn)
        if m:
            max_num = max(max_num, int(m.group(1)))
    next_num = max_num + 1
    logger.info(f"So thu tu tiep theo: {next_num}")

    # Doc anh tu source
    source_files = sorted(SOURCE_DIR.glob("map_*.png"))
    logger.info(f"Tim thay {len(source_files)} anh moi trong {SOURCE_DIR}/")

    if not source_files:
        logger.warning("Khong co anh moi nao de import.")
        return

    # Set cac filename da co (de tranh duplicate)
    existing_filenames = set(df["filename"].tolist())
    existing_labels = set(df["text"].str.upper().tolist())

    new_rows = []
    skipped_duplicate = 0
    skipped_invalid = 0

    for src_path in source_files:
        # Trich xuat label tu ten file: map_LABEL.png
        m = re.match(r"map_([A-Z0-9]+)\.png", src_path.name, re.IGNORECASE)
        if not m:
            logger.warning(f"  Bo qua (ten file khong dung format): {src_path.name}")
            skipped_invalid += 1
            continue

        label = m.group(1).upper()

        # Check do dai label
        if len(label) != 5:
            logger.warning(f"  Bo qua (label khong phai 5 ky tu): {src_path.name} -> '{label}'")
            skipped_invalid += 1
            continue

        # Check duplicate label (anh khac cung label cung OK, neu co thi van add)
        # → khong skip vi co the la anh CAPTCHA khac voi cung text

        # Tao filename moi
        new_filename = f"map_{next_num:05d}.png"

        # Skip neu trung filename (it khi xay ra)
        if new_filename in existing_filenames:
            logger.warning(f"  Trung filename: {new_filename}")
            skipped_duplicate += 1
            continue

        # Copy file
        dst_path = TARGET_DIR / new_filename
        shutil.copy2(src_path, dst_path)

        new_rows.append({"filename": new_filename, "text": label})
        existing_filenames.add(new_filename)
        next_num += 1

    if new_rows:
        # Append vao DataFrame
        new_df = pd.DataFrame(new_rows)
        df_combined = pd.concat([df, new_df], ignore_index=True)
        df_combined.to_csv(METADATA_CSV, index=False)

        logger.info(f"[DONE] Da import {len(new_rows)} anh moi")
        logger.info(f"  Tong so anh: {len(df_combined)}")
        logger.info(f"  Skipped duplicate: {skipped_duplicate}")
        logger.info(f"  Skipped invalid: {skipped_invalid}")
        logger.info(f"  Metadata: {METADATA_CSV}")
    else:
        logger.warning("Khong co anh nao duoc import.")


if __name__ == "__main__":
    main()
