"""
inference.py
============
Module inference — load model TrOCR đã fine-tune và dự đoán text từ ảnh CAPTCHA.

Cách dùng (command line):
    python inference.py path/to/captcha.png

Cách dùng (trong code):
    from inference import CaptchaSolver
    solver = CaptchaSolver()
    text = solver.solve_captcha("path/to/captcha.png")
    print(text)  # "AB3K7"
"""

import sys
import logging
from pathlib import Path

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# ─── Cấu hình logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Hằng số ─────────────────────────────────────────────────────────────────
DEFAULT_MODEL_DIR: str = "./captcha_trocr_model"


class CaptchaSolver:
    """Class giải CAPTCHA sử dụng mô hình TrOCR đã fine-tune.

    Load model và processor một lần khi khởi tạo, sau đó tái sử dụng
    cho nhiều lần inference — tránh overhead load model mỗi lần gọi.

    Args:
        model_dir: Đường dẫn thư mục chứa model và processor đã lưu.
                   Mặc định là './captcha_trocr_model'.

    Raises:
        OSError: Nếu thư mục model không tồn tại hoặc thiếu file cần thiết.
    """

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR) -> None:
        self.model_dir = Path(model_dir)
        self._validate_model_dir()

        # Xác định device: ưu tiên GPU nếu có
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Sử dụng device: {self.device}")

        # Load processor và model
        logger.info(f"Đang load model từ: {self.model_dir}")
        self.processor = TrOCRProcessor.from_pretrained(str(self.model_dir))
        self.model = VisionEncoderDecoderModel.from_pretrained(str(self.model_dir))
        self.model.to(self.device)

        # Chuyển model sang eval mode — tắt dropout, batch norm tracking
        self.model.eval()
        logger.info("✅ Model đã sẵn sàng cho inference.")

    def _validate_model_dir(self) -> None:
        """Kiểm tra thư mục model có tồn tại và đủ file cần thiết.

        Raises:
            OSError: Nếu thư mục không tồn tại hoặc thiếu file model.
        """
        if not self.model_dir.exists():
            raise OSError(
                f"Không tìm thấy thư mục model: '{self.model_dir}'.\n"
                f"Hãy chạy lệnh sau để huấn luyện model trước:\n"
                f"    python train.py --use-real-data"
            )

        # Kiểm tra file config của model (dấu hiệu model đã được lưu đúng)
        config_file = self.model_dir / "config.json"
        if not config_file.exists():
            raise OSError(
                f"Thư mục '{self.model_dir}' không chứa file model hợp lệ.\n"
                f"Hãy chạy lại quá trình huấn luyện:\n"
                f"    python train.py --use-real-data"
            )

    def solve_captcha(self, image_path: str | Path) -> str:
        """Dự đoán text từ ảnh CAPTCHA.

        Pipeline inference:
            1. Kiểm tra file ảnh tồn tại.
            2. Mở ảnh và chuyển sang RGB.
            3. Dùng processor encode ảnh → pixel_values tensor.
            4. Chạy model.generate() với beam search để sinh chuỗi token.
            5. Decode token IDs → chuỗi text.

        Args:
            image_path: Đường dẫn đến file ảnh CAPTCHA.

        Returns:
            Chuỗi ký tự dự đoán (ví dụ: "AB3K7").

        Raises:
            FileNotFoundError: Nếu file ảnh không tồn tại.
        """
        image_path = Path(image_path)

        # ── Bước 1: Kiểm tra file ảnh ─────────────────────────────────────────
        if not image_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file ảnh: '{image_path}'."
            )

        # ── Bước 2: Mở ảnh ────────────────────────────────────────────────────
        image = Image.open(image_path).convert("RGB")

        # ── Bước 3: Encode ảnh ────────────────────────────────────────────────
        pixel_values = self.processor(
            images=image,
            return_tensors="pt",
        ).pixel_values.to(self.device)

        # ── Bước 4: Inference với torch.no_grad() ─────────────────────────────
        # torch.no_grad() tắt gradient computation — tiết kiệm bộ nhớ và
        # tăng tốc độ inference (không cần backprop khi predict)
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                max_length=16,      # Độ dài tối đa của chuỗi sinh ra
                num_beams=4,        # Beam search với 4 beams
                early_stopping=True,
            )

        # ── Bước 5: Decode token IDs → text ──────────────────────────────────
        # skip_special_tokens=True: bỏ qua [CLS], [SEP], [PAD] tokens
        predicted_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]  # Lấy kết quả đầu tiên (batch size = 1)

        return predicted_text.strip()

    def solve_batch(self, image_paths: list[str | Path]) -> list[str]:
        """Dự đoán text cho nhiều ảnh CAPTCHA cùng lúc (batch inference).

        Hiệu quả hơn gọi solve_captcha() nhiều lần vì xử lý song song trên GPU.

        Args:
            image_paths: Danh sách đường dẫn ảnh CAPTCHA.

        Returns:
            Danh sách chuỗi ký tự dự đoán tương ứng.
        """
        images = []
        for path in image_paths:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Không tìm thấy file ảnh: '{path}'.")
            images.append(Image.open(path).convert("RGB"))

        # Encode toàn bộ batch
        pixel_values = self.processor(
            images=images,
            return_tensors="pt",
        ).pixel_values.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                max_length=16,
                num_beams=4,
                early_stopping=True,
            )

        results = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        return [r.strip() for r in results]


def solve_captcha(image_path: str, model_dir: str = DEFAULT_MODEL_DIR) -> str:
    """Hàm tiện ích — giải CAPTCHA từ đường dẫn ảnh (stateless).

    Tạo CaptchaSolver mới mỗi lần gọi. Dùng khi chỉ cần giải 1 ảnh.
    Nếu cần giải nhiều ảnh, hãy dùng class CaptchaSolver trực tiếp.

    Args:
        image_path: Đường dẫn đến file ảnh CAPTCHA.
        model_dir: Thư mục chứa model đã train.

    Returns:
        Chuỗi ký tự dự đoán.
    """
    solver = CaptchaSolver(model_dir=model_dir)
    return solver.solve_captcha(image_path)


if __name__ == "__main__":
    # Chạy từ command line: python inference.py <image_path>
    if len(sys.argv) < 2:
        print("Cách dùng: python inference.py <đường_dẫn_ảnh>")
        print("Ví dụ:     python inference.py data/map_00001.png")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        solver = CaptchaSolver()
        result = solver.solve_captcha(image_path)
        print(f"Kết quả: {result}")
    except FileNotFoundError as e:
        print(f"Lỗi: {e}")
        sys.exit(1)
    except OSError as e:
        print(f"Lỗi model: {e}")
        sys.exit(1)
