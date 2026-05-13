# Requirements Document

## Introduction

Hệ thống AI Local giải CAPTCHA bằng phương pháp Fine-tuning mô hình TrOCR (`microsoft/trocr-small-printed`) của Microsoft thông qua thư viện Hugging Face. Hệ thống nhận ảnh CAPTCHA (chứa 4–6 ký tự chữ và số bị méo, nghiêng, có đường kẻ nhiễu và chấm nhiễu) và trả về chuỗi ký tự dự đoán. Hệ thống hỗ trợ hai nguồn dữ liệu: data giả sinh tự động và data thực từ thư mục `data/` do người dùng tự dán nhãn.

---

## Glossary

- **System**: Toàn bộ hệ thống AI Local giải CAPTCHA.
- **DataGenerator**: Module `generate_data.py` — tạo ảnh CAPTCHA giả và nhãn tương ứng.
- **CaptchaDataset**: Class trong `dataset.py` — tiền xử lý ảnh và nhãn để đưa vào mô hình.
- **Trainer**: Module `train.py` — huấn luyện (fine-tune) mô hình TrOCR.
- **Solver**: Module `inference.py` — tải mô hình đã huấn luyện và dự đoán text từ ảnh CAPTCHA.
- **TrOCR**: Mô hình nhận dạng ký tự quang học dựa trên Transformer của Microsoft (`microsoft/trocr-small-printed`).
- **TrOCRProcessor**: Bộ xử lý ảnh và tokenizer đi kèm với TrOCR từ thư viện Hugging Face Transformers.
- **VisionEncoderDecoderModel**: Kiến trúc mô hình encoder-decoder dùng cho TrOCR.
- **CER**: Character Error Rate — tỷ lệ lỗi ký tự, metric đánh giá chất lượng nhận dạng.
- **metadata.csv**: File CSV chứa hai cột `filename` và `text`, ánh xạ tên file ảnh sang nhãn ký tự.
- **Synthetic Data**: Dữ liệu CAPTCHA giả được sinh tự động bởi DataGenerator.
- **Real Data**: Dữ liệu CAPTCHA thực từ thư mục `data/` kèm `metadata.csv` do người dùng tạo.
- **Checkpoint**: Trạng thái mô hình được lưu trong quá trình huấn luyện.
- **Fine-tuning**: Kỹ thuật học chuyển giao — tiếp tục huấn luyện mô hình đã được pre-train trên tập dữ liệu mới.

---

## Requirements

### Requirement 1: Tạo Dataset Giả (Synthetic Data Generation)

**User Story:** As a developer, I want to automatically generate synthetic CAPTCHA images with labels, so that I can train the model without needing manually labeled real data.

#### Acceptance Criteria

1. THE DataGenerator SHALL tạo ảnh CAPTCHA chứa chuỗi 4–6 ký tự ngẫu nhiên gồm chữ cái in hoa (A–Z) và chữ số (0–9).
2. THE DataGenerator SHALL áp dụng distortion (méo, nghiêng) và noise (đường kẻ nhiễu, chấm nhiễu) lên mỗi ảnh CAPTCHA được tạo ra.
3. THE DataGenerator SHALL tạo đúng 10,000 ảnh cho tập train và 2,000 ảnh cho tập validation, lưu vào các thư mục riêng biệt (`data/synthetic/train/` và `data/synthetic/val/`).
4. THE DataGenerator SHALL lưu nhãn tương ứng vào file `metadata.csv` trong mỗi thư mục, với hai cột `filename` (tên file ảnh) và `text` (chuỗi ký tự).
5. WHEN DataGenerator hoàn thành việc tạo dữ liệu, THE DataGenerator SHALL in ra số lượng ảnh đã tạo thành công cho mỗi tập (train/val).
6. IF một lỗi xảy ra trong quá trình tạo ảnh, THEN THE DataGenerator SHALL ghi log lỗi kèm tên file bị lỗi và tiếp tục tạo các ảnh còn lại.

---

### Requirement 2: Tiền Xử Lý Dữ Liệu (Dataset Preprocessing)

**User Story:** As a developer, I want a reusable dataset class that handles both synthetic and real CAPTCHA data, so that I can feed data consistently into the training pipeline.

#### Acceptance Criteria

1. THE CaptchaDataset SHALL kế thừa `torch.utils.data.Dataset` và implement đầy đủ các phương thức `__len__` và `__getitem__`.
2. WHEN khởi tạo CaptchaDataset, THE CaptchaDataset SHALL nhận vào đường dẫn thư mục ảnh, đường dẫn file `metadata.csv`, và một instance của `TrOCRProcessor`.
3. THE CaptchaDataset SHALL đọc `metadata.csv` để lấy danh sách cặp (filename, text) và tải ảnh tương ứng từ thư mục được chỉ định.
4. WHEN trả về một phần tử, THE CaptchaDataset SHALL dùng `TrOCRProcessor` để chuyển ảnh thành `pixel_values` và tokenize text thành `labels` theo định dạng yêu cầu của `VisionEncoderDecoderModel`.
5. IF một file ảnh không tồn tại tại đường dẫn được chỉ định trong `metadata.csv`, THEN THE CaptchaDataset SHALL raise `FileNotFoundError` kèm tên file bị thiếu.
6. THE CaptchaDataset SHALL hỗ trợ cả Synthetic Data (từ `data/synthetic/`) lẫn Real Data (từ `data/`) thông qua cùng một interface, chỉ khác nhau ở đường dẫn đầu vào.

---

### Requirement 3: Huấn Luyện Mô Hình (Model Training)

**User Story:** As a developer, I want to fine-tune the TrOCR model on CAPTCHA data, so that the model learns to recognize distorted CAPTCHA characters accurately.

#### Acceptance Criteria

1. THE Trainer SHALL load `VisionEncoderDecoderModel` và `TrOCRProcessor` từ checkpoint `microsoft/trocr-small-printed` của Hugging Face.
2. THE Trainer SHALL cấu hình các tham số decoder cần thiết (`decoder_start_token_id`, `eos_token_id`, `pad_token_id`, `vocab_size`) trên model trước khi bắt đầu huấn luyện.
3. THE Trainer SHALL sử dụng `Seq2SeqTrainingArguments` với các tham số có thể cấu hình: `per_device_train_batch_size`, `learning_rate` (mặc định `5e-5`), `num_train_epochs`, `save_steps`, `eval_strategy`.
4. THE Trainer SHALL tích hợp metric CER (Character Error Rate) từ thư viện `evaluate` để đánh giá mô hình sau mỗi epoch validation.
5. THE Trainer SHALL dùng `Seq2SeqTrainer` để huấn luyện, tự động lưu checkpoint tốt nhất (dựa trên CER thấp nhất trên tập val) vào thư mục `./captcha_trocr_model`.
6. WHEN quá trình huấn luyện kết thúc, THE Trainer SHALL lưu model cuối cùng và processor vào `./captcha_trocr_model` để sẵn sàng cho inference.
7. WHILE huấn luyện đang diễn ra, THE Trainer SHALL in ra loss và CER sau mỗi epoch để người dùng theo dõi tiến trình.
8. IF GPU (CUDA) khả dụng trên máy, THEN THE Trainer SHALL tự động sử dụng GPU; IF không có GPU, THEN THE Trainer SHALL fallback sang CPU và thông báo cho người dùng.

---

### Requirement 4: Dự Đoán CAPTCHA (Inference)

**User Story:** As a developer, I want a simple inference function that loads the trained model and predicts CAPTCHA text from an image, so that I can integrate the solver into other applications.

#### Acceptance Criteria

1. THE Solver SHALL load `VisionEncoderDecoderModel` và `TrOCRProcessor` từ thư mục `./captcha_trocr_model` khi được khởi tạo.
2. THE Solver SHALL cung cấp hàm `solve_captcha(image_path: str) -> str` nhận đường dẫn ảnh CAPTCHA và trả về chuỗi ký tự dự đoán.
3. WHEN `solve_captcha` được gọi, THE Solver SHALL tiền xử lý ảnh bằng `TrOCRProcessor`, chạy inference qua model, và decode output thành chuỗi text.
4. THE Solver SHALL chạy inference ở chế độ `torch.no_grad()` để tối ưu bộ nhớ và tốc độ.
5. IF file ảnh đầu vào không tồn tại, THEN THE Solver SHALL raise `FileNotFoundError` kèm đường dẫn file.
6. IF thư mục `./captcha_trocr_model` không tồn tại hoặc thiếu file model, THEN THE Solver SHALL raise `OSError` với thông báo hướng dẫn người dùng chạy `train.py` trước.
7. THE Solver SHALL hỗ trợ chạy trực tiếp từ command line với cú pháp `python inference.py <image_path>` và in kết quả ra stdout.

---

### Requirement 5: Hỗ Trợ Dữ Liệu Thực (Real Data Support)

**User Story:** As a developer, I want the system to support real CAPTCHA images from the `data/` directory, so that I can train on actual data for better accuracy.

#### Acceptance Criteria

1. THE System SHALL hỗ trợ sử dụng Real Data từ thư mục `data/` khi file `data/metadata.csv` tồn tại với định dạng đúng (hai cột `filename` và `text`).
2. THE Trainer SHALL cho phép người dùng chỉ định nguồn dữ liệu (synthetic hoặc real) thông qua tham số dòng lệnh hoặc biến cấu hình trong script.
3. WHERE Real Data được sử dụng, THE Trainer SHALL tự động chia tập dữ liệu thực thành train/val theo tỷ lệ 80/20 nếu chỉ có một thư mục dữ liệu duy nhất.
4. THE System SHALL cho phép kết hợp Synthetic Data và Real Data trong cùng một lần huấn luyện bằng cách merge hai `CaptchaDataset` thành một dataset duy nhất.

---

### Requirement 6: Cấu Hình Dự Án và Tài Liệu (Project Setup & Documentation)

**User Story:** As a developer, I want a complete project setup with dependencies and documentation, so that I can reproduce the environment and understand how to use the system.

#### Acceptance Criteria

1. THE System SHALL cung cấp file `requirements.txt` liệt kê tất cả các thư viện cần thiết với phiên bản cụ thể: `torch`, `torchvision`, `transformers`, `datasets`, `evaluate`, `opencv-python`, `Pillow`, `captcha`, `pandas`, `scikit-learn`.
2. THE System SHALL cung cấp file `README.md` bằng tiếng Việt hướng dẫn đầy đủ: cài đặt môi trường venv, cài dependencies, các bước chạy từng module theo thứ tự, và ví dụ sử dụng `solve_captcha`.
3. THE System SHALL tuân thủ chuẩn PEP8 trong toàn bộ code Python, bao gồm type hinting cho tất cả các hàm và class, và comment giải thích logic quan trọng.
4. THE System SHALL tổ chức code thành 4 file module độc lập: `generate_data.py`, `dataset.py`, `train.py`, `inference.py` tại thư mục gốc của dự án.
