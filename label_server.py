"""
label_server.py
===============
Server web đơn giản để gán nhãn CAPTCHA nhanh.

Hiển thị ảnh CAPTCHA, cho phép nhập nhãn, tự động lưu vào metadata.csv.
Hỗ trợ phím tắt Enter để submit và tự động chuyển sang ảnh tiếp theo.

Cách chạy:
    python label_server.py

Sau đó mở trình duyệt: http://localhost:8080
"""

import csv
import json
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import mimetypes

DATA_DIR = Path(r"C:\Users\Administrator\Desktop\captratrain\data")
METADATA_PATH = DATA_DIR / "metadata.csv"
HOST = "0.0.0.0"
PORT = 8080


def read_metadata() -> list[dict[str, str]]:
    """Đọc metadata.csv."""
    rows = []
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def write_metadata(rows: list[dict[str, str]]) -> None:
    """Ghi metadata.csv."""
    with open(METADATA_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "text"])
        writer.writeheader()
        writer.writerows(rows)


HTML_PAGE = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CAPTCHA Labeling Tool</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
        }
        h1 {
            margin-bottom: 10px;
            color: #00d4ff;
        }
        .stats {
            margin-bottom: 20px;
            font-size: 14px;
            color: #aaa;
        }
        .stats span { color: #00d4ff; font-weight: bold; }
        .progress-bar {
            width: 600px;
            height: 8px;
            background: #333;
            border-radius: 4px;
            margin-bottom: 20px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #00d4ff, #00ff88);
            transition: width 0.3s;
        }
        .main-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 20px;
        }
        .image-container {
            background: #fff;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,212,255,0.2);
        }
        .image-container img {
            display: block;
            width: 300px;
            height: auto;
            image-rendering: pixelated;
        }
        .input-group {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        .filename-label {
            font-size: 13px;
            color: #888;
            margin-bottom: 5px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .goto-input {
            background: #333;
            border: 1px solid #555;
            color: #fff;
            padding: 4px 8px;
            width: 70px;
            border-radius: 4px;
            outline: none;
            font-size: 13px;
        }
        .btn-go {
            background: #00d4ff;
            color: #1a1a2e;
            padding: 4px 10px;
            font-size: 12px;
            border-radius: 4px;
            cursor: pointer;
            border: none;
            font-weight: bold;
        }
        .btn-go:hover { background: #00ff88; }
        input[type="text"] {
            font-size: 24px;
            padding: 10px 20px;
            border: 2px solid #00d4ff;
            border-radius: 8px;
            background: #16213e;
            color: #fff;
            text-transform: uppercase;
            letter-spacing: 4px;
            width: 250px;
            text-align: center;
            outline: none;
        }
        input[type="text"]:focus {
            border-color: #00ff88;
            box-shadow: 0 0 10px rgba(0,255,136,0.3);
        }
        button {
            font-size: 16px;
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.2s;
        }
        .btn-save {
            background: #00d4ff;
            color: #1a1a2e;
        }
        .btn-save:hover { background: #00ff88; }
        .btn-skip {
            background: #444;
            color: #ccc;
        }
        .btn-skip:hover { background: #666; }
        .btn-prev {
            background: #555;
            color: #ccc;
        }
        .btn-prev:hover { background: #777; }
        .nav-buttons {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
        .keyboard-hint {
            font-size: 12px;
            color: #666;
            margin-top: 10px;
        }
        .keyboard-hint kbd {
            background: #333;
            padding: 2px 6px;
            border-radius: 3px;
            border: 1px solid #555;
        }
        .toast {
            position: fixed;
            top: 20px;
            right: 20px;
            background: #00ff88;
            color: #1a1a2e;
            padding: 12px 20px;
            border-radius: 8px;
            font-weight: bold;
            opacity: 0;
            transition: opacity 0.3s;
            z-index: 1000;
        }
        .toast.show { opacity: 1; }
        .filter-buttons {
            display: flex;
            gap: 8px;
            margin-bottom: 15px;
        }
        .filter-buttons button {
            font-size: 13px;
            padding: 6px 14px;
        }
        .filter-buttons button.active {
            background: #00d4ff;
            color: #1a1a2e;
        }
    </style>
</head>
<body>
    <h1>🏷️ CAPTCHA Labeling Tool</h1>
    <div class="stats">
        Đã gán nhãn: <span id="labeled-count">0</span> / <span id="total-count">0</span>
        &nbsp;|&nbsp; Còn lại: <span id="remaining-count">0</span>
    </div>
    <div class="progress-bar">
        <div class="progress-fill" id="progress-fill"></div>
    </div>

    <div class="filter-buttons">
        <button id="btn-unlabeled" class="active" onclick="setFilter('unlabeled')">Chưa gán nhãn</button>
        <button id="btn-all" onclick="setFilter('all')">Tất cả</button>
        <button id="btn-labeled" onclick="setFilter('labeled')">Đã gán nhãn</button>
    </div>

    <div class="main-container">
        <div class="filename-label">
            <span id="filename-label">map_00000.png</span>
            <div style="margin-left: 20px; display: flex; gap: 5px; align-items: center;">
                Nhảy tới: <input type="number" id="goto-input" class="goto-input" min="1" placeholder="STT...">
                <button class="btn-go" onclick="gotoImage()">Đi</button>
            </div>
        </div>
        <div class="image-container">
            <img id="captcha-img" src="" alt="CAPTCHA">
        </div>
        <div class="input-group">
            <input type="text" id="label-input" placeholder="Nhập nhãn..." maxlength="5" autofocus autocomplete="off" oninput="this.value = this.value.toUpperCase().replace(/[^A-Z0-9]/g, '')">
            <button class="btn-save" onclick="saveLabel()">Lưu</button>
        </div>
        <div class="nav-buttons">
            <button class="btn-prev" onclick="prevImage()">← Trước</button>
            <button class="btn-skip" onclick="nextImage()">Bỏ qua →</button>
        </div>
        <div class="keyboard-hint">
            <kbd>Enter</kbd> Lưu & tiếp &nbsp;
            <kbd>←</kbd> Ảnh trước &nbsp;
            <kbd>→</kbd> Bỏ qua &nbsp;
            <kbd>Esc</kbd> Xóa input
        </div>
    </div>

    <div class="toast" id="toast">✅ Đã lưu!</div>

    <script>
        let metadata = [];
        let filteredIndices = [];
        let currentFilterIdx = 0;
        let currentFilter = 'unlabeled';

        // Load metadata từ server
        async function loadMetadata() {
            const res = await fetch('/api/metadata');
            metadata = await res.json();
            applyFilter();
            updateStats();
        }

        function applyFilter() {
            filteredIndices = [];
            for (let i = 0; i < metadata.length; i++) {
                if (currentFilter === 'all') {
                    filteredIndices.push(i);
                } else if (currentFilter === 'unlabeled' && !metadata[i].text) {
                    filteredIndices.push(i);
                } else if (currentFilter === 'labeled' && metadata[i].text) {
                    filteredIndices.push(i);
                }
            }
            currentFilterIdx = 0;
            showCurrent();
        }

        function setFilter(filter) {
            currentFilter = filter;
            document.querySelectorAll('.filter-buttons button').forEach(b => b.classList.remove('active'));
            document.getElementById('btn-' + filter).classList.add('active');
            applyFilter();
        }

        function updateStats() {
            const total = metadata.length;
            const labeled = metadata.filter(m => m.text).length;
            document.getElementById('total-count').textContent = total;
            document.getElementById('labeled-count').textContent = labeled;
            document.getElementById('remaining-count').textContent = total - labeled;
            document.getElementById('progress-fill').style.width = (labeled / total * 100) + '%';
        }

        function showCurrent() {
            if (filteredIndices.length === 0) {
                document.getElementById('filename-label').textContent = 'Không có ảnh nào';
                document.getElementById('captcha-img').src = '';
                document.getElementById('label-input').value = '';
                return;
            }
            const idx = filteredIndices[currentFilterIdx];
            const item = metadata[idx];
            document.getElementById('filename-label').textContent =
                `${item.filename} (${currentFilterIdx + 1}/${filteredIndices.length})`;
            document.getElementById('captcha-img').src = '/images/' + item.filename;
            document.getElementById('label-input').value = item.text || '';
            document.getElementById('label-input').focus();
            document.getElementById('label-input').select();
        }

        async function saveLabel() {
            if (filteredIndices.length === 0) return;
            const idx = filteredIndices[currentFilterIdx];
            const input = document.getElementById('label-input');
            const text = input.value.trim().toUpperCase();
            if (text.length !== 5) {
                alert('Nhãn phải nhập đúng 5 ký tự (A-Z, 0-9)!');
                input.focus();
                return;
            }

            // Gửi lên server
            const res = await fetch('/api/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ index: idx, text: text })
            });

            if (res.ok) {
                metadata[idx].text = text;
                showToast('✅ Đã lưu: ' + text);
                updateStats();
                
                if (currentFilter === 'unlabeled') {
                    // Xóa ảnh vừa lưu khỏi danh sách hiển thị hiện tại
                    filteredIndices.splice(currentFilterIdx, 1);
                    if (currentFilterIdx >= filteredIndices.length) {
                        currentFilterIdx = Math.max(0, filteredIndices.length - 1);
                    }
                    showCurrent();
                } else {
                    nextImage();
                }
            }
        }

        function nextImage() {
            if (currentFilterIdx < filteredIndices.length - 1) {
                currentFilterIdx++;
            }
            showCurrent();
        }

        function prevImage() {
            if (currentFilterIdx > 0) {
                currentFilterIdx--;
                showCurrent();
            }
        }

        function gotoImage() {
            const input = document.getElementById('goto-input');
            const val = parseInt(input.value);
            if (!isNaN(val) && val >= 1 && val <= filteredIndices.length) {
                currentFilterIdx = val - 1;
                showCurrent();
                input.value = ''; // Clear sau khi nhảy
            } else {
                alert('Số thứ tự không hợp lệ (Phải từ 1 đến ' + filteredIndices.length + ')');
            }
        }

        function showToast(msg) {
            const toast = document.getElementById('toast');
            toast.textContent = msg;
            toast.classList.add('show');
            setTimeout(() => toast.classList.remove('show'), 1500);
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                saveLabel();
            } else if (e.key === 'ArrowRight' && document.activeElement.id !== 'label-input') {
                nextImage();
            } else if (e.key === 'ArrowLeft' && document.activeElement.id !== 'label-input') {
                prevImage();
            } else if (e.key === 'Escape') {
                document.getElementById('label-input').value = '';
                document.getElementById('label-input').focus();
            }
        });

        // Init
        loadMetadata();
    </script>
</body>
</html>
"""


class LabelHandler(SimpleHTTPRequestHandler):
    """HTTP handler cho labeling tool."""

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self._send_html()
        elif path == "/api/metadata":
            self._send_metadata()
        elif path.startswith("/images/"):
            self._send_image(path)
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/api/save":
            self._save_label()
        else:
            self.send_error(404)

    def _send_html(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode("utf-8"))

    def _send_metadata(self) -> None:
        rows = read_metadata()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(rows, ensure_ascii=False).encode("utf-8"))

    def _send_image(self, path: str) -> None:
        filename = path.replace("/images/", "")
        filepath = DATA_DIR / filename
        if not filepath.exists():
            self.send_error(404)
            return

        mime_type = mimetypes.guess_type(str(filepath))[0] or "image/png"
        self.send_response(200)
        self.send_header("Content-Type", mime_type)
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with open(filepath, "rb") as f:
            self.wfile.write(f.read())

    def _save_label(self) -> None:
        content_length = int(self.headers["Content-Length"])
        body = self.rfile.read(content_length)
        data = json.loads(body)

        index = data["index"]
        text = data["text"].strip().upper()

        # Đọc, cập nhật, ghi lại metadata
        rows = read_metadata()
        if 0 <= index < len(rows):
            rows[index]["text"] = text
            write_metadata(rows)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True}).encode("utf-8"))
        else:
            self.send_error(400, "Invalid index")

    def log_message(self, format, *args) -> None:
        """Giảm noise log — chỉ log API calls."""
        if "/api/" in str(args[0]):
            super().log_message(format, *args)


def main() -> None:
    if not METADATA_PATH.exists():
        print(f"❌ Không tìm thấy {METADATA_PATH}")
        print("   Chạy 'python fix_rename.py' hoặc 'python create_metadata_template.py' trước.")
        return

    server = HTTPServer((HOST, PORT), LabelHandler)
    print(f"🏷️  CAPTCHA Labeling Tool")
    print(f"   Mở trình duyệt: http://{HOST}:{PORT}")
    print(f"   Nhấn Ctrl+C để dừng server")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Đã dừng server.")
        server.shutdown()


if __name__ == "__main__":
    main()
