"""
synthetic_renderer.py
======================
Renderer chuyên dụng để sinh synthetic CAPTCHA giống real Minecraft Map.

Calibrated từ phân tích 754 ảnh real:
- BGR avg (160, 157, 156), saturation ~10
- 70% pure gray, 24% blue-tinted, 6% other
- 31% flat, 33% mild gradient, 36% complex texture
- 60% bold + 40% regular font, 12 font candidates
- 96% gradient color cho text, char overlap 55% dense

Public API:
    - random_text(length=5) → "ACDEF"
    - render_text_on_image(text, size=128) → (bgr, mask)
    - CAPTCHA_SIZE = 128
    - CHARSET = "ACDEFHJKLMNPQRTUVWXY3479"

File này được dùng bởi:
    - generate_synthetic_crnn.py — sinh data cho CRNN training
    - (và có thể bất kỳ pipeline khác cần synthetic CAPTCHA)
"""

import os
import random
import string
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
# Charset thuc te: 24 ky tu (phan tich tu 500 anh real)
# Loai: O/0, I/1, S/5, B/8, G/6, Z/2 (cac cap de nham)
CHARSET: str = "ACDEFHJKLMNPQRTUVWXY3479"
CAPTCHA_SIZE: int = 128

# Font: mix bold + regular + serif — real data co NHIEU FONT trong 1 anh
# Real CAPTCHA dung: Rockwell-like slab serif, sans-serif (Arial/Verdana),
# serif (Georgia/Times), va mix ngau nhien MOI KY TU.
_FONT_CANDIDATES_BOLD = [
    "arialbd.ttf",          # Arial Bold
    "ariblk.ttf",           # Arial Black
    "verdanab.ttf",         # Verdana Bold
    "tahomabd.ttf",         # Tahoma Bold
    "segoeuib.ttf",         # Segoe UI Bold
    "georgiab.ttf",         # Georgia Bold (Rockwell-like serif)
    "timesbd.ttf",          # Times New Roman Bold (serif)
    "cambriab.ttf",         # Cambria Bold (serif)
    "trebucbd.ttf",         # Trebuchet Bold
    "calibrib.ttf",         # Calibri Bold
    "palab.ttf",            # Palatino Bold (slab serif)
    "courbd.ttf",           # Courier Bold (monospaced serif)
]
_FONT_CANDIDATES_REGULAR = [
    "arial.ttf",            # Arial Regular
    "verdana.ttf",          # Verdana Regular
    "tahoma.ttf",           # Tahoma Regular
    "segoeui.ttf",          # Segoe UI Regular
    "calibri.ttf",          # Calibri Regular
    "trebuc.ttf",           # Trebuchet Regular
    "georgia.ttf",          # Georgia Regular (serif)
    "times.ttf",            # Times New Roman Regular (serif)
    "cour.ttf",             # Courier (monospaced serif)
    "pala.ttf",             # Palatino (slab serif)
    "constan.ttf",          # Constantia (serif)
    "corbel.ttf",           # Corbel
    "Candara.ttf",          # Candara
    "DejaVuSans.ttf",       # Linux fallback
]

# Color palette cho text — weighted distribution tu real data
# Hue trong OpenCV HSV la 0-180 (khong phai 0-360)
# Calibrated hue: real avg=65. Red/Orange 55%, Purple/Mag 17%, Cyan/Blue 18%
_HUE_WEIGHTS = [
    # (hue_range, weight, label)
    ((0, 15), 0.340, "red"),        # Red
    ((15, 30), 0.195, "orange"),    # Orange
    ((165, 180), 0.040, "red2"),    # Red wrap
    ((135, 165), 0.070, "magenta"), # Magenta
    ((120, 135), 0.100, "purple"),  # Purple
    ((30, 45), 0.065, "yellow"),    # Yellow
    ((75, 90), 0.065, "cyan"),      # Cyan
    ((90, 105), 0.065, "lightblue"),# Light blue
    ((105, 120), 0.050, "blue"),    # Blue
    ((50, 75), 0.010, "teal"),      # Teal (rare)
]


def _load_random_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load mot font random — bold or regular."""
    if bold:
        candidates = _FONT_CANDIDATES_BOLD.copy()
    else:
        candidates = _FONT_CANDIDATES_REGULAR.copy()
    random.shuffle(candidates)
    for font_name in candidates:
        try:
            return ImageFont.truetype(font_name, size)
        except (IOError, OSError):
            continue
    for font_name in _FONT_CANDIDATES_BOLD + _FONT_CANDIDATES_REGULAR:
        try:
            return ImageFont.truetype(font_name, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def random_text(length: int = 5) -> str:
    return "".join(random.choices(CHARSET, k=length))


def get_random_real_background(size: int = 128) -> np.ndarray:
    """Sinh BG synthetic giong real CAPTCHA — calibrated tu 754 anh real:

    - BGR avg (160, 157, 156) — gan pure gray
    - Saturation rat thap (avg 10)
    - 70% pure gray, 24% blue-tinted, 6% other
    - Texture: 31% flat, 33% mild, 36% complex (intra-image std~11)

    Args:
        size: kich thuoc (128).

    Returns:
        BGR numpy array (size, size, 3).
    """
    # === Color tint ===
    tint_roll = random.random()
    base_v = random.randint(140, 195)  # value avg 161

    if tint_roll < 0.70:
        # Pure gray (70%): R≈G≈B, sat<15
        bgr_diff = random.randint(-5, 5)
        base_b = base_v + bgr_diff + random.randint(-3, 3)
        base_g = base_v + random.randint(-3, 3)
        base_r = base_v - bgr_diff + random.randint(-3, 3)
    elif tint_roll < 0.94:
        # Blue-tinted (24%): B > R, sat 15-30
        base_b = base_v + random.randint(5, 18)
        base_g = base_v + random.randint(0, 5)
        base_r = base_v - random.randint(5, 15)
    elif tint_roll < 0.98:
        # Green-tinted (4%)
        base_g = base_v + random.randint(5, 15)
        base_b = base_v + random.randint(-5, 5)
        base_r = base_v - random.randint(5, 10)
    else:
        # Neutral/other (2%)
        base_b = base_v + random.randint(-10, 10)
        base_g = base_v + random.randint(-10, 10)
        base_r = base_v + random.randint(-10, 10)

    base = np.array([
        np.clip(base_b, 50, 200),
        np.clip(base_g, 50, 200),
        np.clip(base_r, 50, 200),
    ], dtype=np.float32)

    img = np.full((size, size, 3), base, dtype=np.float32)

    # === Texture level (calibrated: 31% flat, 33% mild, 36% complex) ===
    texture_roll = random.random()

    if texture_roll < 0.31:
        # Flat: noise nhe std ~3
        noise = np.random.normal(0, random.uniform(2, 4), img.shape).astype(np.float32)
        img += noise

    elif texture_roll < 0.64:
        # Mild: gradient + noise std ~7
        gradient_strength = random.uniform(8, 18)
        direction = random.choice(["h", "v", "d"])
        for i in range(size):
            ratio = i / size
            shift = (ratio - 0.5) * gradient_strength
            if direction == "h":
                img[:, i] += shift
            elif direction == "v":
                img[i, :] += shift
            else:
                img[i, :] += shift * (1 - ratio * 0.5)
        noise = np.random.normal(0, random.uniform(5, 8), img.shape).astype(np.float32)
        img += noise

    else:
        # Complex (36% — tang nhieu so voi truoc): patches + noise manh
        num_patches = random.randint(4, 10)
        for _ in range(num_patches):
            patch_size = random.randint(20, 50)
            cx = random.randint(0, size - 1)
            cy = random.randint(0, size - 1)
            patch_color = base + np.array([
                random.randint(-25, 25),
                random.randint(-25, 25),
                random.randint(-25, 25),
            ], dtype=np.float32)
            y1, y2 = max(0, cy - patch_size), min(size, cy + patch_size)
            x1, x2 = max(0, cx - patch_size), min(size, cx + patch_size)
            if y2 > y1 and x2 > x1:
                yy, xx = np.ogrid[y1:y2, x1:x2]
                dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
                alpha = np.clip(1 - dist / patch_size, 0, 1) * 0.4
                for c in range(3):
                    img[y1:y2, x1:x2, c] = (
                        img[y1:y2, x1:x2, c] * (1 - alpha)
                        + patch_color[c] * alpha
                    )
        noise = np.random.normal(0, random.uniform(8, 13), img.shape).astype(np.float32)
        img += noise

    # Augment: flip horizontal
    if random.random() < 0.5:
        img = np.flip(img, axis=1).copy()

    return np.clip(img, 0, 255).astype(np.uint8)


def random_text_color_hsv() -> tuple[int, int, int]:
    """Sinh mau text theo phan phoi giong real CAPTCHA.

    Real: saturation avg 105, range (80, 150).
    Tang saturation range de text ro net hon.

    Returns:
        BGR tuple.
    """
    hue_bins = [w[0] for w in _HUE_WEIGHTS]
    weights = [w[1] for w in _HUE_WEIGHTS]
    chosen_range = random.choices(hue_bins, weights=weights)[0]
    h = random.randint(chosen_range[0], chosen_range[1] - 1)

    # Calibrated to real median: sat=127, val=161
    s = random.randint(100, 180)
    v = random.randint(120, 210)

    hsv_pixel = np.uint8([[[h, s, v]]])
    bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0, 0]
    return (int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2]))


def _random_rotation_angle() -> float:
    """Sinh rotation angle theo phan phoi giong real data.

    Real: 65% trong [-5,5], 21% trong [-15,15], 11% trong [-30,30], 3% den 44.
    """
    roll = random.random()
    if roll < 0.65:
        return random.uniform(-5, 5)
    elif roll < 0.86:
        return random.uniform(-15, 15)
    elif roll < 0.97:
        return random.uniform(-30, 30)
    else:
        return random.uniform(-44, 44)


def _make_gradient_color(
    base_bgr: tuple[int, int, int],
    h: int,
) -> list[tuple[int, int, int]]:
    """Tao gradient color cho 1 ky tu (96% real chars co multi-tone).

    Returns:
        List of RGB colors from top to bottom.
    """
    r, g, b = base_bgr[2], base_bgr[1], base_bgr[0]  # BGR -> RGB
    colors = []
    for row in range(h):
        ratio = row / max(h - 1, 1)
        # Shift hue/brightness nhe — tao hieu ung multi-tone nhe
        shift = int((ratio - 0.5) * random.randint(10, 30))
        nr = max(0, min(255, r + shift))
        ng = max(0, min(255, g - shift // 2))
        nb = max(0, min(255, b + shift // 3))
        colors.append((nr, ng, nb))
    return colors


def _draw_rotated_char(
    char: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    color: tuple[int, int, int],
    angle: float,
    stroke_width: int = 1,
) -> tuple[Image.Image, Image.Image]:
    """Render 1 ky tu voi rotation + gradient color.

    96% real chars co multi-tone → dung gradient fill.
    stroke_width=0-2 (real data phan lon khong co stroke rieng).

    Returns:
        (text_rgba, mask_l).
    """
    bbox = font.getbbox(char)
    pad = max(stroke_width, 1) * 2 + 6
    w = (bbox[2] - bbox[0]) + pad * 2
    h = (bbox[3] - bbox[1]) + pad * 2

    # --- Gradient fill (96% of real chars) ---
    use_gradient = random.random() < 0.96

    if use_gradient:
        gradient_colors = _make_gradient_color(color, h)
        # Render char as mask first
        mask_for_gradient = Image.new("L", (w, h), 0)
        mg_draw = ImageDraw.Draw(mask_for_gradient)
        mg_draw.text(
            (pad - bbox[0], pad - bbox[1]),
            char, font=font, fill=255,
            stroke_width=stroke_width, stroke_fill=255,
        )

        # Apply gradient colors via mask
        text_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        text_np = np.array(text_img)
        mask_np_local = np.array(mask_for_gradient)

        for row in range(h):
            gc = gradient_colors[row]
            text_np[row, :, 0] = gc[0]  # R
            text_np[row, :, 1] = gc[1]  # G
            text_np[row, :, 2] = gc[2]  # B
        text_np[:, :, 3] = mask_np_local  # Alpha from mask
        text_img = Image.fromarray(text_np, "RGBA")
    else:
        text_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_img)
        pil_color = (color[2], color[1], color[0])  # BGR -> RGB
        text_draw.text(
            (pad - bbox[0], pad - bbox[1]),
            char, font=font,
            fill=pil_color + (255,),
            stroke_width=stroke_width,
            stroke_fill=pil_color + (255,),
        )

    # --- Mask (always single color white) ---
    mask_img = Image.new("L", (w, h), 0)
    mask_draw = ImageDraw.Draw(mask_img)
    mask_draw.text(
        (pad - bbox[0], pad - bbox[1]),
        char, font=font, fill=255,
        stroke_width=stroke_width, stroke_fill=255,
    )

    # Rotate per-character
    text_img = text_img.rotate(angle, resample=Image.BICUBIC, expand=True)
    mask_img = mask_img.rotate(angle, resample=Image.BICUBIC, expand=True)

    return text_img, mask_img


def render_text_on_image(
    text: str,
    size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Render CAPTCHA giong 100% real data.

    Args:
        text: 5 ky tu.
        size: 128.

    Returns:
        (noisy_bgr, mask_gray).
    """
    # Background — synthetic giong real (calibrated tu 754 anh)
    bg = get_random_real_background(size)
    noisy_pil = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)).convert("RGBA")
    mask_pil = Image.new("L", (size, size), 0)

    # Base font size — cap 50 de tranh chu qua to tren 128px canvas
    base_font_size = random.randint(36, 50)

    # Char step — calibrated theo font_size de tranh chu thua/cat:
    # Real: 56% images all chars merge (CC=1)
    roll = random.random()
    if roll < 0.55:
        # Dense overlap — chars deeply merged (~50% font_size)
        char_step = int(base_font_size * random.uniform(0.40, 0.55))
    elif roll < 0.80:
        # Medium overlap (~60% font_size)
        char_step = int(base_font_size * random.uniform(0.55, 0.65))
    else:
        # Light — KHÔNG quá thưa, vẫn gần nhau (~70% font_size)
        char_step = int(base_font_size * random.uniform(0.65, 0.75))

    # Text width = from center of first char to center of last char
    text_span = char_step * (len(text) - 1)
    half_char = int(base_font_size * 0.40)  # padding để không cắt chữ
    # Center text với clamp đảm bảo chữ đầu/cuối không bị cắt
    start_x = int((size - text_span) / 2) + random.randint(-3, 3)
    start_x = max(half_char, min(start_x, size - text_span - half_char))
    y_center = size // 2 + random.randint(-4, 4)

    for i, char in enumerate(text):
        color = random_text_color_hsv()
        angle = _random_rotation_angle()

        # === PER-CHAR font selection ===
        # Real: 60% bold/thick, 40% regular — stroke avg=6.9px
        use_bold = random.random() < 0.60
        # Variation nho ±3, cap 50 de khong qua to
        font_size = min(base_font_size + random.randint(-3, 3), 50)
        if use_bold:
            stroke_width = random.choice([1, 2, 2, 3])
        else:
            stroke_width = random.choice([0, 1, 1, 2])
        font = _load_random_font(font_size, bold=use_bold)

        text_img, mask_img = _draw_rotated_char(
            char, font, color, angle, stroke_width=stroke_width
        )

        # Slight blur — real text co canh "lem", khong sac net nhu PIL render
        if random.random() < 0.70:
            blur_r = random.choice([1, 1, 1, 2])
            text_np_tmp = np.array(text_img)
            for ch in range(3):
                text_np_tmp[:,:,ch] = cv2.GaussianBlur(
                    text_np_tmp[:,:,ch], (blur_r*2+1, blur_r*2+1), 0
                )
            text_img = Image.fromarray(text_np_tmp, "RGBA")

        cx = int(start_x + i * char_step)
        paste_x = cx - text_img.width // 2
        # Y-jitter NHO ±5 (giam tu ±10) de chu khong "rot" loan xa
        paste_y = y_center - text_img.height // 2 + random.randint(-5, 5)

        noisy_pil.paste(text_img, (paste_x, paste_y), text_img)

        # Union mask
        char_mask_np = np.array(mask_img)
        x1 = max(0, paste_x)
        y1 = max(0, paste_y)
        x2 = min(size, paste_x + mask_img.width)
        y2 = min(size, paste_y + mask_img.height)
        src_x1 = x1 - paste_x
        src_y1 = y1 - paste_y
        src_x2 = src_x1 + (x2 - x1)
        src_y2 = src_y1 + (y2 - y1)

        if x2 > x1 and y2 > y1:
            existing = np.array(mask_pil)
            new_region = char_mask_np[src_y1:src_y2, src_x1:src_x2]
            existing[y1:y2, x1:x2] = np.maximum(existing[y1:y2, x1:x2], new_region)
            mask_pil = Image.fromarray(existing, mode="L")

    noisy = cv2.cvtColor(np.array(noisy_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    mask_np = np.array(mask_pil)

    # ── Noise lines — chi them MOT IT vi BG real da co san lines ────────────
    noise_overlay = noisy.copy()
    num_lines_roll = random.random()
    if num_lines_roll < 0.40:
        # 40% khong them line moi (vi BG da co)
        num_lines = 0
    elif num_lines_roll < 0.80:
        num_lines = random.randint(1, 3)
    else:
        num_lines = random.randint(3, 6)

    for _ in range(num_lines):
        # 60% curved (Bezier), 40% straight
        use_curve = random.random() < 0.60

        # Color: 73% muted/gray, 27% slightly colored
        if random.random() < 0.73:
            base_c = random.randint(130, 200)
            spread = random.randint(3, 12)
            line_color = (
                int(np.clip(base_c + random.randint(-spread, spread), 0, 255)),
                int(np.clip(base_c + random.randint(-spread, spread), 0, 255)),
                int(np.clip(base_c + random.randint(-spread, spread), 0, 255)),
            )
        else:
            lh = random.randint(0, 179)
            ls = random.randint(30, 90)
            lv = random.randint(130, 200)
            hsv_px = np.uint8([[[lh, ls, lv]]])
            bgr_px = cv2.cvtColor(hsv_px, cv2.COLOR_HSV2BGR)[0, 0]
            line_color = (int(bgr_px[0]), int(bgr_px[1]), int(bgr_px[2]))

        thickness = random.choice([1, 1, 1, 2])

        if use_curve:
            # Bezier curve — constrained to local arcs (real avg ~37px)
            cx_b = random.randint(10, size - 10)
            cy_b = random.randint(10, size - 10)
            spread = random.randint(25, 65)
            p0 = (cx_b - spread // 2 + random.randint(-10, 10),
                  cy_b + random.randint(-spread//3, spread//3))
            p1 = (cx_b + random.randint(-15, 15),
                  cy_b + random.randint(-spread//2, spread//2))
            p2 = (cx_b + spread // 2 + random.randint(-10, 10),
                  cy_b + random.randint(-spread//3, spread//3))
            pts = []
            for t in np.linspace(0, 1, 25):
                bx = int((1-t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0])
                by = int((1-t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1])
                pts.append([bx, by])
            pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(noise_overlay, [pts_arr], False, line_color, thickness, cv2.LINE_AA)
        else:
            # Straight line
            dir_roll = random.random()
            if dir_roll < 0.58:
                angle = random.uniform(-30, 30)
            elif dir_roll < 0.86:
                angle = random.uniform(30, 60) * random.choice([-1, 1])
            else:
                angle = random.uniform(60, 90) * random.choice([-1, 1])

            length = random.randint(20, 55)
            cx = random.randint(5, size - 5)
            cy = random.randint(5, size - 5)
            rad = np.deg2rad(angle)
            x1 = int(cx - length / 2 * np.cos(rad))
            y1 = int(cy - length / 2 * np.sin(rad))
            x2 = int(cx + length / 2 * np.cos(rad))
            y2 = int(cy + length / 2 * np.sin(rad))
            cv2.line(noise_overlay, (x1, y1), (x2, y2), line_color, thickness, cv2.LINE_AA)

    # ── Noise dots — RAT IT (real data hau nhu khong co cham) ────────────────
    num_dots = random.randint(0, 10)
    for _ in range(num_dots):
        x = random.randint(0, size - 1)
        y = random.randint(0, size - 1)
        color = tuple(random.randint(80, 200) for _ in range(3))
        cv2.circle(noise_overlay, (x, y), 1, color, -1)

    # Alpha blend noise overlay — BG da co noise tu real, chi them MOT IT moi
    # Giam alpha vi BG real da co lines/texture, khong can them nhieu
    noise_alpha = random.uniform(0.55, 0.75)
    noisy = cv2.addWeighted(noise_overlay, noise_alpha, noisy, 1.0 - noise_alpha, 0)

    # Wave distortion — 50% probability, moderate amplitude
    if random.random() < 0.50:
        h, w = noisy.shape[:2]
        amp_x = random.uniform(1.0, 3.0)
        freq_x = random.uniform(0.03, 0.08)
        amp_y = random.uniform(0.5, 2.5)
        freq_y = random.uniform(0.03, 0.08)

        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)
        shift_x = np.sin(np.arange(h) * freq_x) * amp_x
        shift_y = np.cos(np.arange(w) * freq_y) * amp_y

        for row in range(h):
            map_x[row, :] = np.arange(w) + shift_y
        for col in range(w):
            map_y[:, col] = np.arange(h) + shift_x

        noisy = cv2.remap(noisy, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask_distorted = cv2.remap(
            mask_np.astype(np.float32), map_x, map_y,
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        )
        mask_np = (mask_distorted > 50).astype(np.uint8) * 255

    return noisy, mask_np


if __name__ == "__main__":
    # Smoke test: render 5 sample CAPTCHAs to /tmp
    import tempfile

    out_dir = Path(tempfile.gettempdir()) / "captra_smoke"
    out_dir.mkdir(exist_ok=True)
    for i in range(5):
        text = random_text()
        bgr, mask = render_text_on_image(text, CAPTCHA_SIZE)
        path = out_dir / f"sample_{i}_{text}.png"
        cv2.imwrite(str(path), bgr)
        print(f"  {path}  text={text}  shape={bgr.shape}")
    print(f"Wrote 5 samples to {out_dir}")
