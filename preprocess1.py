from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def imread_unicode(path: str | Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    """Đọc ảnh bằng OpenCV, hỗ trợ đường dẫn Unicode trên Windows."""
    path = Path(path)
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise ValueError(f"Không đọc được ảnh: {path}")
    return img


def save_image_unicode(path: str | Path, image: np.ndarray) -> None:
    """Lưu ảnh bằng OpenCV, hỗ trợ đường dẫn Unicode trên Windows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f"Không mã hóa được ảnh: {path}")
    buf.tofile(str(path))


def rgba_to_bgr_on_white(img: np.ndarray) -> np.ndarray:
    """Ghép ảnh RGBA lên nền trắng để tránh vùng alpha làm sai threshold."""
    if img.ndim == 3 and img.shape[2] == 4:
        bgr = img[:, :, :3].astype(np.float32)
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        white = np.full_like(bgr, 255.0)
        out = bgr * alpha + white * (1.0 - alpha)
        return np.clip(out, 0, 255).astype(np.uint8)
    return img


def to_gray(img: np.ndarray) -> np.ndarray:
    img = rgba_to_bgr_on_white(img)
    if img.ndim == 2:
        return img.astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Định dạng ảnh không hỗ trợ: shape={img.shape}")


def force_binary(binary: np.ndarray) -> np.ndarray:
    """Ép ảnh về đúng 0/255, tránh xám mờ sau resize/dịch ảnh."""
    return np.where(binary > 0, 255, 0).astype(np.uint8)


def _border_foreground_ratio(binary: np.ndarray, border: int = 2) -> float:
    h, w = binary.shape[:2]
    border = max(1, min(border, h // 4, w // 4))
    mask = np.zeros_like(binary, dtype=bool)
    mask[:border, :] = True
    mask[-border:, :] = True
    mask[:, :border] = True
    mask[:, -border:] = True
    return float(np.count_nonzero(binary[mask])) / float(np.count_nonzero(mask))


def choose_foreground_binary(gray: np.ndarray) -> np.ndarray:
    """
    Trả về ảnh nhị phân foreground trắng, background đen.

    Khác bản cũ: không chỉ chọn theo tỉ lệ pixel nhỏ nhất. Hàm này chấm điểm cả
    tỉ lệ foreground và mức foreground dính viền. Nhờ vậy tránh chọn nhầm nền làm chữ.
    """
    gray = gray.astype(np.uint8)

    # Nếu ảnh gần như đã nhị phân, Otsu vẫn ổn. Với ảnh xám, Otsu giúp tách nét.
    _, inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, norm = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    candidates = []
    for binary in (inv, norm):
        binary = force_binary(binary)
        fg_ratio = float(np.count_nonzero(binary)) / float(binary.size)
        border_ratio = _border_foreground_ratio(binary, border=2)

        # Chữ số thường chiếm khoảng nhỏ/trung bình, nền không nên phủ kín viền.
        # Không loại quá gắt để vẫn xử lý được ảnh crop sát.
        if fg_ratio <= 0.75:
            score = abs(fg_ratio - 0.18) + 1.8 * border_ratio
        else:
            score = 10.0 + fg_ratio + border_ratio
        candidates.append((score, binary))

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def remove_small_components(
    binary: np.ndarray,
    min_area: int = 4,
    keep_ratio_to_largest: float = 0.015,
) -> np.ndarray:
    """
    Xóa nhiễu nhỏ nhưng không phá các chữ bị rời nét.
    - min_area: diện tích tối thiểu tuyệt đối.
    - keep_ratio_to_largest: giữ component nếu đủ lớn so với component lớn nhất.
    """
    binary = force_binary(binary)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), 8)
    if num_labels <= 1:
        return binary

    areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
    largest = max(areas) if areas else 0
    if largest <= 0:
        return binary

    cleaned = np.zeros_like(binary)
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area >= min_area or area >= largest * keep_ratio_to_largest:
            cleaned[labels == label_id] = 255

    return cleaned if np.count_nonzero(cleaned) > 0 else binary


def crop_foreground(binary: np.ndarray, padding: int = 1) -> Optional[np.ndarray]:
    """Crop theo bounding box của foreground, có padding nhẹ."""
    ys, xs = np.where(binary > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    h, w = binary.shape[:2]
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w - 1, x2 + padding)
    y2 = min(h - 1, y2 + padding)

    return binary[y1 : y2 + 1, x1 : x2 + 1]


def resize_keep_ratio_nearest(crop: np.ndarray, max_digit_size: int = 24) -> np.ndarray:
    """
    Resize chữ số sao cho cạnh lớn nhất = max_digit_size, giữ tỉ lệ.
    Dùng INTER_NEAREST để không tạo pixel xám.
    """
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    max_digit_size = int(max(1, max_digit_size))
    scale = max_digit_size / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return force_binary(resized)


def paste_center_by_bbox(resized: np.ndarray, image_size: int = 28) -> np.ndarray:
    """
    Đưa ảnh đã resize vào chính giữa canvas theo bounding box.
    Không dùng moment/trọng tâm để tránh lỗi số 7 bị tụt xuống đáy.
    """
    image_size = int(image_size)
    canvas = np.zeros((image_size, image_size), dtype=np.uint8)
    h, w = resized.shape[:2]

    # Nếu vì lý do nào đó resized lớn hơn canvas, crop lại an toàn.
    if h > image_size or w > image_size:
        scale = min(image_size / float(h), image_size / float(w))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(resized, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        resized = force_binary(resized)
        h, w = resized.shape[:2]

    y = (image_size - h) // 2
    x = (image_size - w) // 2
    canvas[y : y + h, x : x + w] = resized
    return force_binary(canvas)


def bbox_margin(binary: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(binary > 0)
    if len(xs) == 0 or len(ys) == 0:
        h, w = binary.shape[:2]
        return w, h, w, h
    h, w = binary.shape[:2]
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return x1, y1, w - 1 - x2, h - 1 - y2


def recenter_canvas(binary: np.ndarray, image_size: int = 28) -> np.ndarray:
    """Căn lại canvas theo bounding box sau các bước làm dày."""
    crop = crop_foreground(binary, padding=0)
    if crop is None:
        return np.zeros((image_size, image_size), dtype=np.uint8)
    return paste_center_by_bbox(crop, image_size=image_size)


def thicken_if_needed(
    canvas: np.ndarray,
    mode: str = "auto",
    image_size: int = 28,
    min_fg_ratio: float = 0.055,
) -> np.ndarray:
    """
    Làm dày nét nhẹ.
    - none: không làm dày.
    - auto: chỉ làm dày khi nét quá mảnh.
    - always: luôn làm dày 1 lần.
    """
    mode = mode.lower().strip()
    if mode == "none":
        return force_binary(canvas)

    fg_ratio = float(np.count_nonzero(canvas)) / float(canvas.size)
    should_thicken = mode == "always" or (mode == "auto" and fg_ratio < min_fg_ratio)
    if not should_thicken:
        return force_binary(canvas)

    kernel = np.array([[1, 1], [1, 1]], dtype=np.uint8)
    thick = cv2.dilate((canvas > 0).astype(np.uint8), kernel, iterations=1) * 255
    thick = recenter_canvas(thick.astype(np.uint8), image_size=image_size)
    return force_binary(thick)


def extract_digit(
    image_path: str | Path,
    image_size: int = 28,
    max_digit_size: int = 24,
    padding: int = 1,
    min_component_area: int = 4,
    thicken_mode: str = "auto",
    output_mode: str = "mnist",
    return_uint8: bool = True,
) -> np.ndarray:
    """
    Tiền xử lý một ảnh về 28x28.

    output_mode:
    - mnist: foreground trắng, background đen. Phù hợp train MNIST và script Word đang đảo màu cột trái.
    - black_on_white: foreground đen, background trắng. Phù hợp xem trực tiếp.
    """
    img = imread_unicode(image_path, cv2.IMREAD_UNCHANGED)
    gray = to_gray(img)

    binary = choose_foreground_binary(gray)
    binary = remove_small_components(binary, min_area=min_component_area)
    crop = crop_foreground(binary, padding=padding)

    if crop is None:
        canvas = np.zeros((image_size, image_size), dtype=np.uint8)
    else:
        resized = resize_keep_ratio_nearest(crop, max_digit_size=max_digit_size)
        canvas = paste_center_by_bbox(resized, image_size=image_size)
        canvas = thicken_if_needed(canvas, mode=thicken_mode, image_size=image_size)
        canvas = recenter_canvas(canvas, image_size=image_size)

    canvas = force_binary(canvas)

    if output_mode == "black_on_white":
        canvas = 255 - canvas
    elif output_mode != "mnist":
        raise ValueError("output_mode phải là 'mnist' hoặc 'black_on_white'")

    if return_uint8:
        return canvas.astype(np.uint8)
    return (canvas.astype(np.float32) / 255.0)[..., np.newaxis]


def preprocess_from_config(image_path: str | Path, cfg: dict, return_uint8: bool = False) -> np.ndarray:
    pp = cfg.get("preprocess", {}) if isinstance(cfg, dict) else {}
    return extract_digit(
        image_path=image_path,
        image_size=int(pp.get("image_size", 28)),
        max_digit_size=int(pp.get("max_digit_size", pp.get("core_size", 24))),
        padding=int(pp.get("padding", 1)),
        min_component_area=int(pp.get("min_component_area", 4)),
        thicken_mode=str(pp.get("thicken_mode", "auto")),
        output_mode=str(pp.get("output_mode", "mnist")),
        return_uint8=return_uint8,
    )


def process_folder(
    input_folder: str | Path,
    output_folder: str | Path,
    image_size: int = 28,
    max_digit_size: int = 24,
    padding: int = 1,
    min_component_area: int = 4,
    thicken_mode: str = "auto",
    output_mode: str = "mnist",
    clean_output: bool = False,
) -> None:
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    if not input_folder.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục đầu vào: {input_folder}")

    if clean_output and output_folder.exists():
        shutil.rmtree(output_folder)

    image_paths = sorted(
        p for p in input_folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    if not image_paths:
        print(f"Không tìm thấy ảnh trong thư mục: {input_folder}")
        return

    ok_count = 0
    error_count = 0

    for idx, img_path in enumerate(image_paths, start=1):
        try:
            rel_path = img_path.relative_to(input_folder)
            out_path = output_folder / rel_path
            processed = extract_digit(
                image_path=img_path,
                image_size=image_size,
                max_digit_size=max_digit_size,
                padding=padding,
                min_component_area=min_component_area,
                thicken_mode=thicken_mode,
                output_mode=output_mode,
                return_uint8=True,
            )
            save_image_unicode(out_path, processed)
            ok_count += 1
            print(f"[OK {idx}/{len(image_paths)}] {img_path} -> {out_path}")
        except Exception as exc:
            error_count += 1
            print(f"[ERROR {idx}/{len(image_paths)}] {img_path}: {exc}")

    print("\nHoàn thành tiền xử lý.")
    print("Thư mục đầu vào:", input_folder)
    print("Thư mục đầu ra:", output_folder)
    print("Số ảnh xử lý thành công:", ok_count)
    print("Số ảnh lỗi:", error_count)
    print("Cấu hình:")
    print(f"  image_size={image_size}")
    print(f"  max_digit_size={max_digit_size}")
    print(f"  padding={padding}")
    print(f"  min_component_area={min_component_area}")
    print(f"  thicken_mode={thicken_mode}")
    print(f"  output_mode={output_mode}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess ảnh chữ số 28x28 - bản sửa lỗi căn tâm/bị mờ.")
    parser.add_argument("--input", default="data", help="Thư mục chứa ảnh đầu vào. Mặc định: data")
    parser.add_argument("--output", default="output_preprocess1", help="Thư mục lưu ảnh đầu ra. Mặc định: output_preprocess1")
    parser.add_argument("--image-size", type=int, default=28, help="Kích thước ảnh đầu ra. Mặc định: 28")
    parser.add_argument("--max-digit-size", type=int, default=24, help="Cạnh lớn nhất của chữ số sau resize. Mặc định: 24 để chừa lề 2px")
    parser.add_argument("--padding", type=int, default=1, help="Padding khi crop foreground. Mặc định: 1")
    parser.add_argument("--min-component-area", type=int, default=4, help="Diện tích component nhỏ nhất được giữ. Mặc định: 4")
    parser.add_argument("--thicken-mode", choices=["none", "auto", "always"], default="auto", help="Làm dày nét: none/auto/always. Mặc định: auto")
    parser.add_argument("--output-mode", choices=["mnist", "black_on_white"], default="mnist", help="mnist = chữ trắng nền đen; black_on_white = chữ đen nền trắng")
    parser.add_argument("--clean-output", action="store_true", help="Xóa thư mục output trước khi chạy để tránh ảnh cũ sót lại")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    process_folder(
        input_folder=args.input,
        output_folder=args.output,
        image_size=args.image_size,
        max_digit_size=args.max_digit_size,
        padding=args.padding,
        min_component_area=args.min_component_area,
        thicken_mode=args.thicken_mode,
        output_mode=args.output_mode,
        clean_output=args.clean_output,
    )
