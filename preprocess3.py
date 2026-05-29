from pathlib import Path
import argparse
import shutil
import cv2
import numpy as np

from utils import extract_digit

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff'}


def to_uint8_image(output: np.ndarray, output_mode: str) -> np.ndarray:
    """
    output from utils.extract_digit is float32 [0,1], normally:
    - background = 0
    - digit = bright/white

    output_mode:
    - mnist: keep white digit on black background
    - black_on_white: invert to black digit on white background, closer to preprocess2 visual output
    """
    out = np.clip(output, 0.0, 1.0)
    if output_mode == 'black_on_white':
        out = 1.0 - out
    return (out * 255).round().astype(np.uint8)


def save_debug_images(debug: dict, debug_dir: Path, output_mode: str):
    debug_dir.mkdir(parents=True, exist_ok=True)
    for name, img in debug.items():
        if name == 'bbox':
            (debug_dir / 'bbox.txt').write_text(str(img), encoding='utf-8')
            continue
        arr = img
        if arr is None:
            continue
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        # canvas trong debug là dạng chữ sáng nền đen; nếu đang xem kiểu preprocess2 thì đảo màu canvas cho dễ so sánh
        if name == 'canvas' and output_mode == 'black_on_white':
            arr = 255 - arr
        cv2.imwrite(str(debug_dir / f'{name}.png'), arr)


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess3: tiền xử lý ảnh hàng loạt bằng utils.extract_digit và lưu ảnh kết quả cuối.'
    )
    parser.add_argument('--input', default='data', help='Thư mục ảnh đầu vào. Mặc định: data')
    parser.add_argument('--output', default='output_preprocess3', help='Thư mục lưu kết quả. Mặc định: output_preprocess3')
    parser.add_argument('--clean-output', action='store_true', help='Xóa thư mục output trước khi chạy')
    parser.add_argument('--img-size', type=int, default=28, help='Kích thước ảnh đầu ra. Mặc định: 28')
    parser.add_argument('--inner-size', type=int, default=24, help='Cạnh dài nhất của chữ trong canvas. Gợi ý: 24 để gần preprocess2 hơn')
    parser.add_argument('--margin', type=int, default=2, help='Margin khi crop chữ số. Mặc định: 2')
    parser.add_argument('--min-component-area', type=int, default=4, help='Diện tích nhỏ nhất của component được giữ')
    parser.add_argument('--keep-ratio', type=float, default=0.01, help='Tỉ lệ diện tích tối thiểu so với component lớn nhất')
    parser.add_argument('--post-threshold', type=int, default=127, help='Ngưỡng nhị phân sau resize. Đặt -1 để giữ ảnh xám mềm')
    parser.add_argument(
        '--output-mode',
        choices=['mnist', 'black_on_white'],
        default='black_on_white',
        help='mnist = chữ trắng nền đen; black_on_white = chữ đen nền trắng giống preprocess2'
    )
    parser.add_argument('--debug-first', type=int, default=0, help='Lưu ảnh debug cho N ảnh đầu tiên')

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        raise FileNotFoundError(f'Không tìm thấy thư mục đầu vào: {input_dir}')

    if args.clean_output and output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in input_dir.rglob('*')
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    if not image_paths:
        print(f'Không tìm thấy ảnh nào trong: {input_dir}')
        return

    post_threshold = None if args.post_threshold < 0 else args.post_threshold

    ok = 0
    fail = 0
    failed_items = []

    debug_root = output_dir / '_debug'

    for idx, input_file in enumerate(image_paths, start=1):
        rel_path = input_file.relative_to(input_dir)
        output_file = output_dir / rel_path
        output_file.parent.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(input_file), cv2.IMREAD_UNCHANGED)
        if img is None:
            fail += 1
            failed_items.append(f'{input_file}\tKhông đọc được ảnh')
            print(f'[LỖI] {input_file}: Không đọc được ảnh')
            continue

        want_debug = idx <= args.debug_first
        result = extract_digit(
            img,
            img_size=args.img_size,
            inner_size=args.inner_size,
            margin=args.margin,
            min_component_area=args.min_component_area,
            keep_ratio=args.keep_ratio,
            post_threshold=post_threshold,
            return_debug=want_debug,
        )

        if want_debug:
            output, debug = result
        else:
            output = result
            debug = None

        if output is None:
            fail += 1
            failed_items.append(f'{input_file}\tKhông tách được chữ số')
            print(f'[LỖI] {input_file}: Không tách được chữ số')
            continue

        save_img = to_uint8_image(output, args.output_mode)
        cv2.imwrite(str(output_file), save_img)

        if want_debug and debug is not None:
            safe_name = str(rel_path).replace('\\', '_').replace('/', '_')
            save_debug_images(debug, debug_root / safe_name, args.output_mode)

        ok += 1
        print(f'[{ok}] {input_file} -> {output_file}')

    if failed_items:
        (output_dir / '_failed.txt').write_text('\n'.join(failed_items), encoding='utf-8')

    print('\nĐã chạy xong preprocess3 bằng utils.py')
    print('Thư mục đầu vào:', input_dir)
    print('Thư mục kết quả:', output_dir)
    print('Tổng số ảnh:', len(image_paths))
    print('Xử lý thành công:', ok)
    print('Bị lỗi:', fail)
    if failed_items:
        print('Danh sách lỗi:', output_dir / '_failed.txt')
    if args.debug_first > 0:
        print('Ảnh debug:', debug_root)


if __name__ == '__main__':
    main()
