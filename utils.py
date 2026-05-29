import cv2
import numpy as np

# ============================================================
# CÁC HẰNG SỐ CẤU HÌNH
# ============================================================
# Kích thước ảnh đầu ra cuối cùng đưa vào mô hình.
# Mô hình hiện tại đang học với ảnh 28x28 nên mọi ảnh sau tiền xử lý
# đều phải được đưa về đúng kích thước này.
IMG_SIZE = 28

# Kích thước vùng chữ số bên trong canvas 28x28.
# Nghĩa là chữ số sau khi crop + resize sẽ không chiếm full 28x28,
# mà chỉ chiếm khoảng 22x22 ở giữa. Cách này giúp chừa viền an toàn,
# tránh chữ bị sát mép ảnh và gần với kiểu chuẩn hóa thường thấy như MNIST.
INNER_SIZE = 22

# ============================================================
# HÀM PHỤ: CĂN GIỮA CHỮ SỐ THEO TRỌNG TÂM
# ============================================================
def _shift_to_center_of_mass(img: np.ndarray) -> np.ndarray:
    """
    Dịch ảnh sao cho trọng tâm (center of mass) của nét chữ nằm gần giữa ảnh hơn.

    Ý nghĩa:
    - Sau khi crop và resize, chữ số có thể vẫn hơi lệch trái/phải hoặc lệch trên/dưới.
    - Hàm này tính trọng tâm của toàn bộ nét chữ rồi dịch ảnh về giữa.
    - Việc căn giữa giúp dữ liệu đầu vào ổn định hơn, mô hình dễ học hơn.

    Hàm hỗ trợ cả:
    - ảnh nhị phân cứng (0 và 255)
    - ảnh xám mềm (nét có nhiều mức xám khác nhau)

    Tham số:
    - img: ảnh 2 chiều, foreground sáng hơn background.

    Kết quả:
    - Ảnh đã được dịch về gần tâm ảnh.
    """

    # Đổi sang float32 để tính moment ổn định hơn,
    # nhất là khi ảnh đang ở dạng xám mềm.
    work = img.astype(np.float32)

    # cv2.moments tính các moment không gian của ảnh.
    # Từ đó ta suy ra được tọa độ trọng tâm của vùng sáng.
    M = cv2.moments(work)

    # Nếu m00 = 0 nghĩa là ảnh không có foreground hợp lệ
    # (ví dụ ảnh trống hoàn toàn), khi đó không thể tính trọng tâm.
    if M["m00"] == 0:
        return img

    # Tọa độ trọng tâm theo công thức của image moments.
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    # Tính độ dịch cần thiết để đưa trọng tâm về giữa ảnh.
    shift_x = int(round(img.shape[1] / 2.0 - cx))
    shift_y = int(round(img.shape[0] / 2.0 - cy))

    # Ma trận affine biểu diễn phép tịnh tiến ảnh.
    T = np.float32([
        [1, 0, shift_x],
        [0, 1, shift_y]
    ])

    # warpAffine thực hiện phép dịch ảnh.
    # INTER_LINEAR phù hợp hơn cho ảnh xám mềm vì giữ nét mượt hơn.
    shifted = cv2.warpAffine(
        work,
        T,
        (img.shape[1], img.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0  # phần sinh ra ngoài biên sẽ được tô đen
    )

    return shifted

# ============================================================
# HÀM CHÍNH: TIỀN XỬ LÝ ẢNH CHỮ SỐ
# ============================================================
def extract_digit(
    img,
    img_size: int = IMG_SIZE,
    inner_size: int = INNER_SIZE,
    margin: int = 2,
    min_component_area: int = 4,
    keep_ratio: float = 0.01,
    post_threshold: int | None = None,
    return_debug: bool = False
):
    """
    Tiền xử lý ảnh chữ số viết trực tiếp trên digital canvas
    (nền trắng, chữ đen).

    Mục tiêu của hàm:
    1. Tách phần chữ số ra khỏi nền trắng.
    2. Loại bỏ nhiễu nhỏ.
    3. Giữ lại hình dáng chữ số rõ nhất có thể.
    4. Resize và căn giữa về đúng chuẩn đầu vào mô hình.
    5. Trả ra ảnh float32 có kích thước 28x28, giá trị trong [0, 1].

    Ý tưởng chính của pipeline này:
    - Dùng ảnh nhị phân để tìm vùng chữ số và lọc nhiễu.
    - Nhưng crop từ ảnh xám đã đảo màu để giữ được nét mềm hơn.
    - Mặc định KHÔNG threshold lại sau resize để không làm mất chi tiết nét bút.

    Tham số:
    - img: ảnh đầu vào, có thể là grayscale hoặc BGR.
    - img_size: kích thước canvas đầu ra cuối cùng (mặc định 28).
    - inner_size: cạnh dài nhất của chữ số sau resize trước khi đặt vào canvas.
    - margin: nới thêm viền quanh bounding box để tránh crop sát nét.
    - min_component_area: diện tích tối thiểu tuyệt đối để giữ 1 thành phần liên thông.
    - keep_ratio: diện tích tối thiểu tương đối so với thành phần lớn nhất.
    - post_threshold: nếu khác None thì sẽ threshold lại sau resize.
    - return_debug: nếu True thì trả thêm các ảnh trung gian để debug.

    Kết quả:
    - Nếu return_debug=False: trả về ảnh 28x28 đã chuẩn hóa.
    - Nếu return_debug=True: trả về (output, debug_dict).
    - Nếu không tách được chữ số: trả None hoặc (None, None).
    """

    # --------------------------------------------------------
    # BƯỚC 0: KIỂM TRA ĐẦU VÀO
    # --------------------------------------------------------
    if img is None:
        return (None, None) if return_debug else None

    # --------------------------------------------------------
    # BƯỚC 1: CHUYỂN ẢNH VỀ GRAYSCALE
    # --------------------------------------------------------
    # Nếu ảnh là ảnh màu (3 kênh), chuyển về xám để xử lý đơn giản hơn.
    # Nếu đã là ảnh xám rồi thì chỉ cần copy để tránh sửa trực tiếp dữ liệu gốc.
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # --------------------------------------------------------
    # BƯỚC 2: TẠO MASK NHỊ PHÂN
    # --------------------------------------------------------
    # Ảnh người dùng viết trực tiếp trên canvas có quy ước:
    # - nền trắng
    # - chữ đen
    # Vì vậy ta dùng THRESH_BINARY_INV + OTSU để đảo lại thành:
    # - nền đen
    # - chữ trắng
    # Điều này thuận tiện hơn cho việc tìm foreground.
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Nếu toàn bộ ảnh sau threshold không có pixel trắng nào,
    # coi như không phát hiện được chữ số.
    if cv2.countNonZero(binary) == 0:
        return (None, None) if return_debug else None

    # --------------------------------------------------------
    # BƯỚC 3: LỌC NHIỄU BẰNG CONNECTED COMPONENTS
    # --------------------------------------------------------
    # Thay vì chỉ lấy contour lớn nhất, ở đây dùng connected components
    # để giữ được các phần nét bị đứt nhưng vẫn thuộc cùng chữ số.
    # Ví dụ chữ bị hở nhẹ hoặc sinh ra một vài mảnh rời nhưng vẫn đủ lớn.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    # num_labels <= 1 nghĩa là chỉ có background, không có foreground hợp lệ.
    if num_labels <= 1:
        return (None, None) if return_debug else None

    # stats[1:] bỏ qua background (nhãn 0).
    # Lấy diện tích của tất cả thành phần liên thông foreground.
    component_areas = stats[1:, cv2.CC_STAT_AREA]

    # Thành phần lớn nhất thường là phần chính của chữ số.
    largest_area = int(component_areas.max())

    # Ngưỡng giữ thành phần được tính theo 2 tiêu chí:
    # 1. Không nhỏ hơn min_component_area
    # 2. Hoặc ít nhất bằng keep_ratio * largest_area
    # Mục đích là loại bỏ chấm nhiễu rất nhỏ nhưng vẫn giữ phần nét phụ hợp lý.
    area_threshold = max(min_component_area, int(largest_area * keep_ratio))

    # Tạo mask sạch chỉ chứa các thành phần đủ lớn.
    clean_mask = np.zeros_like(binary)
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= area_threshold:
            clean_mask[labels == label_id] = 255

    # Nếu sau lọc mà không còn gì, trả về None.
    if cv2.countNonZero(clean_mask) == 0:
        return (None, None) if return_debug else None

    # --------------------------------------------------------
    # BƯỚC 4: TÌM BOUNDING BOX BAO TOÀN BỘ CHỮ SỐ
    # --------------------------------------------------------
    # findNonZero lấy tất cả tọa độ pixel trắng trong clean_mask.
    coords = cv2.findNonZero(clean_mask)
    if coords is None:
        return (None, None) if return_debug else None

    # boundingRect tạo hình chữ nhật nhỏ nhất bao toàn bộ chữ số.
    x, y, w, h = cv2.boundingRect(coords)

    # --------------------------------------------------------
    # BƯỚC 5: CROP CÓ CHỪA MARGIN
    # --------------------------------------------------------
    # Margin giúp nét không bị crop quá sát, nhất là ở các đầu nét.
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(clean_mask.shape[1], x + w + margin)
    y2 = min(clean_mask.shape[0], y + h + margin)

    # --------------------------------------------------------
    # BƯỚC 6: TẠO ẢNH "INK" MỀM
    # --------------------------------------------------------
    # Đây là điểm quan trọng của pipeline:
    # - clean_mask chỉ dùng để xác định vùng nào là chữ số
    # - nhưng phần crop thật sự được lấy từ ảnh xám đảo màu để giữ nét mềm
    #
    # Ảnh gray ban đầu:
    # - nền trắng gần 255
    # - nét đen gần 0
    # Sau phép 255 - gray:
    # - nền gần 0
    # - nét trở thành sáng hơn
    ink = 255 - gray

    # Những vùng ngoài clean_mask bị xóa về 0 để chỉ giữ phần liên quan đến chữ số.
    ink[clean_mask == 0] = 0

    # Crop đúng vùng chữ số đã tìm được.
    crop = ink[y1:y2, x1:x2]

    # Nếu crop rỗng hoặc toàn đen thì coi như thất bại.
    if crop.size == 0 or np.max(crop) == 0:
        return (None, None) if return_debug else None

    # --------------------------------------------------------
    # BƯỚC 7: RESIZE GIỮ TỈ LỆ
    # --------------------------------------------------------
    # Không ép trực tiếp về 28x28 vì sẽ làm méo chữ số.
    # Ta chỉ scale sao cho cạnh dài nhất = inner_size, cạnh còn lại tự co theo tỉ lệ.
    crop_h, crop_w = crop.shape
    scale = inner_size / max(crop_h, crop_w)

    new_w = max(1, int(round(crop_w * scale)))
    new_h = max(1, int(round(crop_h * scale)))

    # Nếu thu nhỏ thì dùng INTER_AREA thường cho kết quả tốt hơn.
    # Nếu phóng to thì dùng INTER_LINEAR để giữ nét mượt.
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(crop, (new_w, new_h), interpolation=interp)

    # --------------------------------------------------------
    # BƯỚC 8: TÙY CHỌN THRESHOLD LẠI SAU RESIZE
    # --------------------------------------------------------
    # Mặc định để None để giữ nét mềm.
    # Nếu muốn đầu ra nhị phân cứng hơn thì truyền một ngưỡng, ví dụ 127.
    if post_threshold is not None:
        _, resized = cv2.threshold(resized, post_threshold, 255, cv2.THRESH_BINARY)

    # --------------------------------------------------------
    # BƯỚC 9: ĐẶT CHỮ SỐ VÀO CANVAS CHUẨN 28x28
    # --------------------------------------------------------
    # Tạo canvas đen rỗng.
    canvas = np.zeros((img_size, img_size), dtype=np.float32)

    # Tính vị trí đặt sao cho chữ số nằm giữa canvas theo hình học.
    offset_x = (img_size - new_w) // 2
    offset_y = (img_size - new_h) // 2

    # Chèn chữ số đã resize vào canvas.
    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized.astype(np.float32)

    # --------------------------------------------------------
    # BƯỚC 10: CĂN GIỮA LẠI THEO TRỌNG TÂM
    # --------------------------------------------------------
    # Dù đã đặt giữa theo bounding box, chữ số vẫn có thể lệch trọng tâm.
    # Vì vậy tiếp tục dịch lại bằng center of mass.
    canvas = _shift_to_center_of_mass(canvas)

    # --------------------------------------------------------
    # BƯỚC 11: CHUẨN HÓA GIÁ TRỊ VỀ [0, 1]
    # --------------------------------------------------------
    # Mô hình thường nhận đầu vào float32 trong khoảng [0, 1].
    output = np.clip(canvas, 0, 255).astype(np.float32) / 255.0

    # Nếu không cần debug thì trả kết quả luôn.
    if not return_debug:
        return output

    # --------------------------------------------------------
    # THÔNG TIN DEBUG
    # --------------------------------------------------------
    # Các ảnh trung gian rất hữu ích khi muốn kiểm tra:
    # - threshold có đúng không
    # - lọc nhiễu có làm mất nét không
    # - crop có bị sai vùng không
    # - resize/căn giữa có ổn không
    debug = {
        "gray": gray,                             # ảnh xám ban đầu
        "binary": binary,                         # ảnh threshold đảo màu
        "clean": clean_mask,                      # mask sau lọc connected components
        "ink": ink,                               # ảnh xám đảo màu chỉ giữ vùng chữ số
        "crop": crop,                             # vùng crop của chữ số
        "resized": resized,                       # chữ số sau resize giữ tỉ lệ
        "canvas": (output * 255).astype(np.uint8),# canvas cuối trước khi chia về [0,1]
        "bbox": (x1, y1, x2 - x1, y2 - y1),       # bounding box đã crop (x, y, w, h)
    }
    return output, debug
