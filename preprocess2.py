import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def crop_img(input_file, output_file):
    img = cv2.imread(input_file)
    h, w = img.shape[:2]
    xmin = None
    ymin = None
    xmax = None
    ymax = None
    for y in range(h):
        for x in range(w):
            b, g, r = img[y, x]
            if (b<204 and g<240 and r<240):
                if (xmin==None or xmin>x):
                    xmin = x
                if (xmax==None or xmax<x):
                    xmax = x
                if (ymin==None or ymin>y):
                    ymin = y
                if (ymax==None or ymax<y):
                    ymax = y
    # Crop foreground region
    cropped = img[ymin:ymax, xmin:xmax]
    # Save or display
    cv2.imwrite(output_file, cropped)
def resize_by_height(input_file, output_file):
    img = cv2.imread(input_file)

    target_height = 28

    # Get original size
    h, w = img.shape[:2]

    # Compute scale ratio
    scale = target_height / h

    # Compute new width
    new_width = int(w * scale)

    # Resize image
    resized = cv2.resize(img, (new_width, target_height))

    cv2.imwrite(output_file, resized)

def resize(input_file, output_file):
    img = cv2.imread(input_file)

    target = 28

    # Get original size
    h, w = img.shape[:2]

    if h>=w:
        # Compute scale ratio
        scale = target / h
        new_height = 28
        new_width = int(w * scale)
    else:
        # Compute scale ratio
        scale = target / w
        new_height = int(h * scale)
        new_width = 28

    # Resize image
    resized = cv2.resize(img, (new_width, new_height))

    cv2.imwrite(output_file, resized)

def resize_with_white_bg(input_file, output_file, target_size):
    img = Image.open(input_file)
    img.thumbnail(target_size, Image.Resampling.LANCZOS) # Resize while maintaining ratio
    
    # Create white canvas
    new_img = Image.new("RGB", target_size, (255, 255, 255))
    
    # Paste centered
    new_img.paste(
        img, ((target_size[0] - img.size[0]) // 2, (target_size[1] - img.size[1]) // 2)
    )
    new_img.save(output_file)

def conv2BW(input_file, output_file):
    img = cv2.imread(input_file)

    # Change everything to black or white
    img[np.any(img > [50, 50, 50], axis=-1)] = [255, 255, 255]
    img[np.any(img <= [50, 50, 50], axis=-1)] = [0, 0, 0]

    # Convert to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Save or show the result
    cv2.imwrite(output_file, gray_image)

def preprocess_img(input_folder, crop_output_folder, resize_output_folder, 
resize_with_white_bg_output_folder,  conv2BW_output_folder):
    for i in range(0, 10):
        f1 = input_folder + "/" + str(i)
        for filename in os.listdir(f1):
            input_file = os.path.join(f1, filename)
            crop_output_folder1 = crop_output_folder + "/" + str(i)
            os.makedirs(crop_output_folder1, exist_ok=True)
            crop_output_file = crop_output_folder1 + "/" + filename
            
            resize_output_folder1 = resize_output_folder + "/" + str(i)
            os.makedirs(resize_output_folder1, exist_ok=True)
            resize_output_file = resize_output_folder1 + "/" + filename

            resize_with_white_bg_output_folder1 = resize_with_white_bg_output_folder + "/" + str(i)
            os.makedirs(resize_with_white_bg_output_folder1, exist_ok=True)
            resize_with_white_bg_output_file = resize_with_white_bg_output_folder1 + "/" + filename

            conv2BW_output_folder1 = conv2BW_output_folder + "/" + str(i)
            os.makedirs(conv2BW_output_folder1, exist_ok=True)
            conv2BW_output_file = conv2BW_output_folder1 + "/" + filename
            
            crop_img(input_file, crop_output_file)
            resize(crop_output_file, resize_output_file)
            resize_with_white_bg(resize_output_file, resize_with_white_bg_output_file, (28,28))
            conv2BW(resize_with_white_bg_output_file, conv2BW_output_file)

def loadData(input_folder):
    x=[]
    y=[]
    for i in range(0, 10):
        f1 = input_folder + "/" + str(i)
        j = 0
        for filename in os.listdir(f1):
            input_file = os.path.join(f1, filename)
            img = cv2.imread(input_file,0)
            j = j + 1
            # if (j==3):
            #     exit()
            print(j)
            x.append(img)
            y.append(i)
    x = np.array(x)
    y = np.array(y)
    return x, y


def predict_one_folder(input_folder, model_file, predict_folder):
    x, y = loadData(input_folder)
    x = x / 255.0
    model = keras.models.load_model(model_file)
    # Evaluate model
    test_loss, test_acc = model.evaluate(x, y)
    print("Test accuracy:", test_acc)
    
    # Predict
    predictions = model.predict(x)

    predicted_labels = np.argmax(predictions, axis=1)
    wrong_indices = np.where(predicted_labels != y)[0]
    # exit()

    print("Số ảnh đoán sai:", len(wrong_indices))

    for i in range(len(x)):
        # idx = wrong_indices[i]
        idx = i

        plt.imshow(x[idx], cmap='gray')

        plt.title(
            f"Predicted: {predicted_labels[idx]}, Actual: {y[idx]}"
        )

        figure_file = predict_folder + "/f" + str(i) + "_" + str(y[idx]) + "_" + str(predicted_labels[idx]) + ".png"
        # plt.show()
        # Save the figure (formats include .png, .pdf, .svg, etc.)
        plt.savefig(figure_file)

        # Close the figure to free up memory
        plt.close()
def predict_one_file(input_file, model_file, temp_folder):
    crop_output_file = temp_folder +"/" + "crop.png"
    resize_output_file = temp_folder +"/" + "resize.png"
    resize_with_white_bg_output_file = temp_folder +"/" + "resize_with_white_bg.png"
    conv2BW_output_file = temp_folder +"/" + "conv2BW.png"
    crop_img(input_file, crop_output_file)
    resize(crop_output_file, resize_output_file)
    resize_with_white_bg(resize_output_file, resize_with_white_bg_output_file, (28,28))
    conv2BW(resize_with_white_bg_output_file, conv2BW_output_file)
    model = keras.models.load_model(model_file)
    x=[]
    img = cv2.imread(conv2BW_output_file,0)
    x.append(img)
    x = np.array(x)
    x = x / 255.0
    # Predict
    predictions = model.predict(x)

    # Get digit
    digit = np.argmax(predictions[0])

    print("Predicted digit:", digit)
    return digit


# preprocess_img("D:/DigitReg/data3", "D:/DigitReg/preprocess/crop",
#                "D:/DigitReg/preprocess/resize",
#                "D:/DigitReg/preprocess/resize_with_white_bg",
#                "D:/DigitReg/preprocess/conv2BW")
# predict_one_folder("D:/DigitReg/preprocess/conv2BW", "D:/DigitReg/models/model1.keras", 
        # "D:/DigitReg/Predict")

# preprocess_img("D:\\DigitReg\\accept\\png", "D:/DigitReg/preprocess/crop",
#                "D:/DigitReg/preprocess/resize",
#                "D:/DigitReg/preprocess/resize_with_white_bg",
#                "D:/DigitReg/preprocess/conv2BW")

# predict_one_folder("D:/DigitReg/preprocess/conv2BW","D:/DigitReg/models/model1.keras", 
#         "D:/DigitReg/Predict")

# predict_one_file("D:\\DigitReg\\accept\\png\\3\\491.png", "D:/DigitReg/models/model1.keras", "D:/DigitReg/Temp")


def preprocess_img_final_only(input_folder, output_folder):
    """
    Chạy tiền xử lý preprocess2 và chỉ lưu ảnh kết quả cuối cùng.
    Hàm này tự quét toàn bộ ảnh bên trong input_folder, không bắt buộc data/0...data/9.

    Ví dụ:
        data/data1/png/3/491.png
        -> out_preprocess2/data1/png/3/491.png

        data/data2/01/accept/png/3/abc.png
        -> out_preprocess2/data2/01/accept/png/3/abc.png
    """
    import tempfile
    from pathlib import Path

    image_exts = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    if not input_folder.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục đầu vào: {input_folder}")

    image_paths = [
        p for p in input_folder.rglob("*")
        if p.is_file() and p.suffix.lower() in image_exts
    ]

    if not image_paths:
        print(f"Không tìm thấy ảnh nào trong: {input_folder}")
        return

    count = 0

    for input_file in image_paths:
        rel_path = input_file.relative_to(input_folder)
        output_file = output_folder / rel_path
        output_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)

                crop_output_file = temp_dir / "crop.png"
                resize_output_file = temp_dir / "resize.png"
                resize_with_white_bg_output_file = temp_dir / "resize_with_white_bg.png"

                crop_img(str(input_file), str(crop_output_file))
                resize(str(crop_output_file), str(resize_output_file))
                resize_with_white_bg(
                    str(resize_output_file),
                    str(resize_with_white_bg_output_file),
                    (28, 28)
                )
                conv2BW(
                    str(resize_with_white_bg_output_file),
                    str(output_file)
                )

            count += 1
            print(f"[{count}] {input_file} -> {output_file}")

        except Exception as e:
            print(f"[LỖI] {input_file}: {e}")

    print("\nĐã chạy xong preprocess2.")
    print("Thư mục ảnh đầu vào:", input_folder)
    print("Thư mục kết quả cuối:", output_folder)
    print("Tổng số ảnh xử lý:", count)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Chạy tiền xử lý ảnh hàng loạt bằng preprocess2.py và chỉ lưu kết quả cuối."
    )

    parser.add_argument(
        "--input",
        default="data",
        help="Thư mục ảnh đầu vào. Mặc định: data"
    )

    parser.add_argument(
        "--output",
        default="output_preprocess2",
        help="Thư mục lưu ảnh sau tiền xử lý. Mặc định: output_preprocess2"
    )

    args = parser.parse_args()

    preprocess_img_final_only(
        input_folder=args.input,
        output_folder=args.output
    )