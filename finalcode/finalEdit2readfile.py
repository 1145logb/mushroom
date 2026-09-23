import cv2
import numpy as np
from ultralytics import YOLO
import os
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -----------------------------
# 模型
# -----------------------------
model_path = r"C:\Users\sywan\Desktop\code\runs\segment\train10\weights\best.pt"
model = YOLO(model_path)

# -----------------------------
# 圖片資料夾
# -----------------------------
image_folder = r"C:\Users\sywan\Desktop\code\mushP0517\normal\pink"

# -----------------------------
# 輸出 Excel
# -----------------------------
excel_output = r"C:\Users\sywan\Desktop\code\mushroom_result_norpink.xlsx"

# -----------------------------
# 輸出標記圖片資料夾
# -----------------------------
output_image_folder = r"C:\Users\sywan\Desktop\code\output517\normal\pink"
os.makedirs(output_image_folder, exist_ok=True)

# -----------------------------
# 儲存結果
# -----------------------------
results_list = []

# -----------------------------
# white detection
# -----------------------------
def img_processing(cropped_img):
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
    image_hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    return cropped_img, image_hsv, binary

def mark_white_as_blue(cropped_img, image_hsv):

    # ✅ 修正：真正白色範圍
    lower_white_hsv = np.array([20, 0, 150])
    upper_white_hsv = np.array([160, 255, 255])

    mask = cv2.inRange(image_hsv, lower_white_hsv, upper_white_hsv)

    output_img = cropped_img.copy()
    output_img[mask > 0] = [255, 0, 0]

    return output_img, mask


def mark_body_as_red(cropped_img, image_hsv):
    # 標記靈芝主體 (含白邊)
    lower_body_hsv = np.array([0, 20, 40])
    upper_body_hsv = np.array([40, 255, 255])
    mask = cv2.inRange(image_hsv, lower_body_hsv, upper_body_hsv)

    output_img = cropped_img.copy()
    output_img[mask > 0] = [0, 0, 255]  # 紅色
    return output_img, mask


def calculate_white_ratio(white_mask, body_mask):
    white_pixels = np.sum(white_mask > 0)
    mushroom_pixels = np.sum(body_mask > 0)

    if mushroom_pixels == 0:
        return 0.0

    ratio = (white_pixels / mushroom_pixels) * 100
    return ratio


def fit_ellipse(contours):
    if len(contours) == 0:
        return None

    cnt = max(contours, key=cv2.contourArea)

    if cv2.contourArea(cnt) < 500:
        return None

    if len(cnt) < 5:
        return None

    eps = 0.01 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps, True)

    return cv2.fitEllipse(approx), cnt


def quality_dispatch(ratio, length):
    if 3.5 <= length <= 8.5:
        if 5 <= ratio <= 50:
            return "perfect"
        elif 0.1 <= ratio < 5:
            return "normal"
        else:
            return "bad"

    elif 8.5 < length <= 9.5:
        if 0.1 <= ratio <= 50:
            return "normal"
        else:
            return "bad"

    return "bad"


# -----------------------------
# 讀取圖片
# -----------------------------
image_files = [
    f for f in os.listdir(image_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
]

def resize_keep_ratio(img, target_width):
    h, w = img.shape[:2]
    scale = target_width / w
    new_h = int(h * scale)

    resized = cv2.resize(img, (target_width, new_h))
    return resized

# -----------------------------
# 批量處理
# -----------------------------
for image_name in image_files:

    image_path = os.path.join(image_folder, image_name)
    print(f"Processing: {image_name}")

    frame = cv2.imread(image_path)

    frame = resize_keep_ratio(frame, 800)  # 你要統一的寬度

    if frame is None:
        print(f"無法讀取: {image_name}")
        continue

    display = frame.copy()
    h, w = frame.shape[:2]

    results = model(frame, conf=0.8, verbose=False)

    object_id = 1

    for r in results:

        if r.masks is None:
            continue

        masks = r.masks.data.cpu().numpy()

        for mask_data in masks:

            # -----------------------------
            # mask
            # -----------------------------
            mask = cv2.resize(mask_data, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = (mask * 255).astype(np.uint8)

            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:
                continue

            cnt = max(contours, key=cv2.contourArea)

            # -----------------------------
            # crop
            # -----------------------------
            x, y, bw, bh = cv2.boundingRect(cnt)
            crop = frame[y:y+bh, x:x+bw]

            if crop.size == 0:
                continue

            # =============================
            # 🔥 HSV 白邊版本
            # =============================
            croped, image_hsv, binary = img_processing(crop)

            white_img, white_mask = mark_white_as_blue(croped, image_hsv)

            body_img, mushroom_mask   = mark_body_as_red(croped, image_hsv)

            ratio = calculate_white_ratio(
                white_mask,
                mushroom_mask
            )

            # -----------------------------
            # 橢圓
            # -----------------------------
            res = fit_ellipse(contours)
            if res is None:
                continue

            (ellipse, cnt) = res
            (cx, cy), (MA, ma), angle = ellipse

            cx, cy = int(cx), int(cy)

            # -----------------------------
            # 長度
            # -----------------------------
            if MA >= ma:
                major = MA / 2.0
                theta = np.deg2rad(angle)
            else:
                major = ma / 2.0
                theta = np.deg2rad(angle + 90)

            length_px = major * 2
            length_norm = length_px / w
            length = length_norm * 23.9

            # -----------------------------
            # 品質
            # -----------------------------
            quality = quality_dispatch(ratio, length)

            # -----------------------------
            # 畫圖
            # -----------------------------
            cv2.ellipse(display, ellipse, (0, 255, 0), 2)
            cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)

            x1 = int(cx + major * np.cos(theta))
            y1 = int(cy + major * np.sin(theta))
            x2 = int(cx - major * np.cos(theta))
            y2 = int(cy - major * np.sin(theta))

            cv2.line(display, (x1, y1), (x2, y2), (255, 0, 255), 2)

            cv2.putText(
                display,
                f"{ratio:.1f}% | L:{length:.2f} | {quality}",
                (cx, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            # -----------------------------
            # 存 Excel
            # -----------------------------
            results_list.append({
                "Image": image_name,
                "Object_ID": object_id,
                "White_Ratio(%)": round(ratio, 2),
                "Length(cm)": round(length, 3),
                "Quality": quality
            })

            object_id += 1

    # -----------------------------
    # 儲存標記圖片
    # -----------------------------
    save_path = os.path.join(output_image_folder, image_name)
    cv2.imwrite(save_path, display)

# -----------------------------
# 匯出 Excel
# -----------------------------
df = pd.DataFrame(results_list)
df.to_excel(excel_output, index=False)

print("=================================")
print("全部完成")
print("Excel:", excel_output)
print("圖片:", output_image_folder)
print("=================================")