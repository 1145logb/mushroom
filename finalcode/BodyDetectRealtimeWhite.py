import cv2
import numpy as np
from ultralytics import YOLO
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 載入模型
model_path = r"C:\Users\sywan\Desktop\code\runs\segment\train9\weights\best.pt"
model = YOLO(model_path)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("無法開啟攝影機")
    exit()

target_height = 1000

total_count = 0
prev_count = 0

# -----------------------------
# 影像處理
# -----------------------------

def mark_white_as_blue(cropped_img):
    B, G, R = cv2.split(cropped_img)

    mask = (R > 200) & (G > 200) & (B > 200)
    mask = mask.astype(np.uint8) * 255

    output = cropped_img.copy()
    output[mask > 0] = [255, 0, 0]  # 藍色

    return output, mask

def mark_body_as_red(cropped_img):
    B, G, R = cv2.split(cropped_img)

    mask = (R > 80) & (G < 120) & (B < 120)
    mask = mask.astype(np.uint8) * 255

    output = cropped_img.copy()
    output[mask > 0] = [0, 0, 255]

    return output, mask

def calculate_white_ratio(white_mask, body_mask):
    white_pixels = np.sum(white_mask > 0)
    body_pixels = np.sum(body_mask > 0)
    if body_pixels == 0:
        return 0.0
    return (white_pixels / body_pixels) * 100


# -----------------------------
# 主迴圈
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.8, verbose=False)

    display_img = frame.copy()
    current_count = 0

    for r in results:
        h_img, w_img = frame.shape[:2]

        if r.masks is not None:
            current_count = len(r.masks.data)

            for i, mask_data in enumerate(r.masks.data):

                # -----------------------------
                # mask處理
                # -----------------------------
                mask = mask_data.cpu().numpy()
                mask = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
                mask = (mask * 255).astype(np.uint8)

                # bounding box
                box = r.boxes.xyxy[i].cpu().numpy().astype(int)
                x1, y1, x2, y2 = box

                # crop
                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                # -----------------------------
                # 白邊分析
                # -----------------------------

                white_img, white_mask = mark_white_as_blue(crop)
                body_img, body_mask = mark_body_as_red(crop)

                ratio = calculate_white_ratio(white_mask, body_mask)

                # -----------------------------
                # 畫在畫面上
                # -----------------------------
                cv2.putText(display_img, f"{ratio:.1f}%", 
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0,255,255), 2)

                # -----------------------------
                # 紅色上色 (原本功能保留)
                # -----------------------------
                red_img = np.zeros_like(frame)
                red_img[:] = (0, 0, 255)

                masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
                masked_red = cv2.bitwise_and(red_img, red_img, mask=mask)

                blended = cv2.addWeighted(masked_frame, 0.4, masked_red, 0.6, 0)

                inv_mask = cv2.bitwise_not(mask)
                display_img = cv2.bitwise_and(display_img, display_img, mask=inv_mask)
                display_img = cv2.add(display_img, blended)

    # -----------------------------
    # 計數
    # -----------------------------
    if current_count > prev_count:
        total_count += (current_count - prev_count)
        if(total_count > 6):
            total_count = 0
    prev_count = current_count


    cv2.putText(display_img, f"Total: {total_count}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0,255,255), 3)

    # resize
    h, w = display_img.shape[:2]
    scale = target_height / h
    resized = cv2.resize(display_img, (int(w*scale), target_height))

    cv2.imshow("Result", resized)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('z'):
        total_count = 0

cap.release()
cv2.destroyAllWindows()