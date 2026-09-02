import cv2
import numpy as np
from ultralytics import YOLO
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -----------------------------
# 模型
# -----------------------------
model_path = r"C:\Users\sywan\Desktop\code\runs\segment\train9\weights\best.pt"
model = YOLO(model_path)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("無法開啟攝影機")
    exit()

target_height = 1000

reset_flag = False

# -----------------------------
# tracking 用
# -----------------------------
prev_centroids = []
total_count = 0


# -----------------------------
# 工具函式
# -----------------------------
def mark_white_as_blue(img):
    B, G, R = cv2.split(img)
    mask = (R > 200) & (G > 200) & (B > 200)
    mask = mask.astype(np.uint8) * 255
    out = img.copy()
    out[mask > 0] = [255, 0, 0]
    return out, mask


def mark_body_as_red(img):
    B, G, R = cv2.split(img)
    mask = (R > 80) & (G < 120) & (B < 120)
    mask = mask.astype(np.uint8) * 255
    out = img.copy()
    out[mask > 0] = [0, 0, 255]
    return out, mask


def calculate_white_ratio(white_mask, body_mask):
    white = np.sum(white_mask > 0)
    body = np.sum(body_mask > 0)
    if body == 0:
        return 0
    return (white / body) * 100


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


def centroid(cnt):
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


# -----------------------------
# 主迴圈
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.8, verbose=False)

    display = frame.copy()
    h, w = frame.shape[:2]

    current_centroids = []

    for r in results:

        if r.masks is None:
            continue

        masks = r.masks.data.cpu().numpy()

        for i, mask_data in enumerate(masks):

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
            # centroid tracking（重點）
            # -----------------------------
            c = centroid(cnt)
            if c is None:
                continue

            current_centroids.append(c)

            # -----------------------------
            # crop（白邊分析）
            # -----------------------------
            x, y, bw, bh = cv2.boundingRect(cnt)
            crop = frame[y:y+bh, x:x+bw]

            if crop.size == 0:
                continue

            _, white_mask = mark_white_as_blue(crop)
            _, body_mask = mark_body_as_red(crop)
            ratio = calculate_white_ratio(white_mask, body_mask)

            # -----------------------------
            # 橢圓
            # -----------------------------
            res = fit_ellipse(contours)
            if res is None:
                continue

            (ellipse, cnt) = res
            (cx, cy), (MA, ma), angle = ellipse
            cx, cy = int(cx), int(cy)

            cv2.ellipse(display, ellipse, (0, 255, 0), 2)
            cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)

            # 長軸
            if MA >= ma:
                major = MA / 2.0
                theta = np.deg2rad(angle)
            else:
                major = ma / 2.0
                theta = np.deg2rad(angle + 90)

            length_px = major * 2
            length_norm = length_px / w
            length = length_norm*23.9

            x1 = int(cx + major * np.cos(theta))
            y1 = int(cy + major * np.sin(theta))
            x2 = int(cx - major * np.cos(theta))
            y2 = int(cy - major * np.sin(theta))

            cv2.line(display, (x1, y1), (x2, y2), (255, 0, 255), 2)

            # -----------------------------
            # 顯示文字
            # -----------------------------
            cv2.putText(display,
                        f"{ratio:.1f}% | L:{length:.3f}",
                        (cx, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2)

    # -----------------------------
    # ⭐ 穩定計數（重點修正）
    # -----------------------------
    new_objects = 0

    for c in current_centroids:
        if all(np.linalg.norm(np.array(c) - np.array(p)) > 30 for p in prev_centroids):
            new_objects += 1

    total_count += new_objects
    prev_centroids = current_centroids

    # -----------------------------
    # ⭐ 到 6 就重置（只觸發一次）
    # -----------------------------
    if total_count >= 7 and not reset_flag:
        total_count = 1
        reset_flag = True

    # 如果已經重置過，但又開始新的流程 → 解鎖
    if total_count == 1:
        reset_flag = False

    cv2.putText(display,
                f"Total: {total_count}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                3)

    # -----------------------------
    # resize
    # -----------------------------
    scale = target_height / h
    show = cv2.resize(display, (int(w * scale), target_height))

    cv2.imshow("Mushroom System", show)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()