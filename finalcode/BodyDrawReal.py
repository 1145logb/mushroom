import cv2
import numpy as np
from ultralytics import YOLO
import os
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 模型
model_path = r"C:\Users\sywan\Desktop\code\runs\segment\train9\weights\best.pt"
model = YOLO(model_path)

data_list = []

# ⭐開攝像頭
cap = cv2.VideoCapture(0)

frame_id = 0  # 用來當圖片ID

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = frame.copy()
    h_img, w_img = img.shape[:2]

    results = model.predict(source=img, conf=0.8, verbose=False)

    for r in results:

        if r.masks is not None:
            masks = r.masks.data.cpu().numpy()

            for i, mask in enumerate(masks):

                # resize mask
                mask = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
                mask = (mask * 255).astype(np.uint8)

                kernel = np.ones((5,5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if len(contours) > 0:
                    largest_contour = max(contours, key=cv2.contourArea)

                    if cv2.contourArea(largest_contour) < 500:
                        continue

                    if len(largest_contour) >= 5:

                        epsilon = 0.01 * cv2.arcLength(largest_contour, True)
                        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

                        ellipse = cv2.fitEllipse(approx)

                        cv2.ellipse(img, ellipse, (0, 255, 0), 2)

                        (cx, cy), (MA, ma), angle = ellipse
                        cx, cy = int(cx), int(cy)
                        cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)

                        # 長軸
                        if MA >= ma:
                            major = MA / 2.0
                            theta = np.deg2rad(angle)
                        else:
                            major = ma / 2.0
                            theta = np.deg2rad(angle + 90)

                        length_px = major * 2
                        length_norm = length_px / w_img

                        # ⭐即時資料（用 frame_id 當 ID）
                        data_list.append({
                            "frame": frame_id,
                            "mushroom_id": i,
                            "major_axis_px": length_px,
                            "major_axis_norm": length_norm
                        })

                        # 畫長軸
                        x1 = int(cx + major * np.cos(theta))
                        y1 = int(cy + major * np.sin(theta))
                        x2 = int(cx - major * np.cos(theta))
                        y2 = int(cy - major * np.sin(theta))

                        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

    frame_id += 1

    cv2.imshow("Mushroom Real-time Detection", img)

    # 按 q 離開
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ⭐輸出 Excel
df = pd.DataFrame(data_list)
df.to_excel("mushroom_realtime.xlsx", index=False)