import cv2
import numpy as np
from ultralytics import YOLO
import os
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 模型
model_path = r"C:\Users\sywan\Desktop\code\runs\segment\train9\weights\best.pt"
model = YOLO(model_path)

# 資料夾
input_folder = r"C:\Users\sywan\Desktop\code\mushP"
output_folder = r"C:\Users\sywan\Desktop\code\output_draw"

os.makedirs(output_folder, exist_ok=True)
data_list = []
# 讀取資料夾
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):

        image_path = os.path.join(input_folder, filename)
        #print(f"Processing: {filename}")

        results = model.predict(source=image_path, conf=0.8)

        for r in results:
            img = r.orig_img.copy()  # ⭐複製避免改原圖
            h_img, w_img = img.shape[:2]

            if r.masks is not None:
                masks = r.masks.data.cpu().numpy()

                for i, mask in enumerate(masks):
                    # resize mask 到原圖
                    mask = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
                    mask = (mask * 255).astype(np.uint8)
                    kernel = np.ones((5,5), np.uint8)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    # ⭐找輪廓
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # ⭐找最大輪廓（當作主體）
                    if len(contours) > 0:
                        largest_contour = max(contours, key=cv2.contourArea)

                        # ⭐輪廓點數要 >= 5 才能擬合橢圓
                        if len(largest_contour) >= 5:
                            epsilon = 0.01 * cv2.arcLength(largest_contour, True)
                            approx = cv2.approxPolyDP(largest_contour, epsilon, True)

                            ellipse = cv2.fitEllipse(approx)
                            

                            # ⭐畫橢圓
                            cv2.ellipse(img, ellipse, (0, 255, 0), 3)

                            # ⭐中心
                            (cx, cy), (MA, ma), angle = ellipse
                            cx, cy = int(cx), int(cy)
                            cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
                            # 判斷長軸方向
                            if MA >= ma:
                                major = MA / 2.0
                                theta = np.deg2rad(angle)
                            else:
                                major = ma / 2.0
                                theta = np.deg2rad(angle + 90)  # ⭐關鍵
                            length_px = major * 2  # 長軸像素長度
                            length_norm = length_px / w_img  # ⭐重點：標準化
                            data_list.append({
                                "filename": filename,
                                "mushroom_id": i,
                                "major_axis_px": length_px,
                                "major_axis_norm": length_norm
                            })

                            # 計算端點
                            x1 = int(cx + major * np.cos(theta))
                            y1 = int(cy + major * np.sin(theta))
                            x2 = int(cx - major * np.cos(theta))
                            y2 = int(cy - major * np.sin(theta))

                            minor = min(MA, ma) / 2.0
                            theta_perp = theta + np.pi / 2

                            x3 = int(cx + minor * np.cos(theta_perp))
                            y3 = int(cy + minor * np.sin(theta_perp))
                            x4 = int(cx - minor * np.cos(theta_perp))
                            y4 = int(cy - minor * np.sin(theta_perp))

                            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # 顯示
            #cv2.imshow("Result", img)
            #cv2.waitKey(1)

            # 存圖
            save_path = os.path.join(output_folder, filename)
            cv2.imwrite(save_path, img)
df = pd.DataFrame(data_list)
df.to_excel("mushroom_length.xlsx", index=False)

cv2.destroyAllWindows()