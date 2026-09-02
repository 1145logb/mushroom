import cv2
import numpy as np
from ultralytics import YOLO
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 模型
model_path = r"C:\Users\sywan\Desktop\code\runs\segment\train9\weights\best.pt"
model = YOLO(model_path)

# 資料夾
input_folder = r"C:\Users\sywan\Desktop\code\mushP"
output_folder = r"C:\Users\sywan\Desktop\code\output_draw"

os.makedirs(output_folder, exist_ok=True)

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

                    # ⭐找輪廓
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # ⭐畫紅色輪廓
                    cv2.drawContours(img, contours, -1, (0, 0, 255), 3)

            # 顯示
            #cv2.imshow("Result", img)
            #cv2.waitKey(1)

            # 存圖
            save_path = os.path.join(output_folder, filename)
            cv2.imwrite(save_path, img)

cv2.destroyAllWindows()