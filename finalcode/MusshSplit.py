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
output_folder = r"C:\Users\sywan\Desktop\code\output"

os.makedirs(output_folder, exist_ok=True)

# 讀取資料夾
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):

        image_path = os.path.join(input_folder, filename)
        print(f"Processing: {filename}")

        results = model.predict(source=image_path, conf=0.5)

        for r in results:
            img = r.orig_img
            h_img, w_img = img.shape[:2]

            if r.masks is not None:
                masks = r.masks.data.cpu().numpy()

                for i, mask in enumerate(masks):
                    # mask 對齊原圖
                    mask = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
                    mask = (mask * 255).astype(np.uint8)

                    # 擷取
                    segmented = cv2.bitwise_and(img, img, mask=mask)

                    # 找 bounding box
                    ys, xs = np.where(mask > 0)
                    if len(xs) == 0 or len(ys) == 0:
                        continue

                    x1, x2 = xs.min(), xs.max()
                    y1, y2 = ys.min(), ys.max()

                    # ⭐直接裁切（不做任何 resize）
                    cropped = segmented[y1:y2, x1:x2]

                    # 存檔
                    save_name = f"{os.path.splitext(filename)[0]}_obj{i}.png"
                    save_path = os.path.join(output_folder, save_name)
                    cv2.imwrite(save_path, cropped)

cv2.destroyAllWindows()