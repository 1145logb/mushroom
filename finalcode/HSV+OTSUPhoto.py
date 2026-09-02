from ultralytics import YOLO
import cv2
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

model_path = r"C:\Users\sywan\Desktop\code\runs\segment\train9\weights\best.pt"
img_path   = r"C:\Users\sywan\Desktop\code\mushP0517\best\blue\IMG_0641.jpg"

model = YOLO(model_path)

# -----------------------------
# YOLO segmentation crop
# -----------------------------
def cropped_image(img_path, conf_threshold=0.8):
    image = cv2.imread(img_path)
    result = model(img_path)[0]

    crops = []
    masks_list = []
    boxes_info = []



    for i, box in enumerate(result.boxes):
        if box.conf[0] >= conf_threshold:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            if result.masks is not None:

                mask = result.masks.data[i].cpu().numpy()

                mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask_resized = (mask_resized > 0.5).astype(np.uint8) * 255

                segmented = cv2.bitwise_and(image, image, mask=mask_resized)

                crop_img = segmented[y1:y2, x1:x2]

                crops.append(crop_img)
                masks_list.append(mask_resized[y1:y2, x1:x2])
                boxes_info.append((x1, y1, x2, y2, float(box.conf[0])))

            else:
                crop_img = image[y1:y2, x1:x2]

                crops.append(crop_img)
                masks_list.append(None)
                boxes_info.append((x1, y1, x2, y2, float(box.conf[0])))

    return crops, masks_list, boxes_info


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

# -----------------------------
# white ratio
# -----------------------------
def calculate_white_ratio(white_mask, mushroom_mask):

    white_pixels = np.sum(white_mask > 0)
    mushroom_pixels = np.sum(mushroom_mask > 0)

    if mushroom_pixels == 0:
        return 0.0

    ratio = (white_pixels / mushroom_pixels) * 100
    return ratio


# -----------------------------
# main
# -----------------------------
crops, masks_list, boxes = cropped_image(img_path)

if not crops:
    print("❌ 沒有偵測到靈芝")

else:
    for i, crop in enumerate(crops):

        # ✅ 修正：正確解包
        croped, image_hsv, binary = img_processing(crop)

        white_img, white_mask = mark_white_as_blue(croped, image_hsv)

        body_img, mushroom_mask   = mark_body_as_red(croped, image_hsv)

        ratio = calculate_white_ratio(
            white_mask,
            mushroom_mask
        )

        print(
            f"第 {i+1} 顆靈芝 "
            f"信心值: {boxes[i][4]:.3f} "
            f"白邊占比: {ratio:.2f}%"
        )

        cv2.imshow(f"Crop {i+1} - 原圖", croped)
        cv2.imshow(f"Crop {i+1} - 白邊", white_img)
        cv2.imshow(f"Crop {i+1} - white mask", white_mask)
        cv2.imshow(f"Crop {i+1} - mushroom mask", body_img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()