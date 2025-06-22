import cv2

def resize_and_show(image, window_name='Resized Image'):
    new_width = 500
    new_height = 800
    resized_image = cv2.resize(image, (new_width, new_height))
    cv2.imshow(window_name, resized_image)

# 讀取與預處理
img = cv2.imread(r'C:\Users\User\Desktop\deeplearning\mushroomphoto\pretty_jpg\IMG_0634.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        # 轉灰階
blurred = cv2.medianBlur(gray, 17)                  # 去噪
edges = cv2.Laplacian(blurred, -1, 11, 5)        # Laplacian 邊緣偵測

# 二值化處理（轉成黑白）
_, binary = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 輪廓偵測
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 繪製輪廓
contour_img = img.copy()
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)  


resize_and_show(img, 'Original')
resize_and_show(binary, 'Binary Image')
resize_and_show(contour_img, 'Contours Detected')

cv2.waitKey(0)
cv2.destroyAllWindows()
