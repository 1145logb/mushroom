import cv2
import numpy as np
from scipy.spatial import distance

# 读取图像
image = cv2.imread("C:\\Users\\User\\Desktop\\deeplearning\\mushroompic\\mush1.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 二值化
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# 形态学处理，去掉噪点
kernel = np.ones((5, 5), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# 轮廓检测
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
max_contour = max(contours, key=cv2.contourArea)

# 获取外接矩形
x, y, w, h = cv2.boundingRect(max_contour)

# 计算最大轮廓中最远的两点（确保摆放方向不同也能检测）
max_distance = 0
point1, point2 = None, None
for i in range(len(max_contour)):
    for j in range(i + 1, len(max_contour)):
        dist = distance.euclidean(max_contour[i][0], max_contour[j][0])
        if dist > max_distance:
            max_distance = dist
            point1, point2 = max_contour[i][0], max_contour[j][0]

# 显示检测结果
cv2.drawContours(image, [max_contour], -1, (0, 255, 0), 2)
cv2.circle(image, tuple(point1), 5, (255, 0, 0), -1)
cv2.circle(image, tuple(point2), 5, (255, 0, 0), -1)

# 计算真实长度（需标定）
pixels_per_cm = 50  # 假设 50 像素 = 1cm，需校准
real_length = max_distance / pixels_per_cm
print(f"灵芝最大长度：{real_length:.2f} cm")

# 显示结果
cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
