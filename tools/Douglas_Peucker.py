import cv2
import numpy as np

#图像的标记点值

# 全局变量
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # 当左键按下时
        print("点击坐标：", (x, y))
# 读取图像
image = cv2.imread('neuvsnap_0410_203238.jpg', 0)

# 使用Canny算子检测图像中的边缘
edges = cv2.Canny(image, 50, 150)



# 轮廓检测
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 初始化特征点列表
feature_points = []

# 对每个轮廓应用改进的Douglas-Peucker算法
for contour in contours:
    # 计算轮廓的周长
    epsilon = 0.01 * cv2.arcLength(contour, True)
    # 使用改进的Douglas-Peucker算法进行曲线简化
    approx = cv2.approxPolyDP(contour, epsilon, True)
    # 提取特征点
    for point in approx:
        x, y = point.ravel()
        feature_points.append((x, y))

# 在图像上绘制特征点
for point in feature_points:
    cv2.circle(image, point, 3, (0, 255, 0), -1)  # 使用绿色绘制特征点

# 显示带有特征点的图像
cv2.imshow('Feature Points', image)

# 设置鼠标回调函数
cv2.setMouseCallback('Feature Points', mouse_callback)

cv2.waitKey(0)
cv2.destroyAllWindows()