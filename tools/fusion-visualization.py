import numpy as np
import open3d as o3d
import cv2

#点云投影到2D图像中

# 读取点云数据和图像
point_cloud = o3d.io.read_point_cloud("D:/3Dplantdatas/相机数据/neuvsnap_0410_203238.pcd")
image = cv2.imread("neuvsnap_0410_203238.jpg")
points = np.asarray(point_cloud.points)
# 读取相机内参矩阵(K)

# Step 2: Load camera intrinsic parameters and extrinsic matrix
# camera_matrix = np.array([[1737.289452802048, 0, 1034.214344002579],
#                           [0, 1732.459816732432, 500.8291899328670],
#                           [0, 0, 1]])

camera_matrix = np.array([[1686.4112483378, 0, 1048.37299068491],
                          [0, 1681.77170179537, 498.541473285944],
                          [0, 0, 1]], dtype=np.float32)
R = np.array([[-0.99990782,0.00779703,-0.01111607],
 [-0.00769016,-0.99992412,-0.00962401 ],
 [-0.01119027,-0.00953764,0.9998919]])

T = np.array([[-0.01699598],
              [0.00031841],
              [0.01059312]])
distCoeffs = np.array([0.0754663,-0.4918826,0,0,0],dtype=np.float32)  # Assuming no distortion
# 计算投影点在图像上的位置
projected_points, _ = cv2.projectPoints(points, R, T, camera_matrix,distCoeffs)
overlay_image = image.copy()
# 将点云数据融合到图像上
#红色
for point in projected_points:
    x,y=point.ravel()
    x = int(x)
    y = int(y)
    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
        color = point_cloud.colors[int(x), int(y)]
        overlay_image[y, x] = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))

# 显示融合后的图像
cv2.imshow("Fused Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()