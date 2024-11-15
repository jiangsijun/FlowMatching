import cv2
import numpy as np

#给特定点映射在2D图像上的呈现

def transform_point_cloud(point_cloud, camera_matrix, rotation_matrix, translation_vector):
    # 将点云坐标从雷达坐标系转换为相机坐标系
    transformed_points = np.dot(rotation_matrix, point_cloud.T).T + translation_vector

    # 将相机坐标系中的点云投影到图像平面
    projected_points = np.dot(camera_matrix, transformed_points.T).T

    # 将二维坐标归一化
    image_points = projected_points[:, :2] / projected_points[:, 2:]

    return image_points

# Example parameters
points_cloud = np.array([[ 0.53600001, -0.31600001,  1.274],
                         [0.064,0.18799999,1.418 ],
                         [-0.228,-0.42899999,3.0650001],
                          [-0.41,-0.081,1.302],

                        [-0.31200001,0.082,1.349],
                         [-0.41800001,0.17299999,1.418],
                         [-0.22400001,-0.56999999,3.0840001],
                          ], dtype=np.float32)

camera_matrix = np.array([[1686.4112483378, 0, 1048.37299068491],
                          [0, 1681.77170179537, 498.541473285944],
                          [0, 0, 1]], dtype=np.float32)
R = np.array([[-0.99990782,0.00779703,-0.01111607],
 [-0.00769016,-0.99992412,-0.00962401 ],
 [-0.01119027,-0.00953764,0.9998919]])

T = np.array([[-0.01699598],
              [0.00031841],
              [0.01059312]])
distCoeffs = np.array([0.0754663,-0.4918826,0,0,0],dtype=np.float32)
# Project point cloud to image
projected_points, _ = cv2.projectPoints(points_cloud, R, T, camera_matrix, None)

print("Projected points on image:")
print(projected_points)