import open3d as o3d
import cv2
import numpy as np
# 点云上映射颜色

def main():
    camera_matrix = np.array([[1686.4112483378, 0, 1048.37299068491],
                              [0, 1681.77170179537, 498.541473285944],
                              [0, 0, 1]], dtype=np.float32)
    R = np.array([[-0.99990782, 0.00779703, -0.01111607],
                  [-0.00769016, -0.99992412, -0.00962401],
                  [-0.01119027, -0.00953764, 0.9998919]])

    T = np.array([[-0.01699598],
                  [0.00031841],
                  [0.01059312]])
    distCoeffs = np.array([0.0754663, -0.4918826, 0, 0, 0], dtype=np.float32)  # Assuming no distortion

    # 读取点云文件
    pcd = o3d.io.read_point_cloud("D:/3Dplantdatas/相机数据/neuvsnap_0410_203238.pcd")
    image = cv2.imread("neuvsnap_0410_203238.jpg")
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    projected_points, _ = cv2.projectPoints(np.asarray(pcd.points), R, T, camera_matrix, distCoeffs)
    projected_points = projected_points.reshape(-1, 2)  # 调整形状以便索引

    point_color = []
    for point in projected_points:
        x, y = point  # 调整坐标顺序
        if x >= 0 and x < image.shape[1] and y >= 0 and y < image.shape[0]:
            color = image[int(y), int(x)]/255 # 归一化到0到1之间
        else:
            color = [0, 1, 0]  # 如果坐标超出图像范围，设为白色
        point_color.append(color)

    # 可视化点云
    pcd.colors = o3d.utility.Vector3dVector(point_color)
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    main()
