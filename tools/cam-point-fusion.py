import open3d as o3d
import cv2
import numpy as np

#点云投影到2D图像中

# Step 1: Load PCD file
pcd = o3d.io.read_point_cloud("D:/3Dplantdatas/相机数据/neuvsnap_0410_203238.pcd")

# Step 2: Load camera intrinsic parameters and extrinsic matrix
# camera_matrix = np.array([[1737.289452802048, 0, 1034.214344002579],
#                           [0, 1732.459816732432, 500.8291899328670],
#                           [0, 0, 1]])

camera_matrix = np.array([[1686.4112483378, 0, 1048.37299068491],
                          [0, 1681.77170179537, 498.541473285944],
                          [0, 0, 1]], dtype=np.float32)

distCoeffs = np.array([0.0754663,-0.4918826,0,0,0],dtype=np.float32)  # Assuming no distortion

# External parameters (rotation and translation)
###1737
# R = np.array([[-0.99790137,0.02584761,-0.05936964],
#  [-0.02560838,-0.9996605,-0.0047871 ],
#  [ -0.05947322,-0.0032567,0.99822459]])
#
# T = np.array([[0.04375626],
#               [-0.00384675],
#               [-0.00273908]])
###1696
R = np.array([[-0.99990782,0.00779703,-0.01111607],
 [-0.00769016,-0.99992412,-0.00962401 ],
 [-0.01119027,-0.00953764,0.9998919]])

T = np.array([[-0.01699598],
              [0.00031841],
              [0.01059312]])
##solvePnP
# R_pnp = np.array([[0.99355161,-0.01423718,-0.11248337],
#  [0.00138926,0.99353903,-0.11348249 ],
#  [0.11337228,0.11259444,0.98715207]])
#
# T_pnp= np.array([[0.1683657],
#               [0.14532059],
#               [-2.30044808]])
# R_1 = np.array([[ -0.73650638,-0.14262031,-0.66122447],
#  [0.04133038,0.02584761,0.16645912 ],
#  [ -0.67516676,0.09526954,0.73148722]])
#
# T_1 = np.array([[0.75843432],
#               [-0.21623066],
#               [-0.3193027 ]])
# distCoeffs = np.array([0.2379,-1.288,0,0],dtype=np.float32)
M=np.array([[-1.50175366e+03,3.21772737e+01,-2.29897257e+01,9.93579118e+02],
  [1.60107532e+01, -1.56903951e+03, 1.76974741e+02, 2.85451976e+02],
  [4.08032455e-01, 5.14449810e-01,  8.28786110e-01,  8.04089791e-01]]
)
M_1=np.array(
[[-1.49991131e+03,  2.96999571e+01, -1.15080278e+01,  9.80521993e+02],
  [1.51738332e+01, -1.56884818e+03,  1.75943295e+02,  2.86654792e+02],
  [1.96143851e-01,  1.02444499e-02,  1.10922696e-02,  2.19712083e-01]]

)
points = np.asarray(pcd.points)
# Step 3: Project point cloud onto image plane
projected_points, _ = cv2.projectPoints(points, R, T, camera_matrix,distCoeffs)

X,Y,Z = points.T
u = np.dot(M[0],[X,Y,Z,1])
v = np.dot(M[1],[X,Y,Z,1])
m_projected_points=np.concatenate([[u.T],[v.T]]).T.reshape(-1,1,2)

# Step 4: Extract color from image
image = cv2.imread("neuvsnap_0410_203238.jpg")
colors = []
for point in projected_points:
    u, v = point.ravel()
    u = int(u)
    v = int(v)
    if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
        cv2.circle(image, (u, v), 1, (0, 255, 0), -1)  # Draw a green circle at projected point

# Step 6: Show the fused image
cv2.imshow("Fused Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()