from scipy.optimize import least_squares
import numpy as np

#求映射矩阵 P=K[R|t]

# 定义目标函数
def objective_function(m, points_cloud, points_pixel):
    m = m.reshape((3, 4))  # 将参数向量m转换为3x4矩阵形式
    X, Y, Z = points_cloud.T
    u, v = points_pixel.T
    u_pred = np.dot(m[0], [X, Y, Z, 1])
    v_pred = np.dot(m[1], [X, Y, Z, 1])
    # w_pred = np.dot(m[2], [X, Y, Z, 1])
    # u_pred /= w_pred
    # v_pred /= w_pred
    residuals = np.concatenate([(u_pred - u), (v_pred - v)])
    return residuals
points_cloud = np.array([[0.227, 0.229, 1.0700001],
                         [0.212, -0.044, 1.027],
                         [-0.047, 0.222, 1.097],
                          [-0.048, -0.038, 1.057],

                        [0.20299999, 0.24699999, 1.1619999],
                         [0.19599999, -0.019, 1.1390001],
                         [-0.085, 0.23999999, 1.184],
                          [-0.081, -0.03, 1.155],

                         [0.324, 0.244, 1.149],
                         [0.30000001, -0.035, 1.092],
                         [0.039, 0.243, 1.1900001],
                          [0.038, -0.032, 1.141],

                         [0.18700001, 0.257, 1.2309999],
                         [0.17, -0.03, 1.184],
                         [-0.098, 0.248, 1.243],
                          [-0.105, -0.026, 1.2180001],

                         [0.103, 0.25400001, 1.232],
                         [0.088, -0.028, 1.1950001],
                         [-0.18700001, 0.25400001, 1.255],
                          [-0.17900001, -0.022, 1.23],

                         [0.099, 0.25400001, 1.2309999],
                         [0.088, -0.028, 1.1950001],
                         [-0.18700001, 0.25400001, 1.255],
                          [-0.185, -0.02, 1.243],

                         [0.097, 0.226, 1.102],
                         [0.09, -0.044, 1.052],
                         [-0.18000001, 0.223, 1.119],
                         [-0.17900001, -0.036, 1.091],

                         [0.233, 0.228, 1.1],
                         [0.219, -0.041, 1.0470001],
                         [-0.043, 0.226, 1.122],
                         [-0.04, -0.038, 1.071],

                          ], dtype=np.float32)

# 2D图像中对应的三个点的坐标（相机坐标系）
points_pixel = np.array([[642, 108],
                         [642, 548],
                         [1070, 109],
                         [1070, 550],

                        [528, 112],
                         [527, 536],
                         [941, 116],
                         [942, 536],

                         [527, 110],
                         [529, 534],
                         [941, 114],
                         [945, 536],

                         [750, 123],
                         [752, 525],
                         [1138, 124],
                         [1138, 524],

                         [862, 124],
                         [861, 524],
                         [1250, 120],
                         [1248, 523],

                         [860, 123],
                         [861, 525],
                         [1249, 120],
                         [1250, 525],

                         [844, 115],
                         [846, 553],
                         [1270, 111],
                         [1268, 554],

                         [ 635, 110],
                         [636, 550],
                         [1063, 110],
                         [1063, 552],
                         ], dtype=np.float32)
# 使用最小二乘法拟合约束方程组，得到初始解
m_init = np.random.rand(12)  # 初始解
result_lsq = least_squares(objective_function, m_init, args=(points_cloud, points_pixel))

# 使用Levenberg-Marquardt算法进行优化
result = least_squares(objective_function, result_lsq.x, method='lm', args=(points_cloud, points_pixel))

# 输出最终结果
print("Optimized m values:", result_lsq.x)
vaule=[ 0.91278074,0.84274417,0.96026848,1.15334279,
        0.41912989 ,0.44374147,0.59014326,0.19819018,
        0.58300362,0.07233162,-0.17547 ,0.32903095]