import numpy as np
from DepthHelper.params.SystemParams import SystemParam
import cv2

# 这个计算 有点繁琐，应该可以再优化一下
PI = 3.1415926


class FlowDepthHelper:
    def __init__(self, cols: int, rows: int, systemParam: SystemParam):
        # xArray,yArray指示当前的位置在笛卡尔坐标系下的x轴,y轴坐标值，坐标系中心在影像坐标系的正中心
        self.cols = cols
        self.rows = rows
        self.xArray = np.arange(cols)
        self.srcxArray = np.expand_dims(self.xArray, 0).repeat(rows, axis=0)

        self.yArray = np.arange(rows).reshape(rows, 1)
        self.srcyArray = np.repeat(self.yArray, cols, axis=1) + 1e-5  # 加上很小的角度，避免除0

        self.m_system = systemParam
        self.initOpticalFlowField()

        return

    def SetSystemParam(self, systemParam: SystemParam):
        self.m_system = systemParam
        self.initOpticalFlowField()
        return

    def initOpticalFlowField(self):
        # 这里的值是针对源图像的固定值，后续不应进行修改
        _, _, cx, cy = self.m_system.GetRefCamera().GetParam()

        self.xArray = self.srcxArray - cx  # 以像素数为单位，将图像坐标系转换至相机坐标系
        self.yArray = self.srcyArray - cy

        self.fi = np.arctan(self.xArray / self.yArray)
        self.r = np.sqrt(self.xArray ** 2 + self.yArray ** 2 + 1)  # self.f **2)

        self.sincita = 1 / self.r
        zArray = np.ones((self.rows, self.cols, 1), float)
        # zArray = zArray * self.m_cam1.m_fx
        xArray = self.xArray.reshape(self.rows, self.cols, 1)
        yArray = self.yArray.reshape(self.rows, self.cols, 1)

        self.P = np.concatenate((xArray, yArray, zArray),
                                axis=2)  # [[[x,y,z][x,y,z]]]  [    -601.06     -388.03      3252.3]
        self.pVec_unit = np.linalg.norm(self.P, axis=2)  # 每个元素是(x,y,z)的长度坐标
        self.pVec_unit = self.P / np.expand_dims(self.pVec_unit, axis=2)  # 化为单位向量

        # 关于外参的计算，计算两坐标轴之间的平移单位向量
        self.m_t = self.m_system.GetCamExt().GetTVector()
        self.tVec_len = np.linalg.norm(self.m_t)
        self.tVec_unit = (self.m_t / self.tVec_len).transpose(1, 0)

    def TransformDepth(self, flow_up):
        # 只有flow_up为规定的变量，其他值均为恒定值
        rows, cols = flow_up[0].shape
        _, _, ref_cx, ref_cy = self.m_system.GetSrcCamera().GetParam()

        zArray = np.ones((rows, cols, 1), float)
        xArray = (flow_up[0] + self.srcxArray - ref_cx).reshape(rows, cols, 1)
        yArray = (flow_up[1] + self.srcyArray - ref_cy).reshape(rows, cols, 1)

        # vVec:src中每个匹配点的位置
        vVec = np.concatenate((xArray, yArray, zArray), axis=2)  # 估测的光流位置

        # 由原坐标系向目标坐标系的转变
        tmpvVec = np.swapaxes(vVec, 0, 2)  # 0和2的坐标维度对调
        w, h, c = vVec.shape
        tmpvVec = tmpvVec.reshape((c, -1))
        # rinv = np.linalg.inv(self.m_system.GetCamExt().GetRMatrix())
        rinv = self.m_system.GetCamExt().GetRMatrix()
        tmpvVec = rinv @ tmpvVec

        tmpvVec = tmpvVec.reshape((c, h, w))
        vVec = np.swapaxes(tmpvVec, 0, 2)
        # self.m_outCam.m_R.multiple(tmpvVec,)

        # vVec =  vVec @ (self.m_outCam.m_R)  #转换到位移出来的的坐标系
        vVec_unit = np.linalg.norm(vVec, axis=2)
        vVec_unit = vVec / np.expand_dims(vVec_unit, axis=2)  # 左图中心到对应点的单位向量（指示方向）

        cosbeta = np.sum(self.pVec_unit * vVec_unit, axis=2)
        cosbeta = np.where(cosbeta > 1, 1, cosbeta  )
        cosbeta = np.where(cosbeta < -1, -1, cosbeta)
        beta = np.arccos(cosbeta)  # 求出顶角大小
        maskbeta = np.abs(beta)
        cut = 1 / (PI * 100)  # 截断值
        maskbeta = np.where(maskbeta < cut, 1, 0)
        posmask = np.where(maskbeta * beta > 0, 1, 0)
        negmask = maskbeta - posmask

        beta = beta * (1 - maskbeta) + posmask * cut - negmask * cut
        # print(beta / PI * 180)
        # beta = np.where(beta. < 1/3.14 , 1/3.14 , beta)   # 这个夹角过小导致正中央这一行的值过大，因此在这里截断
        sinbeta = np.sin(beta)

        # print(vVec_unit.shape , )
        # cosalpha = np.sum(vVec_unit *(-self.tVec_unit) , axis=2)

        # alpha = np.arccos(cosalpha )
        # sinalpha = np.sin(alpha)
        cosgama = -np.sum(self.tVec_unit * self.pVec_unit, axis=2)
        cosgama = np.where(cosgama > 1, 1, cosgama)
        cosgama = np.where(cosgama < -1, -1, cosgama)
        gama = np.arccos(cosgama)
        sinalpha = np.sin(PI - gama - beta)

        depthplus = self.tVec_len * sinalpha / sinbeta

        cutplus = depthplus / self.r
        return cutplus




