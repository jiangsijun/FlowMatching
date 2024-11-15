# import sys

# print(sys.path)

import numpy as np
from params.Intrinsics import IntrinsicsParam
from params.Extrinsics import ExtrinsicsParam

#理论来说这个数据结构应该在当前研究中应用，因此只调试这种
class SystemParam:
    def __init__(self , lidarExt:ExtrinsicsParam , cameraExt:ExtrinsicsParam , refCamera:IntrinsicsParam , srcCamera:IntrinsicsParam , lidarParam:IntrinsicsParam):
        self.m_ref_cam = refCamera
        self.m_src_cam = srcCamera
        self.m_lidar = lidarParam
        self.m_lidar_cam = lidarExt
        self.m_cam_cam = cameraExt

    def SetLeftCamera(self , refCamera:IntrinsicsParam):
        self.m_ref_cam = refCamera

    def SetRightCamera(self, srcCamera:IntrinsicsParam):
        self.m_src_cam = srcCamera

    def SetLiDARParam(self , lidarParam :IntrinsicsParam) :
        self.m_lidar = lidarParam
    
    def SetLiDARExt(self , lidarExt:ExtrinsicsParam):
        self.m_lidar_cam = lidarExt

    def SetCamExt(self, camExt:ExtrinsicsParam) :
        self.m_cam_cam = camExt

    def GetRefCamera(self) :
        return self.m_ref_cam
    
    def GetSrcCamera(self) :
        return self.m_src_cam
    
    def GetLiDARParam(self):
        return self.m_lidar
    
    def GetLiDARExt(self):
        return self.m_lidar_cam
    
    def GetCamExt(self):
        return self.m_cam_cam
        

# #两个相机之间的外参  
# class Camera2Camera :
#     def __init__(self , R , t , leftCamera:CameraParam , rightCamera:CameraParam) :
#         self.m_R = R
#         self.m_t = t
#         self.m_t_len = np.linalg.norm(self.m_t)
#         self.m_t_unit = self.m_t / self.m_t_len
        
#         self.m_ref_cam = leftCamera
#         self.m_src_cam = rightCamera

# #单目相机和拍摄的雷达之间的外参
# class LiDAR2Camera:
#     def __init__(self, R , t , leftCamera: CameraParam) : 
#         self.m_R = R
#         self.m_t = t
#         self.m_t_len = np.linalg.norm(self.m_t)
#         self.m_t_unit = self.m_t / self.m_t_len
#         self.m_ref_cam= leftCamera

