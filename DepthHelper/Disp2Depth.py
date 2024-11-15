# from typing import Any
from params.SystemParams import SystemParam
import numpy as np

class DispDepthHelper:
    def __init__(self, sysparam:SystemParam) :
        self.m_system = sysparam
        pass

    def SetSystemParam(self, sysparam:SystemParam):
        self.m_system = sysparam 
        return
    
    def GetSystemParam(self):
        return self.m_system
    
    def TransformDepth(self, flow_up:np.array):
        # 这里的T的单位是mm，即为B*f，换算时应统一单位，原本的B * f / 视差，在这里写作B / disp
        depth = np.linalg.norm(self.m_system.GetCamExt().GetTVector())  / flow_up
        return depth
     