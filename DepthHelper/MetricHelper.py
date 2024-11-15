import torch
import sys
import numpy as np
import cv2

from DispHelper.DispHelper import DispHelper

sys.path.append('DepthHelper')

from Disp2Depth import DispDepthHelper
from Flow2Depth import FlowDepthHelper
from params.SystemParams import SystemParam

class MetricHelper:


    def __init__(self, cols: int, rows: int, sysparam=None):
        self.cols, self.rows = cols, rows
        self.len = len
        if sysparam is None:
            self.m_disp = None
            self.m_flow = None
            return
        self.m_system = sysparam
        self.m_disp = DispDepthHelper(sysparam)
        self.m_flow = FlowDepthHelper(cols, rows, sysparam)
        # self.m_flow = DispDepthHelper(sysparam)

    def SetSysparam(self, sysparam: SystemParam):
        self.m_system = sysparam
        if self.m_disp is None and self.m_flow is None:
            self.m_system = sysparam
            self.m_disp = DispDepthHelper(sysparam)
            self.m_flow = FlowDepthHelper(self.cols, self.rows, sysparam)
            # self.m_flow = DispDepthHelper(sysparam)
            return
        self.m_disp.SetSystemParam(sysparam)
        self.m_flow.SetSystemParam(sysparam)

    def Compare(self, disp_up: torch.Tensor, flow_up: torch.Tensor, gt_depth: torch.Tensor):
        disp_up_tensor = torch.from_numpy(disp_up).float()
        flow_up_tensor = torch.from_numpy(flow_up).float()
        gt_depth_tensor = torch.from_numpy(gt_depth).float()  # 这里读到的数据有点怪，应该存在单位的影响
        print(disp_up_tensor,flow_up_tensor)
        dispdepth = self.m_disp.TransformDepth(torch.abs(disp_up_tensor))  #这里这个*10不应该写，但写完结果相似了，后面需要看看
        flowdepth = self.m_flow.TransformDepth(torch.abs(flow_up_tensor))

        cmpdisp = torch.abs(gt_depth_tensor - dispdepth).numpy()
        cmpflow = torch.abs(gt_depth_tensor - flowdepth).numpy()
        #  看看异常值有多少


        # 缩小异常值

        # 比较后的错误
        dispErr = self.NormLize256(cmpdisp).transpose((1 , 2 , 0))
        flowErr = self.NormLize256(cmpflow).transpose((1, 2, 0))
        dispColor = cv2.applyColorMap(dispErr, 2)
        flowColor = cv2.applyColorMap(flowErr, 2)

        # dispret = self.NormLize256(dispdepth).transpose((1 , 2 , 0))
        # flowret = self.NormLize256(flowdepth).transpose((1, 2, 0))
        # # dispColor = cv2.applyColorMap(dispret, 2)
        # flowColor = cv2.applyColorMap(flowret, 2)


        cmpdisp = torch.mean(torch.abs(gt_depth_tensor - dispdepth))
        cmpflow = torch.mean(torch.abs(gt_depth_tensor - flowdepth))

        print("disp depth:", cmpdisp.item(), ", flow depth:", cmpflow.item())
        # if cmpdisp > cmpflow:
        #     print("flow accuracy is higher")
        # else:
        #     print("disp accuracy is higher")
        return cmpdisp.item() , cmpflow.item() , dispColor , flowColor

    def NormLize256(self,Image:np.ndarray):
        return ((Image - Image.min()) * 256 / (Image.max() - Image.min())).astype(np.uint8)