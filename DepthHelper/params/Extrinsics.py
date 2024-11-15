import numpy as np
import torch

class ExtrinsicsParam:
    m_R = None
    m_t = None
    m_Rt = None

    def __init__(self, rmatrix, tvector):
        if isinstance(rmatrix, torch.Tensor):
            rmatrix = rmatrix.cpu().numpy()
        if isinstance(tvector, torch.Tensor):
            tvector = tvector.cpu().numpy()
        self.m_R = rmatrix
        self.m_t = tvector
        self.m_Rt = np.concatenate((rmatrix, tvector), axis=1)

    def SetRMatrix(self, rmatrix):
        if isinstance(rmatrix, torch.Tensor):
            rmatrix = rmatrix.cpu().numpy()
        self.m_R = rmatrix
        self.m_Rt = np.concatenate((rmatrix, self.m_t), axis=1)

    def SetTVector(self, tvector):
        if isinstance(tvector, torch.Tensor):
            tvector = tvector.cpu().numpy()
        self.m_t = tvector
        self.m_Rt = np.concatenate((self.m_R, tvector), axis=1)

    def SetRT(self, RT):
        if isinstance(RT, torch.Tensor):
            RT = RT.cpu().numpy()
        self.m_Rt = RT
        self.m_R = RT[:, :3]
        self.m_t = RT[:, -1]

    def GetRMatrix(self):
        return self.m_R

    def GetTVector(self):
        return self.m_t

    def GetRT(self):
        return self.m_Rt