import cv2
import torch
import sys
sys.path.append('DLNRHelper')

from .core.dlnr import DLNR
from .core.utils.utils import InputPadder

class DLNRHelper:
    def __init__(self, path: str, args, DEVICE: str):
        self.loadModel(path, args, DEVICE)

    def loadModel(self, path: str, args, DEVICE: str):
        self.device = DEVICE
        self.m_model = torch.nn.DataParallel(DLNR(args))

        # Map the model to the appropriate device
        state_dict = torch.load(path, map_location=torch.device(DEVICE))
        self.m_model.load_state_dict(state_dict)

        self.m_model = self.m_model.module
        self.m_model.to(DEVICE)
        self.m_model.eval()

    def Estim(self, src1: cv2.Mat, src2: cv2.Mat, valid_iters: int):
        with torch.no_grad():
            img1 = torch.from_numpy(src1).permute(2, 0, 1).float()[None].to(self.device)
            img2 = torch.from_numpy(src2).permute(2, 0, 1).float()[None].to(self.device)
            padder = InputPadder(img1.shape)
            img1, img2 = padder.pad(img1, img2)
            flow_up = self.m_model(img1, img2, iters=16)

            flow_up = padder.unpad(flow_up[0]).squeeze(0).cpu().numpy()

        return flow_up