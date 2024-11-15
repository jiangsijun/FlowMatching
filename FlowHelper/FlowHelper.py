import sys
sys.path.append('FlowHelper')
import cv2
import torch
from core.utils.utils import InputPadder
import collections

from .core.raft import RAFT

class FlowHelper:
    def __init__(self, path: str, args, DEVICE: str):
        self.loadModel(path, args, DEVICE)

    def loadModel(self, path: str, args, DEVICE: str):
        self.device = DEVICE
        self.m_model = RAFT(args)  # Using RAFT model, can be replaced with other models later
        state_dict = torch.load(path, map_location=torch.device(DEVICE))
        new_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove "module."
            new_state_dict[name] = v
        self.m_model.load_state_dict(new_state_dict)  # load_state_dict shows missing data
        self.m_model.to(DEVICE)
        self.m_model.eval()

    def Estim(self, src1: cv2.Mat, src2: cv2.Mat, valid_iters: int):
        with torch.no_grad():
            img1 = torch.from_numpy(src1).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
            img2 = torch.from_numpy(src2).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
            padder = InputPadder(img1.shape)
            img1, img2 = padder.pad(img1, img2)

            _, flow_up = self.m_model(img1, img2, iters=valid_iters, test_mode=True)
            flow_up = padder.unpad(flow_up).squeeze(0).cpu().numpy()
        return flow_up