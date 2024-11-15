import cv2
import torch
from .core.utils.utils import InputPadder
import collections

from .core.raft_stereo import RAFTStereo


class DispHelper:
    def __init__(self, path: str, args, DEVICE: str):
        self.loadModel(path, args, DEVICE)
        return

    def loadModel(self, path: str, args, DEVICE: str):
        # self.m_model = RAFTStereo(args)  #这里用了RAFT模型，后面可以换为其他模型
        # state_dict = torch.load(path,map_location=torch.device(DEVICE))
        self.device = DEVICE
        # new_state_dict = collections.OrderedDict()
        # for k, v in state_dict.items():
        #     name = k[7:]  # remove "module."
        #     new_state_dict[name] = v
        # self.m_model.load_state_dict(new_state_dict) #这里load_state_dict显示缺了数据
        # self.m_model.to(DEVICE)
        # self.m_model.eval()
        self.m_model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
        self.m_model.load_state_dict(torch.load(path))
        self.m_model = self.m_model.module
        self.m_model.to(DEVICE)
        self.m_model.eval()

    def Estim(self, src1: cv2.Mat, src2: cv2.Mat, valid_iters: int):
        with torch.no_grad():
            img1 = torch.from_numpy(src1).permute(2, 0, 1).float()[None].to(self.device)
            img2 = torch.from_numpy(src2).permute(2, 0, 1).float()[None].to(self.device)
            # img1 = img1.unsqueeze(0)
            # img2 = img2.unsqueeze(0)
            padder = InputPadder(img1.shape, divis_by=32)
            img1, img2 = padder.pad(img1, img2)
            _, flow_up = self.m_model(img1, img2, iters=valid_iters, test_mode=True)
            flow_up = padder.unpad(flow_up).squeeze(0).numpy()

        return flow_up

        # if calculateType == InputType.WithoutBackGroundCut.value :
        #     flow_up.cpu().numpy().squeeze()

        #     #裁剪flow_up
        #     return None
        # if calculateType == InputType.WithROIFullPic.value :
        #     #裁剪flow_up
        #     return None
        # intensor = -flow_up.cpu().detach().numpy().squeeze() #torch转opencvMat
        # # transforms.ToPILImage()
        # return intensor


