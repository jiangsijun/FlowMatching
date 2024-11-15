import cv2
import sys
sys.path.append('UniflowHelper')
import torch
from .utils.utils import InputPadder
from .unimatch.unimatch import UniMatch

class UniDispHelper:
    def __init__(self, path: str, args, DEVICE: str):
        self.loadModel(path, args, DEVICE)

    def loadModel(self, path: str, args, DEVICE: str):
        self.device = DEVICE

        self.m_model = UniMatch(feature_channels=128,
                         num_scales=1,
                         upsample_factor=8,
                         num_head=1,
                         ffn_dim_expansion=4,
                         num_transformer_layers=6,
                         reg_refine=False,
                         task='stereo').to(self.device)

        # model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
        # self.m_model.load_state_dict(torch.load("../raft/gmstereo-scale1-sceneflow.pth", map_location=DEVICE)['model'])

        # Map the model to the appropriate device
        state_dict = torch.load(path, map_location=torch.device(DEVICE))
        self.m_model.load_state_dict(state_dict['model'])

        self.m_model.to(DEVICE)
        self.m_model.eval()

    def Estim(self, src1: cv2.Mat, src2: cv2.Mat , iter):
        with torch.no_grad():
            img1 = torch.from_numpy(src1).permute(2, 0, 1).float()[None].to(self.device)
            img2 = torch.from_numpy(src2).permute(2, 0, 1).float()[None].to(self.device)
            padder = InputPadder(img1.shape)
            img1, img2 = padder.pad(img1, img2)
            flow_up = self.m_model(img1, img2,
                            attn_type='self_swin2d_cross_1d',
                            attn_splits_list=[2],
                            corr_radius_list=[-1],
                            prop_radius_list=[-1],
                            num_reg_refine=1,
                            task='stereo',
                            )
            flow_pr = flow_up['flow_preds'][-1]

            flow_pr = padder.unpad(flow_pr[0]).cpu()

        return flow_pr
