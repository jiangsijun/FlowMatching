import cv2
import sys
sys.path.append('CrocoHelper')
import torch
from PIL import Image
from stereoflow.criterion import *
from models.croco_downstream import CroCoDownstreamBinocular
from models.head_downstream import PixelwiseTaskWithDPT
from stereoflow.engine import tiled_pred
from stereoflow.datasets_flow import img_to_tensor

class CrocoFlowHelper:
    def __init__(self, path: str, args, DEVICE: str):
        self.loadModel(path, args, DEVICE)

    def loadModel(self, path: str, args, DEVICE: str):
        self.device = DEVICE

        ckpt = torch.load(path, self.device)
        self.ckpt_args = ckpt['args']  # 模型里也存了参数
        task = 'flow'
        self.tile_conf_mode =  self.ckpt_args.tile_conf_mode
        num_channels = {'stereo': 1, 'flow': 2}[task]
        self.with_conf = eval( self.ckpt_args.criterion).with_conf
        if self.with_conf: num_channels += 1
        # print('head: PixelwiseTaskWithDPT()')
        head = PixelwiseTaskWithDPT()
        head.num_channels = num_channels
        # print('croco_args:', ckpt_args.croco_args)
        self.m_model = CroCoDownstreamBinocular(head, ** self.ckpt_args.croco_args)
        msg = self.m_model.load_state_dict(ckpt['model'], strict=True)
        # Map the model to the appropriate device
        # state_dict = torch.load(path, map_location=torch.device(DEVICE))


        self.m_model.to(DEVICE)
        self.m_model.eval()

    def Estim(self, src1: cv2.Mat, src2: cv2.Mat , iter):
        with torch.no_grad():
            # img1 = torch.from_numpy(src1).permute(2, 0, 1).float()[None].to(self.device)
            # img2 = torch.from_numpy(src2).permute(2, 0, 1).float()[None].to(self.device)
            img1 = img_to_tensor(src1).to(self.device).unsqueeze(0)
            img2 = img_to_tensor(src2).to(self.device).unsqueeze(0)
            with torch.inference_mode():
                pred, _, _ = tiled_pred(self.m_model, None, img1, img2, None, conf_mode=self.tile_conf_mode, overlap=0.9,
                                        crop= self.ckpt_args.crop, with_conf=self.with_conf, return_time=False)

            pred = pred.squeeze(0).squeeze(0).cpu().numpy()

        return pred
