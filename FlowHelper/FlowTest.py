import sys
sys.path.append('FlowHelper')

import cv2
import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from core.raft import RAFT
from core.utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
from FlowHelper import FlowHelper


DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    return img
    # img = torch.from_numpy(img).permute(2, 0, 1).float()
    # return img[None].to(DEVICE)

def demo(args):
    helper = FlowHelper()
    helper.loadModel(args.restore_ckpt , args , DEVICE)
    img1 = cv2.imread("/data/jsj/dev/cpp/DisparityToCut/build-Debug/231118/plant1/1_1_rectify.jpg")
    img2 = cv2.imread("/data/jsj/dev/cpp/DisparityToCut/build-Debug/231118/plant1/1_2_rectify.jpg")
    helper.Estim(img1 , img2 , 32)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='/data/jsj/dev/python/RAFT-master/models/raft-small.pth')
    parser.add_argument('--small' , default=True)
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="datasets/Middlebury/MiddEval3/testH/*/im0.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="datasets/Middlebury/MiddEval3/testH/*/im1.png")
    parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    
    args = parser.parse_args()

    demo(args)
