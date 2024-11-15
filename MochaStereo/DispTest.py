import sys
sys.path.append('MochaStereo')

import argparse
import glob
import numpy as np
import torch
from pathlib import Path
from core.mocha_stereo import Mocha
from core.utils.utils import InputPadder
from PIL import Image


DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def demo(args):
    model = torch.nn.DataParallel(Mocha(args), device_ids=[0])
    model.load_state_dict(torch.load("../raft/mocha-stereo.pth"))

    model = model.module
    model.to(DEVICE)
    model.eval()


    with torch.no_grad():
        image1 = load_image("D:\\flow-matching-git\dataset\school\\view3\Image0001.png")
        image2 = load_image("D:\\flow-matching-git\dataset\school\\view3\Image0002.png")

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)


        flow_up = model(image1, image2, iters=args.valid_iters, test_mode=True)
        print(flow_up)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="datasets/Middlebury/MiddEval3/testH/*/im0.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="datasets/Middlebury/MiddEval3/testH/*/im1.png")
    parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg",
                        help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true',
                        help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")

    args = parser.parse_args()

    demo(args)
