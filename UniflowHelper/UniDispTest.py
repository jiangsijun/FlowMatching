import sys

from sympy.unify import unify

sys.path.append('UniflowHelper')

import argparse
import glob
import numpy as np
import torch
# from tqdm import tqdm
from pathlib import Path
# from core.raft_stereo import RAFTStereo
from utils.utils import InputPadder
from PIL import Image
# from matplotlib import pyplot as plt
from unimatch.unimatch import UniMatch

DEVICE = 'cuda'


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def demo(args):
    model = UniMatch(feature_channels=128,
                     num_scales=1,
                     upsample_factor=8,
                     num_head=1,
                     ffn_dim_expansion=4,
                     num_transformer_layers=6,
                     reg_refine=False,
                     task='flow')

    # model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    model.load_state_dict(torch.load("../raft/gmflow-scale1-things.pth", map_location=DEVICE)['model'])

    # model = model.module
    model.to(DEVICE)
    model.eval()

    # output_directory = Path(args.output_directory)
    # output_directory.mkdir(exist_ok=True)

    with torch.no_grad():
        # left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        # right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        # print(f"Found {len(left_images)} images. Saving files to {output_directory}/")
        #
        # for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
        image1 = load_image("D:\\flow-matching-git\dataset\school\\view3\Image0001.png")
        image2 = load_image("D:\\flow-matching-git\dataset\school\\view3\Image0002.png")

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_up = model(image1, image2,
                           attn_type=args.attn_type,
                           attn_splits_list=[2],
                           corr_radius_list=[-1],
                           prop_radius_list=[-1],
                           num_reg_refine=1,
                           task='flow',
                           )
        print(flow_up)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='flow', choices=['flow', 'stereo', 'depth'], type=str)
    parser.add_argument('--num_scales', default=1, type=int,
                        help='feature scales: 1/8 or 1/8 + 1/4')
    parser.add_argument('--feature_channels', default=128, type=int)
    parser.add_argument('--upsample_factor', default=8, type=int)
    parser.add_argument('--num_head', default=1, type=int)
    parser.add_argument('--ffn_dim_expansion', default=4, type=int)
    parser.add_argument('--num_transformer_layers', default=6, type=int)
    parser.add_argument('--reg_refine', action='store_true',
                        help='optional task-specific local regression refinement')

    # model: parameter-free
    parser.add_argument('--attn_type', default='self_swin2d_cross_1d', type=str,
                        help='attention function')
    parser.add_argument('--attn_splits_list', default=[2], type=int, nargs='+',
                        help='number of splits in attention')
    parser.add_argument('--corr_radius_list', default=[-1], type=int, nargs='+',
                        help='correlation radius for matching, -1 indicates global matching')
    parser.add_argument('--prop_radius_list', default=[-1], type=int, nargs='+',
                        help='self-attention radius for propagation, -1 indicates global attention')
    parser.add_argument('--num_reg_refine', default=1, type=int,
                        help='number of additional local regression refinement')

    args = parser.parse_args()

    demo(args)
