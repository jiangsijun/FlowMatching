import sys

from sympy.unify import unify

sys.path.append('CrocoFlowHelper')

import argparse
import glob
import numpy as np
import torch
# from tqdm import tqdm
from pathlib import Path
# from core.raft_stereo import RAFTStereo
from PIL import Image
from stereoflow.criterion import *
from models.croco_downstream import CroCoDownstreamBinocular
from models.head_downstream import PixelwiseTaskWithDPT
from stereoflow.engine import tiled_pred
from stereoflow.datasets_stereo import img_to_tensor

from utils.utils import InputPadder


DEVICE = 'cuda'


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()/255
    return img[None].to(DEVICE)


def demo(args):
    ckpt = torch.load("../raft/crocostereo.pth", 'cpu')

    ckpt_args = ckpt['args'] #模型里也存了参数
    task = "stereo"
    tile_conf_mode = ckpt_args.tile_conf_mode
    num_channels = {'stereo': 1, 'flow': 2}[task]
    with_conf = eval(ckpt_args.criterion).with_conf
    if with_conf: num_channels += 1
    print('head: PixelwiseTaskWithDPT()')
    head = PixelwiseTaskWithDPT()
    head.num_channels = num_channels
    print('croco_args:', ckpt_args.croco_args)
    model = CroCoDownstreamBinocular(head, **ckpt_args.croco_args)
    msg = model.load_state_dict(ckpt['model'], strict=True)
    # model.eval()
    # model = model.to('cuda')


    # model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    # model.load_state_dict(torch.load("../raft/gmflow-scale1-things.pth", map_location=DEVICE)['model'])

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

        padder = InputPadder(image1.shape,divis_by=32)
        image1, image2 = padder.pad(image1, image2)
        # im1 = img_to_tensor(image1).to(DEVICE).unsqueeze(0)
        # im2 = img_to_tensor(image2).to(DEVICE).unsqueeze(0)
        with torch.inference_mode():
            # InputPadder()
            # pred, _, _ =z tiled_pred(model, None, im1, im2, None, conf_mode=tile_conf_mode, overlap=tile_overlap,
                                    # crop=cropsize, with_conf=with_conf, return_time=False)
            pred, _, _ = tiled_pred(model, None, image1, image2, None, conf_mode=tile_conf_mode, overlap=0.9,
                                    crop=ckpt_args.crop, with_conf=with_conf, return_time=False)

        pred = pred.squeeze(0).squeeze(0).cpu().numpy()
        print(pred)


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


