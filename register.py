import os
import argparse

# third party
import numpy as np
import nibabel as nib
import torch

# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm   # nopep8

# parse commandline args
parser = argparse.ArgumentParser()
parser.add_argument('--moving', default='dataset/OASIS/imagesTs/OASIS_0415_0000.nii.gz', help='moving image (source) filename')
parser.add_argument('--fixed', default='dataset/OASIS/imagesTs/OASIS_0416_0000.nii.gz', help='fixed image (target) filename')
parser.add_argument('--moved', default='output/moved.nii.gz', help='warped image output filename')
parser.add_argument('--model', default='models/OASIS/2000.pt', help='pytorch model for nonlinear registration')
parser.add_argument('--warp',  default= 'output/warp.nii.gz', help='output warp deformation filename')
parser.add_argument('-g', '--gpu', help='GPU number(s) - if not supplied, CPU is used')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
args = parser.parse_args()

# device handling
if args.gpu and (args.gpu != '-1'):
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
else:
    device = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# load moving and fixed images
add_feat_axis = not args.multichannel
moving = vxm.py.utils.load_volfile(args.moving, add_batch_axis=True, add_feat_axis=add_feat_axis)
fixed, fixed_affine = vxm.py.utils.load_volfile(
    args.fixed, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)

# load and set up model
model = vxm.networks.VxmDense.load(args.model, device)
model.to(device)
model.eval()

# set up tensors and permute
input_moving = torch.from_numpy(moving).to(device).float().permute(0, 4, 1, 2, 3)
input_fixed = torch.from_numpy(fixed).to(device).float().permute(0, 4, 1, 2, 3)

# predict
moved, warp = model(input_moving, input_fixed, registration=True)

# save moved image
if args.moved:
    moved = moved.detach().cpu().numpy().squeeze()
    vxm.py.utils.save_volfile(moved, args.moved, fixed_affine)

# save warp
if args.warp:
    warp = warp.detach().cpu().numpy().squeeze()
    vxm.py.utils.save_volfile(warp, args.warp, fixed_affine)

# 读取output/warp.nii.gz
warp_nifti = nib.load('output/warp.nii.gz')
warp = warp_nifti.get_fdata()
print('变形前的warp的shape：', warp.shape)

warp_reshaped = np.moveaxis(warp, [0, 1, 2], [3, 0, 1])
warp_final = np.expand_dims(warp_reshaped, axis=3)
print('变形后的warp的shape：', warp_final.shape)

warp2_nifti = nib.Nifti1Image(warp_final, affine=None)
nib.save(warp2_nifti, 'output/warp_corrected.nii.gz')
