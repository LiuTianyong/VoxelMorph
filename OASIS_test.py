import os
import argparse

import numpy as np
import nibabel as nib
import torch

os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # nopep8
from voxelmorph.torch import layers
from voxelmorph.py.utils import dice as DICE
from voxelmorph.py.utils import get_jacobi_matrix
from tqdm import tqdm  # 导入tqdm
import time


def transformer_3D_dice(moving_file, fixed_file, moved_file, model_file, warp_file, warp_cor_file, moving_label_file, fixed_label_file,
         moved_label_file, multichannel, gpu):
    if gpu and (gpu != '-1'):
        device = 'cuda'
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    else:
        device = 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # 加载配准图和固定图
    add_feat_axis = not multichannel
    moving = vxm.py.utils.load_volfile(moving_file, add_batch_axis=True, add_feat_axis=add_feat_axis)
    fixed, fixed_affine = vxm.py.utils.load_volfile(fixed_file, add_batch_axis=True, add_feat_axis=add_feat_axis,
                                                    ret_affine=True)

    # 加载标签
    moving_label = vxm.py.utils.load_volfile(moving_label_file, add_batch_axis=True, add_feat_axis=add_feat_axis)
    fixed_label = nib.load(fixed_label_file).get_fdata()


    # 加载模型
    model = vxm.networks.VxmDense.load(model_file, device)
    model.to(device)
    model.eval()

    # 初始化变形模型
    inshape = moving_label.shape[1:-1]

    transformer = layers.SpatialTransformer(inshape, mode='nearest').to(device)
    transformer.eval()

    # 设置张量并进行置换
    input_moving = torch.from_numpy(moving).to(device).float().permute(0, 4, 1, 2, 3)
    input_fixed = torch.from_numpy(fixed).to(device).float().permute(0, 4, 1, 2, 3)
    moving_label = torch.from_numpy(moving_label).to(device).float().permute(0, 4, 1, 2, 3)

    # 预测
    moved, warp = model(input_moving, input_fixed, registration=True)

    # 预测变形场
    moved_label = transformer(moving_label, warp)

    # 计算dice系数
    dice_value = DICE(moved_label[0, 0, ...].cpu().detach().numpy(), fixed_label, include_zero=True)
    print('mean_dice_value:', np.mean(dice_value))

    warp_np = np.moveaxis(warp[0, ...].cpu().detach().numpy(), [0, 1, 2, 3], [3, 1, 2, 0])
    # 增加一个维度
    warp_np = np.expand_dims(warp_np, axis=0)

    # 计算jacobian行列式
    jacobian_determinant = get_jacobi_matrix(warp_np)
    # 计算满足条件的元素个数
    num_negative_jacobian = np.sum(jacobian_determinant <= 0)
    # 计算雅可比行列式的总元素个数
    total_elements = np.prod(jacobian_determinant.shape)
    # 计算占比
    percentage_negative_jacobian = (num_negative_jacobian / total_elements) * 100.0

    # Save moved image
    if moved_file:
        moved = moved.detach().cpu().numpy().squeeze()
        vxm.py.utils.save_volfile(moved, moved_file, fixed_affine)

    # Save moved label
    if moved_label_file:
        moved_label = moved_label.detach().cpu().numpy().squeeze()
        vxm.py.utils.save_volfile(moved_label, moved_label_file, fixed_affine)

    # Save warp
    if warp_file:
        warp = warp.detach().cpu().numpy().squeeze()
        vxm.py.utils.save_volfile(warp, warp_file, fixed_affine)

    # 读取output/warp.nii.gz
    warp_nifti = nib.load(warp_file)
    warp = warp_nifti.get_fdata()

    warp_reshaped = np.moveaxis(warp, [0, 1, 2], [3, 0, 1])
    warp_final = np.expand_dims(warp_reshaped, axis=3)

    warp_cor_nifti = nib.Nifti1Image(warp_final, affine=None)
    nib.save(warp_cor_nifti, warp_cor_file)

    return np.mean(dice_value), num_negative_jacobian, percentage_negative_jacobian, total_elements

if __name__ == '__main__':

    dataset_name = 'OASIS'

    # Read Test.txt
    with open(f'{dataset_name}_test.txt', 'r') as file:
        test_lines = file.readlines()

    # 检测output/OASIS/moved文件夹是否存在
    if not os.path.exists(f'output/{dataset_name}/moved'):
        os.makedirs(f'output/{dataset_name}/moved')

    # 检测output/OASIS/warp文件夹是否存在
    if not os.path.exists(f'output/{dataset_name}/warp'):
        os.makedirs(f'output/{dataset_name}/warp')

    # 检测output/OASIS/warp_cor文件夹是否存在
    if not os.path.exists(f'output/{dataset_name}/warp_cor'):
        os.makedirs(f'output/{dataset_name}/warp_cor')

    # 检测output/OASIS/moved_label文件夹是否存在
    if not os.path.exists(f'output/{dataset_name}/moved_label'):
        os.makedirs(f'output/{dataset_name}/moved_label')

    dice_values, jacobian_dets = [], []

    # 开始时间
    start_time = time.time()
    for line in tqdm(test_lines, desc="Processing samples", ncols=100):
        fixed_file, moving_file = line.strip().split(',')

        # 使用反斜杠和下划线分割路径
        fixed_file_name = fixed_file.split('\\')[-1].split('_')[1]
        moving_file_name = moving_file.split('\\')[-1].split('_')[1]

        # 获取moving文件的标签文件名
        moving_label_file = 'dataset/OASIS/labelsTr/OASIS_{}_0000.nii.gz'.format(moving_file_name)
        fixed_label_file = 'dataset/OASIS/labelsTr/OASIS_{}_0000.nii.gz'.format(fixed_file_name)

        # 配准后的label文件名
        moved_label_file = os.path.join('output/OASIS/moved_label',
                                        fixed_file_name + '_' + moving_file_name + '_moved_label.nii.gz')

        # 配准后的文件名
        moved_file_name = fixed_file_name + '_' + moving_file_name + '_moved.nii.gz'
        moved_file = os.path.join('output/OASIS/moved', moved_file_name)

        # 保存形变场
        warp_file_name = fixed_file_name + '_' + moving_file_name + '_warp.nii.gz'
        warp_file = os.path.join('output/OASIS/warp', warp_file_name)

        # 保存变化后的变形场
        warp_cor_file_name = fixed_file_name + '_' + moving_file_name + '_warp_cor.nii.gz'
        warp_cor_file = os.path.join('output/OASIS/warp_cor', warp_cor_file_name)

        # 模型文件名
        model_file = 'models/OASIS/2000.pt'

        multichannel = False
        gpu = '0'

        dice, num_negative_jacobian, percentage_negative_jacobian, total_elements  = transformer_3D_dice(moving_file, fixed_file, moved_file, model_file, warp_file, warp_cor_file, moving_label_file,
             fixed_label_file, moved_label_file, multichannel, gpu)

        print('dice_value:{}  jacobian_value:{} percentage_negative_jacobian:{}'.format(dice, num_negative_jacobian,
                                                                                        percentage_negative_jacobian))

        dice_values.append(dice)
        jacobian_dets.append(percentage_negative_jacobian)

    # 结束时间
    end_time = time.time()
    print('end_time: ', end_time)

    print('time: ', end_time - start_time)
    print('avg_time:  ', (end_time - start_time) / len(test_lines))

    print('mean_dice_value:  ', np.mean(dice_values))
    print('mean_jacobian_det:', np.mean(jacobian_dets))
