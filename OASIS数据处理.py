import os
import nibabel as nib
import numpy as np
import pickle

# 从.pkl文件中加载数据
# def pkload(fname):
#     with open(fname, 'rb') as f:
#         return pickle.load(f)

# 定义源目录和目标目录
# source_dir = 'dataset/IXI_data/Train'
# img_target_dir = 'dataset/IXI_data_nii/Train/img'
# label_target_dir = 'dataset/IXI_data_nii/Train/label'

# source_dir = 'dataset/IXI_data/Test'
# img_target_dir = 'dataset/IXI_data_nii/Test/img'
# label_target_dir = 'dataset/IXI_data_nii/Test/label'

# 遍历源目录下的所有.pkl文件
# for root, dirs, files in os.walk(source_dir):
#     for file in files:
#         if file.endswith('.pkl'):
#             # 构建完整的文件路径
#             pkl_file_path = os.path.join(root, file)
#
#             # 加载图像和标签数据
#             image, label = pkload(pkl_file_path)
#
#             # 构建目标文件路径
#             img_nii_file_path = os.path.join(img_target_dir, file.replace('.pkl', '_image.nii.gz'))
#             label_nii_file_path = os.path.join(label_target_dir, file.replace('.pkl', '_label.nii.gz'))
#
#             # 确保目标目录存在
#             os.makedirs(os.path.dirname(img_nii_file_path), exist_ok=True)
#             os.makedirs(os.path.dirname(label_nii_file_path), exist_ok=True)
#
#             # 将数据包装成Nifti1Image对象
#             img_nifti = nib.Nifti1Image(image, affine=np.eye(4))  # 使用单位矩阵作为仿射矩阵
#             label_nifti = nib.Nifti1Image(label, affine=np.eye(4))  # 使用单位矩阵作为仿射矩阵
#
#             # 保存为.nii.gz文件
#             nib.save(img_nifti, img_nii_file_path)
#             nib.save(label_nifti, label_nii_file_path)
#
#             print(f'Saved: {img_nii_file_path} and {label_nii_file_path}')

import os

# 指定要读取文件路径的目录
directory = 'dataset/OASIS_split/test'

# 获取目录下的所有文件路径
file_paths = [os.path.join(directory, filename) for filename in os.listdir(directory)]

# 对文件路径列表进行排序
file_paths = sorted(file_paths)

# 将文件路径写入到文件中
with open('OASIS_test.txt', 'w') as file:

    for i, path in enumerate(file_paths):
        if i == len(file_paths) - 1:
            file.write(file_paths[i] + ',' + file_paths[0] + '\n')
        else:
            file.write(file_paths[i] + ',' + file_paths[i+1] + '\n')