import os
import random
import argparse
import time
import numpy as np
import torch
from thop import profile

# 日志
import logging
# 可视化
import visdom

# 可视化
vis = visdom.Visdom()
loss_window = None  # 用于存储损失可视化窗口的引用


log_format = '%(asctime)s - %(levelname)s - %(message)s'
log_level = logging.INFO
current_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
log_file_name = f'{current_time}_train.log'
log_file_path = os.path.join(os.getcwd() + '\log', log_file_name)
logging.basicConfig(filename=log_file_path, level=log_level, format=log_format)


dataset = 'LungCT'

# 使用 pytorch 后端导入 voxelmorph
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # nopep8

# 解析命令行
parser = argparse.ArgumentParser()

# 参数
# parser.add_argument('--img-list',default='IXI_train.txt', help='line-seperated list of training files')
parser.add_argument('--img-list',default='train.txt', help='line-seperated list of training files')

parser.add_argument('--img-prefix', help='optional input image file prefix')
parser.add_argument('--img-suffix', help='optional input image file suffix')
parser.add_argument('--atlas', help='atlas filename (default: data/atlas_norm.npz)')
parser.add_argument('--model-dir', default=f'models/{dataset}',
                    help='model output directory (default: models)')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')

# 训练参数
parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=2000,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--steps-per-epoch', type=int, default=100,
                    help='frequency of model saves (default: 100)')
parser.add_argument('--load-model' ,help='optional model file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-6, help='learning rate (default: 1e-4)')
parser.add_argument('--cudnn-nondet', action='store_true',
                    help='disable cudnn determinism - might slow down training')

# 网络参数
parser.add_argument('--enc', type=int, nargs='+',
                    help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+',
                    help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')
parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')
parser.add_argument('--out_params_flag', type=bool, default=True,
                    help='whether to output model parameters and training parameters')
# 损失函数参数
parser.add_argument('--image-loss', default='mse',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--lambda', type=float, dest='weight', default=0.01,
                    help='weight of deformation loss (default: 0.01)')
args = parser.parse_args()

bidir = args.bidir

logging.info(' - '.join(('Dataset: ', dataset)))

# 加载和准备训练数据
train_files = vxm.py.utils.read_file_list(args.img_list, prefix=args.img_prefix,
                                          suffix=args.img_suffix)
assert len(train_files) > 0, 'Could not find any training data.'

# 如果数据是多通道的，则无需附加额外的特征轴
add_feat_axis = not args.multichannel

if args.atlas:
    # 扫描图集生成器
    atlas = vxm.py.utils.load_volfile(args.atlas, np_var='vol',
                                      add_batch_axis=True, add_feat_axis=add_feat_axis)
    generator = vxm.generators.scan_to_atlas(train_files, atlas,
                                             batch_size=args.batch_size, bidir=args.bidir,
                                             add_feat_axis=add_feat_axis)
else:
    # 扫描对扫描发生器
    generator = vxm.generators.scan_to_scan(
        train_files, batch_size=args.batch_size, bidir=args.bidir, add_feat_axis=add_feat_axis)
    # 特殊匹配方式
    # generator = vxm.generators.ct_to_ct(
    #     train_files, batch_size=args.batch_size, bidir=args.bidir, add_feat_axis=add_feat_axis)

# 从采样输入中提取形状
inshape = next(generator)[0][0].shape[1:-1]

# 准备模型文件夹
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)

# 设备处理
gpus = args.gpu.split(',')
nb_gpus = len(gpus)
device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
assert np.mod(args.batch_size, nb_gpus) == 0, \
    'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size, nb_gpus)

# 启用 cudnn 决定论似乎能大大加快训练速度
torch.backends.cudnn.deterministic = not args.cudnn_nondet

# unet 架构
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

if args.load_model:
    # 加载初始模型（如果指定）
    model = vxm.networks.VxmDense.load(args.load_model, device)
else:
    # 否则配置新模型
    model = vxm.networks.VxmDense(
        inshape=inshape,
        nb_unet_features=[enc_nf, dec_nf],
        bidir=bidir,
        int_steps=args.int_steps,
        int_downsize=args.int_downsize
    )

if nb_gpus > 1:
    # 通过 DataParallel 使用多个 GPU
    model = torch.nn.DataParallel(model)
    model.save = model.module.save

# 为训练准备模型并发送至设备
model.to(device)
model.train()

# 设置优化器
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# 准备图像损失
if args.image_loss == 'ncc':
    image_loss_func = vxm.losses.NCC().loss
elif args.image_loss == 'mse':
    image_loss_func = vxm.losses.MSE().loss
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

# 如果是双向的，则需要两个图像损失函数
if bidir:
    losses = [image_loss_func, image_loss_func]
    weights = [0.5, 0.5]
else:
    losses = [image_loss_func]
    weights = [1]

# 准备变形损失
losses += [vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss]
weights += [args.weight]

# 保存最佳模型
best_loss = float('inf')  # 初始化为正无穷大
best_model_state = None

out_params_flag = args.out_params_flag

# 训练
for epoch in range(args.initial_epoch, args.epochs):

    # save model checkpoint
    if epoch % 20 == 0:
        model.save(os.path.join(model_dir, '%04d.pt' % epoch))

    epoch_loss = []
    epoch_total_loss = []
    epoch_step_time = []

    for step in range(args.steps_per_epoch):

        step_start_time = time.time()

        # 生成输入（和真实输出）并将其转换为张量
        inputs, y_true = next(generator)

        inputs = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs]
        y_true = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in y_true]

        # 通过模型运行输入，生成扭曲图像和流场
        y_pred = model(*inputs)

        # 首次输出模型参数个数和训练参数
        if out_params_flag:
            flops, params = profile(model, (*inputs,))

            print('flops: ', flops, 'params: ', params)
            print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
            logging.info('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))

            out_params_flag = False

        # 计算总损失
        loss = 0
        loss_list = []
        for n, loss_function in enumerate(losses):
            curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
            loss_list.append(curr_loss.item())
            loss += curr_loss

        epoch_loss.append(loss_list)
        epoch_total_loss.append(loss.item())

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算时间
        epoch_step_time.append(time.time() - step_start_time)

    average_epoch_loss = np.mean(epoch_total_loss)
    if average_epoch_loss < best_loss:
        best_loss = average_epoch_loss
        best_model_state = model.state_dict()  # 保存当前模型的状态


    # 打印训练信息
    epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
    time_info = '%.4f sec/step' % np.mean(epoch_step_time)
    losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
    loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)

    logging.info(' - '.join((epoch_info, time_info, loss_info)))

    print(' - '.join((epoch_info, time_info, loss_info)), flush=True)

    if loss_window is None:
        loss_window = vis.line(X=[epoch], Y=[np.mean(epoch_total_loss)], opts=dict(title='{} Loss'.format(dataset)))
    else:
        vis.line(X=[epoch], Y=[np.mean(epoch_total_loss)], win=loss_window, update='append')

# 保存模型
model.save(os.path.join(model_dir, '%04d.pt' % args.epochs))
# 保存最佳模型
best_model_path = os.path.join(model_dir, 'best_model.pt')
torch.save(best_model_state, best_model_path)
