# 官方库
from __future__ import print_function

import os
import torch
import time
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
# 私人库
from model import DnCNN, ADNet, Deam
from public import findLastCheckpoint, parse_args, sum_squared_error, log
import data_process as dp
from data_process import DenoisingDataset

import socket
import pandas as pd
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data import get_training_set, get_eval_set

from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from real_dataloader import *


if __name__ == '__main__':
    # 加载模型
    model = DnCNN()
    # model = ADNet()
    # 参数初始化
    # 运行解析器并放置提取的数据在 argparse.Namespace 对象
    args = parse_args(0)
    batch_size = args.batch_size  # 一次训练所抓取的数据样本数量 BATCH_SIZE的大小影响训练速度和模型优化。
    cuda = torch.cuda.is_available()  # 函数返回值为bool值
    n_epoch = args.epoch  # 当一个完整的数据集通过了神经网络一次并且返回了一次，为一个epoch
    sigma = args.sigma  # 高斯噪声级别
    # 保存模型路径
    save_dir = os.path.join('models', args.model + '_' + 'sigma' + str(sigma))
    if not os.path.exists(save_dir):  # 若不存在路径则添加路径
        os.mkdir(save_dir)

    initial_epoch = findLastCheckpoint(save_dir=save_dir)  # load the last model in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        # model.load_state_dict(torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch)))
        model = torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch))  # 加载模型
    # model.train()
    # model.train()的作用是启用 Batch Normalization 和 Dropout。

# 如果模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加model.train()。
    # model.train()是保证BN层能够用到每一批数据的均值和方差。对于Dropout，model.train()是随机取一部分网络连接来训练更新参数。
    # criterion = nn.MSELoss(reduction = 'sum')  # PyTorch 0.4.1
    criterion = sum_squared_error()  # 测量元素均方误差
    # GPU or CPU
    if cuda:
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # adam优化算法
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)  # learning rates
    # 按设定的间隔调整学习率。这个方法适合后期调试使用，观察 loss 曲线，为每个实验定制学习率调整时机
    # milestones(list)- 一个 list，每一个元素代表何时调整学习率， list 元素必须是递增的。如 milestones=[30,80,120]
    # gamma(float)- 学习率调整倍数，默认为 0.1 倍，即下降 10 倍。

    # 训练
    # epoch = 1
    for epoch in range(initial_epoch, n_epoch):
        scheduler.step(epoch)  # step to the learning rate in this epcoh 恢复optimizer的学习率
        xs = dp.datagenerator(data_dir=args.train_data)   # 路径名 生成多批次的数据集
        xs = xs.astype('float32') / 255.0  # 数组转为float32 , 归一化
        xs = torch.from_numpy(xs.transpose((0, 3, 1, 2)))  # tensor of the clean patches, NXCXHXW
        # 变为张量， 238336 ， 1， 40 ， 40
        DDataset = DenoisingDataset(xs, sigma)    # 返回batch_y, batch_x
        # 加噪声
        # 238336 / batch_size
        DLoader = DataLoader(dataset=DDataset, num_workers=1, drop_last=True, batch_size=batch_size, shuffle=True)
        # DLoader = DataLoader(dataset=DDataset, num_workers=1, drop_last=True, batch_size=batch_size, shuffle=False)
        # dataset (Dataset) – 决定数据从哪读取或者从何读取；
        # num_workers (python:int, optional) – 是否多进程读取数据（默认为０);
        # drop_last (bool, optional) – 当样本数不能被batchsize整除时，最后一批数据是否舍弃（default: False)
        # batchszie：批大小，决定一个epoch有多少个Iteration；
        # shuffle (bool, optional) –每一个 epoch是否为乱序 (default: False)；

        epoch_loss = 0
        start_time = time.time()
        for n_count, batch_yx in enumerate(DLoader):
            # batch_yx 为两个列表， 列表的第一个和第二个元素均是64，1，40，40的张量
            # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中
            optimizer.zero_grad()  # 清空过往梯度；
            if cuda:
                batch_x, batch_y = batch_yx[1].cuda(), batch_yx[0].cuda()
            # loss = criterion(model(batch_yx[0]), batch_yx[1])
            loss = criterion(model(batch_y), batch_x)  # 计算损失函数 测量预测值与实际值之间差异的函数
            epoch_loss += loss.item()
# .item()方法是，取一个元素张量里面的具体元素值并返回该值，可以将一个零维张量转换成int型或者float型，在计算loss，accuracy时常用到。
# 作用：
# 1.item（）取出张量具体位置的元素元素值
# 2.并且返回的是该位置元素值的高精度值
# 3.保持原元素类型不变；必须指定位置
# 4.节省内存（不会计入计算图）


            loss.backward()  # 反向传播，计算当前梯度
            optimizer.step()  # 根据梯度更新网络参数

            ceshi1 = xs.size(0)
            ceshi = xs.size(0) // batch_size
            if n_count % 10 == 0:  # 每循环10遍，输出一次结果
                print('%4d %4d / %4d loss = %2.4f' % (
                epoch + 1, n_count, xs.size(0) // batch_size, loss.item() / batch_size))
        elapsed_time = time.time() - start_time  # 结束时间

        log('epcoh = %4d , loss = %4.4f , time = %4.2f s' % (epoch + 1, epoch_loss / n_count, elapsed_time))
        np.savetxt('train_result.txt', np.hstack((epoch + 1, epoch_loss / n_count, elapsed_time)), fmt='%2.4f')
        # 保存
        torch.save(model, os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))  # 保存模型
    # torch.save(model, os.path.join(save_dir, 'model_001.pth'))
    # torch.save(model, os.path.join(save_dir, 'model_%03d.pth' % (epoch + 1)))















# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
#
# # Training settings
# parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
#
# parser.add_argument('--batch_size', type=int, default=2, help='training batch size')
# parser.add_argument('--upscale_factor', type=int, default=1, help="super resolution upscale factor")
# parser.add_argument('--batchSize', type=int, default=2, help='training batch size')
# parser.add_argument('--nEpochs', type=int, default=2000, help='number of epochs to train for')
# parser.add_argument('--start_iter', type=int, default=1, help='starting epoch')
# parser.add_argument('--lr', type=float, default=0.0001, help='learning rate. default=0.0001')
# parser.add_argument('--data_augmentation', type=bool, default=True, help='if adopt augmentation when training')
# parser.add_argument('--hr_train_dataset', type=str, default='', help='the training dataset')
# parser.add_argument('--Ispretrained', type=bool, default=True, help='If load checkpoint model')
# parser.add_argument('--pretrained_sr', default='Real.pth', help='sr pretrained base model')
# parser.add_argument('--pretrained', default='./Deam_models', help='Location to load checkpoint models')
# parser.add_argument("--noiseL", type=float, default=25, help='noise level')
# parser.add_argument('--save_folder', default='./checkpoint/', help='Location to save checkpoint models')
# parser.add_argument('--statistics', default='./statistics/', help='Location to save statistics')
# parser.add_argument('--epoch', default=180, type=int, help='number of train epoches')#epoch 整型  默认180
# # Testing settings
# parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size, default=1')
# parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
# parser.add_argument('--test_dataset', type=str, default='Set12', help='the testing dataset')
# parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
#
# # Global settings
# parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
# parser.add_argument('--gpus', default=1, type=int, help='number of gpus')
# parser.add_argument('--data_dir', type=str, default='./data', help='the dataset dir')
# parser.add_argument('--model_type', type=str, default='Deam', help='the name of model')
# parser.add_argument('--patch_size', type=int, default=128, help='Size of cropped HR image')
# parser.add_argument('--Isreal', default=True, help='If training/testing on RGB images')
#
#
# opt = parser.parse_args()
# gpus_list = range(opt.gpus)
# hostname = str(socket.gethostname())
# cudnn.benchmark = True
# print(opt)
#
#
# def train(epoch):
#     epoch_loss = 0
#     model.train()
#     for iteration, batch in enumerate(training_data_loader, 1):
#         target = Variable(batch[0])
#         noise = torch.FloatTensor(target.size()).normal_(mean=0, std=opt.val_noiseL / 255.)
#         input = target + noise
#
#         input = input.cuda()
#         target = target.cuda()
#
#         model.zero_grad()
#         optimizer.zero_grad()
#         t0 = time.time()
#
#         prediction = model(input)
#
#         # Corresponds to the Optimized Scheme
#         loss = criterion(prediction, target)/(input.size()[0]*2)
#
#         t1 = time.time()
#         epoch_loss += loss.data
#         loss.backward()
#         optimizer.step()
#
#         if (iteration+1) % 50 == 0:
#             model.eval()
#             SC = 'net_epoch_' + str(epoch) + '_' + str(iteration + 1) + '.pth'
#             torch.save(model.state_dict(), os.path.join(opt.save_folder, SC))
#             model.train()
#         if not os.path.exists(opt.save_folder):  # 结果路径
#             os.mkdir(opt.save_folder)  # 创造目录
#         #
#         print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration, len(training_data_loader), loss.data, (t1 - t0)))
#     print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
#     if not os.path.exists(os.path.join(opt.save_folder, 'model_%03d.pth' % (epoch + 1))):  # 结果路径
#         torch.save(model, os.path.join(opt.save_folder, 'model_%03d.pth' % (epoch + 1)))  # 保存模型
#
# def batch_PSNR(img, imclean, data_range):
#     Img = img.data.cpu().numpy().astype(np.float32)
#     Iclean = imclean.data.cpu().numpy().astype(np.float32)
#     PSNR = 0
#     for i in range(Img.shape[0]):
#         PSNR += compare_psnr(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
#     return (PSNR / Img.shape[0])
#
#
# def test(testing_data_loader):
#     psnr_test= 0
#     model.eval()
#     for batch in testing_data_loader:
#         target = Variable(batch[0])
#         noise = torch.FloatTensor(target.size()).normal_(mean=0, std=opt.noiseL / 255.)
#         input = target + noise
#
#         input = input.cuda()
#         target = target.cuda()
#         with torch.no_grad():
#             prediction = model(input)
#             prediction = torch.clamp(prediction, 0., 1.)
#         psnr_test += batch_PSNR(prediction, target, 1.)
#     print("===> Avg. PSNR: {:.4f} dB".format(psnr_test / len(testing_data_loader)))
#     return psnr_test / len(testing_data_loader)
#
#
# def print_network(net):
#     num_params = 0
#     for param in net.parameters():
#         num_params += param.numel()
#     print(net)
#     print('Total number of parameters: %d' % num_params)
#
#
# def checkpoint(epoch,psnr):
#     model_out_path = opt.save_folder+hostname+opt.model_type+"_psnr_{}".format(psnr)+"_epoch_{}.pth".format(epoch)
#     torch.save(model.state_dict(), model_out_path)
#     print("Checkpoint saved to {}".format(model_out_path))
#
#
# if __name__ == '__main__':
#     print('===> Loading datasets')
#
#     if opt.Isreal:
#         train_set = Dataset_h5_real(src_path=os.path.join(opt.data_dir, 'train', 'train.h5'), patch_size=opt.patch_size, train=True)
#         training_data_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size, shuffle=True, num_workers=4,
#                                 drop_last=True)
#         test_set = Dataset_h5_real(src_path=os.path.join(opt.data_dir, 'test', 'val.h5'), patch_size=opt.patch_size, train=False)
#         testing_data_loader = DataLoader(dataset=test_set, batch_size=opt.testBatchSize, shuffle=False, num_workers=0, drop_last=True)
#     else:
#         train_set = get_training_set(os.path.join(opt.data_dir, 'Train400'), opt.hr_train_dataset, opt.upscale_factor,
#                                      opt.patch_size, opt.data_augmentation)
#         training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
#
#         test_set = get_eval_set(os.path.join(opt.data_dir+'/Test', opt.test_dataset), opt.upscale_factor)
#         testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)
#
#     print('===> Building model ', opt.model_type)
#     model = Deam(opt.Isreal)
#
#     model = torch.nn.DataParallel(model, device_ids=gpus_list)
#     criterion = nn.MSELoss()
#
#     print('---------- Networks architecture -------------')
#     print_network(model)
#     print('----------------------------------------------')
#
#     if opt.Ispretrained:
#         model_name = os.path.join(opt.pretrained, opt.pretrained_sr)
#         model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
#         print(model_name + ' model is loaded.')
#
#     optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
#
#     PSNR = []
#     for epoch in range(opt.start_iter, opt.nEpochs + 1):
#         train(epoch)
#         psnr = test(testing_data_loader)
#         PSNR.append(psnr)
#         data_frame = pd.DataFrame(
#             data={'epoch': epoch, 'PSNR': PSNR}, index=range(1, epoch+1)
#         )
#         data_frame.to_csv(os.path.join(opt.statistics, 'training_logs.csv'), index_label='index')
#         # learning rate is decayed by a factor of 10 every half of total epochs
#         if (epoch + 1) % (opt.nEpochs / 2) == 0:
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] /= 10.0
#             print('Learning rate decay: lr={}'.param_group['lr'])