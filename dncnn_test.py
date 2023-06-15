# 官方库
import os
import time
import torch
import cv2
import torch.nn as nn
import argparse
import pandas as pd
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import public
from imageio import imread
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from torch.autograd import Variable
 # 私人库

from public import parse_args, log
from data_process import save_result, show
from model import DnCNN, Deam
#
if __name__ == '__main__':
    # 参数
    args = parse_args(1)  #测试集

    # model = DnCNN()
    if not os.path.exists(os.path.join(args.model_dir, args.model_name)):  # model若不存在，则加载model

        model_cpu = torch.load(os.path.join(args.model_dir, 'model.pth'), map_location='cpu')  # 映射到CPU，此模型不会被加载到cuda
        model = torch.load(os.path.join(args.model_dir, 'model.pth'), map_location='cpu')  # 映射到CPU
        # load weights into new model
        log('load trained model on Train400 dataset by kai')  #张凯的模型
    else:
        # model.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_name)))
        model_cpu = torch.load(os.path.join(args.model_dir, args.model_name), map_location='cpu')
        model = torch.load(os.path.join(args.model_dir, args.model_name), map_location='cpu')
        log('load trained model')   #本地训练的模型

    #    params = model.state_dict()
    #    print(params.values())
    #    print(params.keys())
    #
    #    for key, value in params.items():
    #        print(key)    # parameter name
    #    print(params['dncnn.12.running_mean'])
    #    print(model.state_dict())

    model_cpu.eval()  # evaluation mode  设置为预测模式
    #    model.train()

    if torch.cuda.is_available():  # GPU
        model = model.cuda()  # 加载到GPU

    if not os.path.exists(args.result_dir):  # 结果路径
        os.mkdir(args.result_dir)  # 创造目录

    for set_cur in args.set_names:  # 测试图片的文件名

        # if not os.path.exists(os.path.join(args.result_dir, set_cur)):  # 未找到保存文件的路径，则创造路径
        #     os.mkdir(os.path.join(args.result_dir, set_cur))
        psnrs = []  # 计算psnr与ssim的数组
        ssims = []

        for im in os.listdir(os.path.join(args.set_dir, set_cur)):  # 返回指定的文件夹包含的文件或文件夹的名字的列表
            if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):
                # 判断字符串是否以指定后缀结尾，如果以指定后缀结尾返回True，否则返回False
                x = np.array(imread(os.path.join(args.set_dir, set_cur, im)), dtype=np.float32) / 255.0
                # 读入文件，float32位，得到后每个数/255,归一化
                #  dtype:数组中的数据类型

                np.random.seed(seed=0)  # for reproducibility 随机数种子
                y = x + np.random.normal(0, args.sigma / 255.0, x.shape)   # 从正态（高斯）分布中抽取随机样本
                # 分布的均值（中心）0，分布的标准差（宽度）噪声级别/255 输出值的维度。为X的维度
                # Add Gaussian noise without clipping
                y = y.astype(np.float32)  # 转换数据类型 float32位
                y_ = torch.from_numpy(y).view(1, -1, y.shape[0], y.shape[1])
                # 创建张量，维度 1 1 481 321
                # view重构维度
                torch.cuda.synchronize()  # 等待当前设备上所有流中的所有核心完成
                start_time = time.time()  # 计算代码运行时间
                # ceshi = y_.nelement
                # ceshi2 = y_.squeeze(2)
                # ceshi = y.size
                if y.size < 154402:  # 图片较小，用GPU测试
                    y_ = y_.cuda()
                    x_ = model(y_)  # 使用模型对y_进行处理，输出x_
                    # inference
                else:   # 图片较大，用GPU测试
                    x_ = model_cpu(y_)
                x_ = x_.view(y.shape[0], y.shape[1])  # 把x_维度处理为二维
                x_ = x_.cpu()  # 将变量放在CPU上
                x_ = x_.detach().numpy().astype(np.float32)  # 整理为float32 数组 阻断反向传播
                torch.cuda.synchronize()
                elapsed_time = time.time() - start_time
                print('%10s : %10s : %2.4f second' % (set_cur, im, elapsed_time))

                psnr_x_ = compare_psnr(x, x_)  # 比较 原图 与 加噪声再去噪的图 计算psnr
                ssim_x_ = compare_ssim(x, x_)

                # ssim_x_ = compare_ssim(x, x_)

                # if args.save_result:
                name, ext = os.path.splitext(im)  # 文件名 后缀
                # show(np.hstack((y, x_)))  # show the image
                # save_result(x_, path=os.path.join(args.result_dir, set_cur,
                #                                   name + '_dncnn' + ext))
                # save the denoised image  矩阵 ， 路径
                psnrs.append(psnr_x_)  # 向列表末尾添加元素
                ssims.append(ssim_x_)
        psnr_avg = np.mean(psnrs)  # np.mean求平均值
        ssim_avg = np.mean(ssims)
        psnrs.append(psnr_avg)
        ssims.append(ssim_avg)
        # if args.save_result:
        # ceshi = os.path.join(args.result_dir, 'sigma50', set_cur)
        public.path_creat(os.path.join(args.result_dir, 'sigma'+str(args.sigma), set_cur))
        # ceshi = 'sigma'+str(args.sigma)
        ceshi = np.hstack((psnrs, ssims))
        ceshi1 = os.path.join(args.result_dir, 'sigma'+str(args.sigma), set_cur, 'results.txt')
        save_result(np.hstack((psnrs, ssims)), path=os.path.join(args.result_dir, 'sigma'+str(args.sigma), set_cur, 'results.txt'))
        # 以文本形式 保存每一张图片的PSNR与SSIM结果
        log('Datset: {0:10s} \n  PSNR = {1:2.2f}dB, SSIM = {2:1.4f}'.format(set_cur, psnr_avg, ssim_avg))

