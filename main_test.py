# -*- coding: utf-8 -*-

# =============================================================================
#  @article{zhang2017beyond,
#    title={Beyond a {Gaussian} denoiser: Residual learning of deep {CNN} for image denoising},
#    author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
#    journal={IEEE Transactions on Image Processing},
#    year={2017},
#    volume={26}, 
#    number={7}, 
#    pages={3142-3155}, 
#  }
# by Kai Zhang (08/2018)
# cskaizhang@gmail.com
# https://github.com/cszn
# modified on the code from https://github.com/SaoYan/DnCNN-PyTorch
# =============================================================================

# run this to test the model
from PIL import Image
import argparse
import cv2
import os, time, datetime
# import PIL.Image as Image
import numpy as np
import torch.nn as nn
from imageio import imread, imwrite
import torch.nn.init as init
import torch
# from skimage.measure import compare_psnr, compare_ssim
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

# from skimage.io import imread, imsave
# from PIL import Image
import matplotlib.pyplot as plt
import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default='data/Test', type=str, help='directory of test dataset')
    parser.add_argument('--set_names', default=['Set68', 'Set12'], help='directory of test dataset')
    parser.add_argument('--sigma', default=25, type=int, help='noise level')
    parser.add_argument('--model_dir', default=os.path.join('models', 'DnCNN_sigma25'), help='directory of the model')
    # os.path.join  路径拼接
    parser.add_argument('--model_name', default='model_001.pth', type=str, help='the model name')
    parser.add_argument('--result_dir', default='results', type=str, help='directory of test dataset')
    parser.add_argument('--save_result', default=0, type=int, help='save the denoised image, 1 or 0')
    return parser.parse_args()


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)
    # datetime.datetime.now().strftime：格式化时间：年份-月份-天


def save_result(result, path):
    from imageio import imread, imwrite
    from skimage import io
    path = path if path.find('.') != -1 else path + '.png'
    ext = os.path.splitext(path)[-1]  # 文件扩展名
    if ext in ('.txt', '.dlm'):
        np.savetxt(path, result, fmt='%2.4f')  # 保存为txt文件，数据按%2.4f格式写入
    else:
        # result1 = result.astype(np.uint8)
        # result2 = Image.fromarray(result)
        # result2.show()
        # imsave(path, result)
        # cv2.imwrite(path, result)
        # io.imsave(path, result)
        # result.save(path)
        # show(result, path)
        # plt.figure(figsize=figsize)  # 创造一个图形
        # plt.imshow(result, interpolation='nearest', cmap='gray')
        # plt.savefig(path
        result = np.clip(result, 0, 1)
        outputImg = Image.fromarray(result * 255.0)
        # "L"代表将图片转化为灰度图
        outputImg = outputImg.convert('L')
        outputImg.save(path)
        # outputImg.show()
        # imwrite(path, )
        # imshow(result)
        # imsave(path, np.clip(result, 0, 1))

        # 保存图片  路径名称，数组变量  np.clip：result被限制在0~1之间


def show(x, path=None, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    # plt.figure(figsize=figsize)  # 创造一个图形
    plt.imshow(x, interpolation='nearest', cmap='gray')
    # plt.show()
    # interpolation='nearest'如果显示分辨率与图像分辨率不同(通常是这种情况)，则简单地显示图像而无需尝试在像素之间插值.它将导致图像将像素显示为多个像素的平方
    if title:
        plt.title(title)  # 设置图像标题
    if cbar:
        plt.colorbar()
    plt.show()
    # plt.savefig(path, bbox_inches ="tight")

class DnCNN(nn.Module):

    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []
        layers.append(
            nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            layers.append(
                nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding,
                      bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y - out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


if __name__ == '__main__':

    args = parse_args()

    # model = DnCNN()
    if not os.path.exists(os.path.join(args.model_dir, args.model_name)):  # model_001若不存在，则加载model

        model = torch.load(os.path.join(args.model_dir, 'model.pth'), map_location='cpu')  # 映射到CPU
        # load weights into new model
        log('load trained model on Train400 dataset by kai')
    else:
        # model.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_name)))
        model = torch.load(os.path.join(args.model_dir, args.model_name), map_location='cpu')
        log('load trained model')

    #    params = model.state_dict()
    #    print(params.values())
    #    print(params.keys())
    #
    #    for key, value in params.items():
    #        print(key)    # parameter name
    #    print(params['dncnn.12.running_mean'])
    #    print(model.state_dict())

    model.eval()  # evaluation mode  设置为预测模式
    #    model.train()

    #    if torch.cuda.is_available():  #GPU
    #        model = model.cuda()

    if not os.path.exists(args.result_dir):  # 结果路径
        os.mkdir(args.result_dir)  # 创造目录

    for set_cur in args.set_names:

        if not os.path.exists(os.path.join(args.result_dir, set_cur)):
            os.mkdir(os.path.join(args.result_dir, set_cur))
        psnrs = []
        ssims = []

        for im in os.listdir(os.path.join(args.set_dir, set_cur)):  # 返回指定的文件夹包含的文件或文件夹的名字的列表
            if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):
                x = np.array(imread(os.path.join(args.set_dir, set_cur, im)), dtype=np.float32) / 255.0
                #  dtype:数组中的数据类型
                np.random.seed(seed=0)  # for reproducibility
                y = x + np.random.normal(0, args.sigma / 255.0, x.shape)  # Add Gaussian noise without clipping
                y = y.astype(np.float32)
                y_ = torch.from_numpy(y).view(1, -1, y.shape[0], y.shape[1])

                torch.cuda.synchronize()
                start_time = time.time()
                #               y_ = y_.cuda()
                x_ = model(y_)  # inference
                x_ = x_.view(y.shape[0], y.shape[1])
                x_ = x_.cpu()
                x_ = x_.detach().numpy().astype(np.float32)
                torch.cuda.synchronize()
                elapsed_time = time.time() - start_time
                print('%10s : %10s : %2.4f second' % (set_cur, im, elapsed_time))

                psnr_x_ = compare_psnr(x, x_)
                ssim_x_ = compare_ssim(x, x_)
                # if args.save_result:
                name, ext = os.path.splitext(im)
                show(np.hstack((y, x_)))  # show the image
                save_result(x_, path=os.path.join(args.result_dir, set_cur,
                                                  name + '_dncnn' + ext))  # save the denoised image
                psnrs.append(psnr_x_)
                ssims.append(ssim_x_)
        psnr_avg = np.mean(psnrs)
        ssim_avg = np.mean(ssims)
        psnrs.append(psnr_avg)
        ssims.append(ssim_avg)
        # if args.save_result:
        save_result(np.hstack((psnrs, ssims)), path=os.path.join(args.result_dir, set_cur, 'results.txt'))
        log('Datset: {0:10s} \n  PSNR = {1:2.2f}dB, SSIM = {2:1.4f}'.format(set_cur, psnr_avg, ssim_avg))