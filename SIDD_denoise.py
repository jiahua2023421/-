import os
import cv2
import numpy as np
from skimage import img_as_ubyte
import argparse
from model import Deam
from tqdm import tqdm
from scipy.io import loadmat, savemat
import torch
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import matplotlib.pyplot as plt


def show(x, title=None, cbar=False, figsize=None):
    plt.figure(figsize=figsize)
    plt.imshow(x)  # #interpolation 插值方法  #cmap: 颜色图谱（colormap), 默认绘制为RGB(A)颜色空间
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()  # 输出图片


def save_result(result, path, xuhao):  # 数组三通道图片，路径， 图片序号
    path = path if path.find('.') != -1 else path + '.png'
    result = np.clip(result, 0, 1)  # 标准化
    ceshi = path + '/real' + str(xuhao) + '.png'
    plt.imsave(ceshi, result)  # result应为0到1之间的数


def denoise(model, noisy_image, list):
    with torch.autograd.set_grad_enabled(False):
        torch.cuda.synchronize()
        phi_Z = model(noisy_image)
        # psnr_test = batch_PSNR(phi_Z, noisy_image, 1.)  #计算两张图片的PSNR
        # print("===> Avg. PSNR: {:.4f} dB".format(psnr_test))
        Img = noisy_image.data.cpu().numpy().astype(np.float32)
        Iclean = phi_Z.data.cpu().numpy().astype(np.float32)
        ceshi = Iclean[0, :, :, :]
        ceshi2 = Img[0, :, :, :]
        Iclean2 = ceshi[:, :, ::-1].transpose((2, 1, 0))
        Img2 = ceshi2[:, :, ::-1].transpose((2, 1, 0))
        # show(np.hstack((Iclean2, Img2)))  # 显示图片
        save_result(Iclean2, './real_result', list)
        torch.cuda.synchronize()
        im_denoise = phi_Z.cpu().numpy()
    im_denoise = np.transpose(im_denoise.squeeze(), (1, 2, 0))
    im_denoise = img_as_ubyte(im_denoise.clip(0, 1))

    return im_denoise


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    SSIM1 = 0
    SSIM2 = 0
    SSIM3 = 0
    SSIM = 0
    ceshi = Img.shape[0]
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
        ceshi = Iclean[i, :, :, :]
        ceshi2 = Img[i, :, :, :]
        # ceshi3 = Iclean[:, :]

        Iclean2 = ceshi[:, :, ::-1].transpose((2, 1, 0))
        Img2 = ceshi2[:, :, ::-1].transpose((2, 1, 0))
        # img2 = np.resize(img2, (img1.shape[0], img1.shape[1], img1.shape[2]))
        ce = ceshi[0]
        ce2 = ceshi2[0]
        # show(np.hstack((Iclean2, Img2)))
        SSIM1 += compare_ssim(ceshi[0], ceshi2[0], data_range=data_range)
        SSIM2 += compare_ssim(ceshi[1], ceshi2[1], data_range=data_range)
        SSIM3 += compare_ssim(ceshi[2], ceshi2[2], data_range=data_range)
        SSIM += ((SSIM1 + SSIM2 + SSIM3) / 3)
        # print('ce')
    return PSNR / Img.shape[0]


def test(args):
    use_gpu = True
    # load the pretrained model
    print('Loading the Model')
    # args = parse_benchmark_processing_arguments()
    checkpoint = torch.load(os.path.join(args.pretrained, args.model))
    net = Deam(args.Isreal)
    if use_gpu:
        net = torch.nn.DataParallel(net).cuda()
        net.load_state_dict(checkpoint)
    net.eval()

    # load SIDD benchmark dataset and information
    noisy_data_mat_file = os.path.join(args.data_folder, 'BenchmarkNoisyBlocksSrgb.mat')
    # noisy_data_mat_file = os.path.join(args.data_folder, '0009_NOISY_RAW_010.MAT')
    noisy_data_mat_name = os.path.basename(noisy_data_mat_file).replace('.mat', '')
    noisy_data_mat = loadmat(noisy_data_mat_file)[noisy_data_mat_name]  # 数组，40，32，256，256，3

    npose = (noisy_data_mat.shape[0])
    nsmile = noisy_data_mat.shape[1]
    poseSmile_cell = np.empty((npose, nsmile), dtype=object)
    psnr_test = 0
    i = 0
    for image_index in tqdm(range(noisy_data_mat.shape[0])):  # 40 *32 张图片
        for block_index in range(noisy_data_mat.shape[1]):
            noisy_image = noisy_data_mat[image_index, block_index, :, :, :]  # 每一张图片，256*256*3

            # noisy_image = noisy_data_mat[image_index, block_index, :, :]
            noisy_image = np.float32(noisy_image / 255.)
            # save_result(noisy_image, path='./data/Benchmark_test/png', xuhao=i)
            noisy_image = torch.from_numpy(noisy_image.transpose((2, 0, 1))[np.newaxis, ])
            poseSmile_cell[image_index, block_index] = denoise(net, noisy_image, i)
            i += 1
            # print('duandian')

    submit_data = {
        'DenoisedBlocksSrgb': poseSmile_cell
    }

    savemat(
        os.path.join(os.path.dirname(noisy_data_mat_file), 'SubmitSrgb.mat'),
        submit_data
    )
