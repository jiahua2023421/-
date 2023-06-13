import os
import math
import SIDD_denoise
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from skimage import img_as_ubyte
import torch
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import matplotlib.pyplot as plt
from model import Deam
import os
import argparse
import time
import numpy as np
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', type=str, default='./Deam_models/', help="Checkpoints directory,  (default:./checkpoints)")
parser.add_argument('--Isreal', default=True, help='Location to save checkpoint models')
parser.add_argument('--data_folder', type=str, default='./data/Benchmark_test', help='Location to save checkpoint models')
parser.add_argument('--out_folder', type=str, default='./Dnd_result', help='Location to save checkpoint models')
parser.add_argument('--model', type=str, default='Real.pth', help='Location to save checkpoint models')
parser.add_argument('--Type', type=str, default='SIDD', help='To choose the testing benchmark dataset, SIDD or Dnd')
parser.add_argument('--result_dir', default='deam_results/real', type=str, help='directory of test dataset')  # 测试结果目录
args = parser.parse_args()
use_gpu = True

print('Loading the Model')
net = Deam(args.Isreal)
checkpoint = torch.load(os.path.join(args.pretrained, args.model))
model = torch.nn.DataParallel(net).cuda()
# model = torch.nn.DataParallel(net)
model.eval()

def normalize(data):
    return data/255.

def qiepian():
    import cv2
    img = cv2.imread("./data/cut/NOISY_SRGB_010.png")
    print(img.shape)
    ii = 0
    for i in range(0, math.floor((img.shape[0]) / 256)):
        for j in range(0, math.floor((img.shape[1]) / 256)):
            cropped = img[0+i*256:256+i*256, 0+j*256:256+j*256]  # 裁剪坐标为[y0:y1, x0:x1]
            lujing = './data/cut/noise/thor' + str(ii) + '.png'
            cv2.imwrite(lujing, cropped)
            ii += 1
def denoise(model, noisy_image, list):
    with torch.autograd.set_grad_enabled(False):
        torch.cuda.synchronize()
        # torch.synchronize()
        phi_Z = model(noisy_image)
        # psnr_test = batch_PSNR(phi_Z, noisy_image, 1.)  #计算两张图片的PSNR
        # print("===> Avg. PSNR: {:.4f} dB".format(psnr_test))
        # Img = noisy_image.data.cpu().numpy().astype(np.float32)
        # Iclean = phi_Z.data.cpu().numpy().astype(np.float32)
        Img = noisy_image.data.cpu().numpy().astype(np.float32)
        Iclean = phi_Z.data.cpu().numpy().astype(np.float32)
        ceshi = Iclean[0, :, :, :]
        ceshi2 = Img[0, :, :, :]
        Iclean2 = ceshi[:, :, ::-1].transpose((2, 1, 0))
        Img2 = ceshi2[:, :, ::-1].transpose((2, 1, 0))
        # show(np.hstack((Iclean2, Img2)))  # 显示图片
        save_result(Iclean2, './real_result/denoise', list)
        torch.cuda.synchronize()
        im_denoise = phi_Z.cpu().numpy()   # ndarray 1 3 256 256
    im_denoise = np.transpose(im_denoise.squeeze(), (1, 2, 0))  # ndarray 256 256 3
    im_denoise = img_as_ubyte(im_denoise.clip(0, 1))

    return phi_Z

def batch_SSIM(img, imclean, data_range):    # 1，3，256，256 tensor
    Img = img.data.cpu().numpy().astype(np.float32)  # ndarray 1 3 256 256
    Iclean = imclean.data.cpu().numpy().astype(np.float32)  # ndarray 1 3 256 256
    PSNR = 0
    SSIM1 = 0
    SSIM2 = 0
    SSIM3 = 0
    SSIM = 0
    ceshi = Img.shape[0]
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)  # 3 256 256 ndarray
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
    return SSIM / Img.shape[0]
def batch_PSNR(img, imclean, data_range):    # 1，3，256，256 tensor
    Img = img.data.cpu().numpy().astype(np.float32)  # ndarray 1 3 256 256
    Iclean = imclean.data.cpu().numpy().astype(np.float32)  # ndarray 1 3 256 256
    PSNR = 0
    SSIM1 = 0
    SSIM2 = 0
    SSIM3 = 0
    SSIM = 0
    ceshi = Img.shape[0]
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)  # 3 256 256 ndarray
        ceshi = Iclean[i, :, :, :]
        ceshi2 = Img[i, :, :, :]
        # ceshi3 = Iclean[:, :]

        Iclean2 = ceshi[:, :, ::-1].transpose((2, 1, 0))
        Img2 = ceshi2[:, :, ::-1].transpose((2, 1, 0))
        # img2 = np.resize(img2, (img1.shape[0], img1.shape[1], img1.shape[2]))
        ce = ceshi[0]
        ce2 = ceshi2[0]
        # show(np.hstack((Iclean2, Img2)))  # 去噪图片，噪声图片
        SSIM1 += compare_ssim(ceshi[0], ceshi2[0], data_range=data_range)
        SSIM2 += compare_ssim(ceshi[1], ceshi2[1], data_range=data_range)
        SSIM3 += compare_ssim(ceshi[2], ceshi2[2], data_range=data_range)
        SSIM += ((SSIM1 + SSIM2 + SSIM3) / 3)
        # print('ce')
    return PSNR / Img.shape[0]
def save_result(result, path, xuhao):  # 数组三通道图片，路径， 图片序号
    path = path if path.find('.') != -1 else path + '.png'
    result = np.clip(result, 0, 1)  # 标准化
    ceshi = path + '/real' + str(xuhao) + '.png'
    plt.imsave(ceshi, result)  # result应为0到1之间的数
def main():
    use_gpu = True
    # load the pretrained model
    print('Loading the Model')
    # args = parse_benchmark_processing_arguments()
    # checkpoint = torch.load(os.path.join(args.pretrained, args.model))
    net = Deam(args.Isreal)
    if use_gpu:
        net = torch.nn.DataParallel(net).cuda()
        net.load_state_dict(checkpoint)
    net.eval()

    files_path = './data/cut/noise'
    test_path = './data/cut/result'
    files_source = os.listdir(files_path)
    test_source = os.listdir(test_path)
    psnr_test = 0
    i = 1
    psnrs = []  # 计算psnr与ssim的数组
    ssims = []
    i2 = 0
    PSNR = 0
    SSIM = 0
    for f in files_source:
            # if not os.path.exists(os.path.join(opt.result_dir)):  # 未找到保存文件的路径，则创造路径
            #     os.mkdir(os.path.join(opt.result_dir))
            # public.path_creat(opt.result_dir)
        g = test_source[i2]
        SEED = 0
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        start_time = time.time()
        image_path = os.path.join(files_path, f)
        clean_path = os.path.join(test_path, g)
            # image
        # Img = plt.imread(image_path)
        noisy_image = plt.imread(image_path)
        clean_image = plt.imread(clean_path)
        # noisy_image = np.float32(Img / 255.)
        noisy_image = torch.from_numpy(noisy_image.transpose((2, 0, 1))[np.newaxis, ])
        clean_image = torch.from_numpy(clean_image.transpose((2, 0, 1))[np.newaxis, ])
        poseSmile_cell = denoise(net, noisy_image, i2)
        # poseSmile_cell = normalize(poseSmile_cell)


        psnrs.append(batch_PSNR(clean_image, poseSmile_cell, data_range=1.0))
        ssims.append(batch_SSIM(clean_image, poseSmile_cell, data_range=1.0))
        ceshi = psnrs[i2]
        PSNR += ceshi
        ceshi2 = ssims[i2]
        SSIM += ceshi2
        print("{:.4f}".format(ceshi))
        print("{:.4f}".format(ceshi2))
        i2 += 1
    print("平均PSNR{:.4f}".format(PSNR/len(files_source)))
    print("平均SSIM{:.4f}".format(SSIM / len(files_source)))


if __name__ == "__main__":
    main()
    SIDD_denoise.test(args)
    # qiepian()