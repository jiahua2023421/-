import os
import math
import SIDD_denoise
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import matplotlib.pyplot as plt
from model import Deam
import os
import argparse
import numpy as np
import public
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
parser = argparse.ArgumentParser()
# parser.add_argument('--pretrained', type=str, default='./Deam_models/', help="Checkpoints directory,  (default:./checkpoints)")  #
parser.add_argument('--pretrained', type=str, default='./real_model/', help="Checkpoints directory,  (default:./checkpoints)")  #
parser.add_argument('--model', type=str, default='Real6.pth', help='Location to save checkpoint models')  #
# parser.add_argument('--model', type=str, default='model_001.pth', help='Location to save checkpoint models')  #
parser.add_argument('--result_dir', default='deam_results/real', type=str, help='directory of test dataset')  # 测试结果目录
args = parser.parse_args()
use_gpu = True
print('Loading the Model')
net = Deam(True)
checkpoint = torch.load(os.path.join(args.pretrained, args.model))
model = torch.nn.DataParallel(net).cuda()
model.eval()
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
def denoise(model, noisy_image):
    with torch.autograd.set_grad_enabled(False):
        torch.cuda.synchronize()
        phi_Z = model(noisy_image)
        phi_Z = noisy_image.cuda() - phi_Z
    return phi_Z


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    SSIM1 = 0
    SSIM2 = 0
    SSIM3 = 0
    SSIM = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
        ceshi = Iclean[i, :, :, :]
        ceshi2 = Img[i, :, :, :]
        SSIM1 += compare_ssim(ceshi[0], ceshi2[0], data_range=data_range)
        SSIM2 += compare_ssim(ceshi[1], ceshi2[1], data_range=data_range)
        SSIM3 += compare_ssim(ceshi[2], ceshi2[2], data_range=data_range)
        SSIM += ((SSIM1 + SSIM2 + SSIM3)/3)
    return PSNR / Img.shape[0], SSIM / Img.shape[0]
def save_result(result, path, xuhao):  # 数组三通道图片，路径， 图片序号
    path = path if path.find('.') != -1 else path + '.png'
    result = np.clip(result, 0, 1)  # 标准化
    ceshi = path + '/real' + str(xuhao) + '.png'
    plt.imsave(ceshi, result)  # result应为0到1之间的数

def tensor_ndarray(data):
    data1 = data.data.cpu().numpy().astype(np.float32)
    data2 = data1[0, :, :, :]
    data3 = data2[:, :, ::-1].transpose((1, 2, 0))
    return data3
def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x)  # #interpolation 插值方法  #cmap: 颜色图谱（colormap), 默认绘制为RGB(A)颜色空间
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()  # 输出图片
def main():
    use_gpu = True
    print('Loading the Model')
    net = Deam(True)
    if use_gpu:
        net = torch.nn.DataParallel(net).cuda()
        net.load_state_dict(checkpoint)
    net.eval()
    files_path = './data/cut/noise'  # 噪声图片路径
    test_path = './data/cut/result'  # 无噪声图，计算SSIM和PSNR
    files_source = os.listdir(files_path)
    test_source = os.listdir(test_path)
    psnrs = []  # 计算psnr与ssim的数组
    ssims = []
    i2 = 0
    PSNR = 0
    SSIM = 0
    for f in files_source:
        g = test_source[i2]
        SEED = 0
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        image_path = os.path.join(files_path, f)
        clean_path = os.path.join(test_path, g)
        noisy_image = plt.imread(image_path)  # 读彩色图片， 256 256 3 数组
        clean_image = plt.imread(clean_path)
        noisy_image = torch.from_numpy(noisy_image.transpose((2, 0, 1))[np.newaxis, ])  # 转为张量 1 3 256 256
        clean_image = torch.from_numpy(clean_image.transpose((2, 0, 1))[np.newaxis, ])
        poseSmile_cell = denoise(net, noisy_image)  # 返回张量 1 3 256 256
        c = tensor_ndarray(clean_image)  # 张 量 转数组
        p = tensor_ndarray(poseSmile_cell)
        n = tensor_ndarray(noisy_image)
        # show(np.hstack((c, p, n)))  # 显示图片结果
        psnr, ssim = batch_PSNR(clean_image, poseSmile_cell, 1.)  # 计算单个图片，输入为张量
        psnrs.append(psnr)
        ssims.append(ssim)
        ceshi = psnrs[i2]
        PSNR += ceshi
        ceshi2 = ssims[i2]
        SSIM += ceshi2
        print("{:.4f}".format(ceshi))
        print("{:.4f}".format(ceshi2))
        i2 += 1
    print("平均PSNR{:.4f}".format(PSNR/len(files_source)))
    print("平均SSIM{:.4f}".format(SSIM / len(files_source)))
    psnrs.append(PSNR/len(files_source))
    ssims.append(SSIM / len(files_source))
    public.path_creat(args.result_dir)
    save_result1(np.hstack((psnrs, ssims)),
                path='deam_results/real/result6.txt')   # 保存平均值

def save_result1(result, path):
    np.savetxt(path, result, fmt='%2.4f')  # 保存为txt文件，数据按%2.4f格式写入

if __name__ == "__main__":
    main()
    # SIDD_denoise.test(args)
    # qiepian()