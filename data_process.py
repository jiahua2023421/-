import glob   # 文件名操作模块glob
import cv2     # 读取图像首先要导入OpenCV包
import numpy as np
from torch.utils.data import Dataset   # torch.utils.data.Dataset（Dataset） 是一个表示数据集的抽象类
import torch
import os
from PIL import Image

# 参数
patch_size, stride = 40, 10  # 填充大小，步长
aug_times = 1   # 增强次数，每个图块生成几张增强图像
scales = [1, 0.9, 0.8, 0.7] #
batch_size = 128      # 批量大小

# 图片加噪声
class DenoisingDataset(Dataset):
    """Dataset wrapping tensors.
    Arguments:
        xs (Tensor): clean image patches
        sigma: noise level, e.g., 25
    """
    def __init__(self, xs, sigma):
        super(DenoisingDataset, self).__init__()
        self.xs = xs
        self.sigma = sigma
        # 数据设置类的初始化xs (Tensor):清洁图像， sigma: 噪声级


    def __getitem__(self, index): # 用于得到批量数据内容
        batch_x = self.xs[index]
        noise = torch.randn(batch_x.size()).mul_(self.sigma/255.0) # 生成噪声，size定义形状，与对象相同
        batch_y = batch_x + noise   # 加噪声
        return batch_y, batch_x    # 返回批量batch_y, batch_x

    def __len__(self):
        return self.xs.size(0) # 返回xs.size(0)指batchsize批量大小的值

#保存图片
def save_result(result, path):
    path = path if path.find('.') != -1 else path + '.png'
    ext = os.path.splitext(path)[-1]  # 文件扩展名
    if ext in ('.txt', '.dlm'):
        np.savetxt(path, result, fmt='%2.4f')  # 保存为txt文件，数据按%2.4f格式写入
    else:
        # 标准化
        result = np.clip(result, 0, 1)
        outputImg = Image.fromarray(result * 255.0)
        # "L"代表将图片转化为灰度图
        outputImg = outputImg.convert('L')
        outputImg.save(path)

#显示图片
def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')# #interpolation 插值方法  #cmap: 颜色图谱（colormap), 默认绘制为RGB(A)颜色空间
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show() # 输出图片

#旋转图片
def data_aug(img, mode=0):
    # data augmentation
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img) # 上下翻转张量
    elif mode == 2:
        return np.rot90(img) # 进行90度的旋转，轴1到轴2的方向
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2) # K,以90度旋转的次数
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

# get multiscale patches from a single image从单个图像中获取多尺度补丁
def gen_patches(file_name):
    # get multiscale patches from a single image
    img = cv2.imread(file_name, 0)  # gray scale flag=0
    h, w = img.shape
    patches = []
    for s in scales:
        h_scaled, w_scaled = int(h*s), int(w*s)
        img_scaled = cv2.resize(img, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)# 以s的比例缩放图像得到img_scaled
        # extract patches
        for i in range(0, h_scaled-patch_size+1, stride):
            for j in range(0, w_scaled-patch_size+1, stride):
                x = img_scaled[i:i+patch_size, j:j+patch_size]#从上到下，从左到右，按步长以补丁大小截取图像块
                for k in range(0, aug_times):# 这里表示操作一次aug_times=1
                    x_aug = data_aug(x, mode=np.random.randint(0, 8))# 对x进行翻转，角度随机
                    patches.append(x_aug)# 将得到的x_aug加到patches中
    return patches

#图片归一化
def datagenerator(data_dir='data/Train400', verbose=False):
    # generate clean patches from a dataset
    file_list = glob.glob(data_dir+'/*.png')  # get name list of all .png files
    # initrialize
    data = []
    # generate patches
    for i in range(len(file_list)):
        patches = gen_patches(file_list[i])
        for patch in patches:
            data.append(patch)# 按补丁顺序向数据中加补丁
        if verbose:
            print(str(i+1) + '/' + str(len(file_list)) + ' is done ^_^')

    data = np.array(data, dtype='uint8')
    data = np.expand_dims(data, axis=3)
    tes=len(data)
    discard_n = len(data)-len(data)//batch_size*batch_size  # because of batch namalization，
    data = np.delete(data, range(discard_n), axis=0)
    print('^_^-training data finished-^_^')
    return data