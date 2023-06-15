from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import torch

import numpy as np
import cv2
from imageio import imread
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from model import DnCNN
import matplotlib.pyplot as plt
from model import Deam
from data_process import show, save_result

# 为使模型可用，能用GPU时将FALSE改为True,部分注释（checkpoint\cuda等）需要回复
use_gpu = False
# load the pretrained model
print('Loading the Model')
# args = parse_benchmark_processing_arguments()
# checkpoint = torch.load(os.path.join('./Deam_models/', 'Real.pth'))
net = Deam(True)
if use_gpu:
    net = torch.nn.DataParallel(net).cuda()
#    net.load_state_dict(checkpoint)
net.eval()


def denoise(model, noisy_image):
    with torch.autograd.set_grad_enabled(False):
        #     torch.cuda.synchronize()
        # torch.synchronize()
        phi_Z = model(noisy_image)
        # Img = noisy_image.data.cpu().numpy().astype(np.float32)
        # Iclean = phi_Z.data.cpu().numpy().astype(np.float32)
        Img = noisy_image.data.numpy().astype(np.float32)
        Iclean = phi_Z.data.numpy().astype(np.float32)
        ceshi = Iclean[0, :, :, :]
        ceshi2 = Img[0, :, :, :]
        Iclean2 = ceshi[:, :, ::-1].transpose((1, 2, 0))
        Img2 = ceshi2[:, :, ::-1].transpose((1, 2, 0))
        show(np.hstack((Iclean2, Img2)))  # 显示图片
        #   torch.cuda.synchronize()
        im_denoise = phi_Z.numpy()  # ndarray 1 3 256 256
    im_denoise = np.transpose(im_denoise.squeeze(), (1, 2, 0))  # ndarray 256 256 3
    #  im_denoise = img_as_ubyte(im_denoise.clip(0, 1))

    return phi_Z


def tensor_ndarray(data):
    data1 = data.data.cpu().numpy().astype(np.float32)
    data2 = data1[0, :, :, :]
    data3 = data2[:, :, ::-1].transpose((1, 2, 0))
    return data3


class App:
    def __init__(self, master):
        self.master = master
        master.title("图片处理应用")
        self.select_image_button = Button(master, text="选择图片", command=self.select_image)
        self.select_image_button.pack()
        self.image_label = Label(master)
        self.image_label.pack()
        self.save_image_button = Button(master, text="保存图片", command=self.save_image)
        self.save_image_button.pack()

    def select_image(self):
        file_path = filedialog.askopenfilename()
        if file_path.endswith(".jpg") or file_path.endswith(".bmp") or file_path.endswith(".png"):
            # self.image = Image.open(file_path)
            # self.photo = ImageTk.PhotoImage(self.image)
            # self.image_label.config(image=self.photo)
            noisy_image = Image.open(file_path)
            noisy_image2 = noisy_image.resize((256, 256))
            noisy_image2 = noisy_image2.convert("RGB")

            self.image = noisy_image2
            self.photo = ImageTk.PhotoImage(self.image)
            self.image_label.config(image=self.photo)

            noisy_image2.save('C:/Users/63086/Desktop/14.png', 'png')
            noisy_image = plt.imread('C:/Users/63086/Desktop/14.png')
            noisy_image = torch.from_numpy(noisy_image.transpose((2, 0, 1))[np.newaxis,])
            poseSmile_cell = denoise(net, noisy_image)
            p = tensor_ndarray(poseSmile_cell)
            n = tensor_ndarray(noisy_image)
            show(np.hstack((p, n)))  # 显示图片结果

    def save_image(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg")
        if file_path:
            self.image.save(file_path)


root = Tk()
app = App(root)
root.mainloop()
