from __future__ import print_function
import os
import time
from model import Deam
import socket
import pandas as pd
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from real_dataloader import *
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--batch_size', type=int, default=2, help='training batch size')  #


parser.add_argument('--nEpochs', type=int, default=1, help='number of epochs to train for')  #
parser.add_argument('--start_iter', type=int, default=1, help='starting epoch')  #
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate. default=0.0001')  #


parser.add_argument('--Ispretrained', type=bool, default=False, help='If load checkpoint model')  #
parser.add_argument('--pretrained_sr', default='Real.pth', help='sr pretrained base model')  #
parser.add_argument('--pretrained', default='./Deam_models', help='Location to load checkpoint models')  #
parser.add_argument('--save_folder', default='./real_model/', help='Location to save checkpoint models')  #
parser.add_argument('--statistics', default='./statistics/', help='Location to save statistics')  #

# Testing settings
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size, default=1')  #

parser.add_argument('--gpus', default=1, type=int, help='number of gpus')  #
parser.add_argument('--data_dir', type=str, default='./data', help='the dataset dir')  #
parser.add_argument('--model_type', type=str, default='Deam', help='the name of model')  #
parser.add_argument('--patch_size', type=int, default=128, help='Size of cropped HR image')  #
parser.add_argument('--Isreal', default=True, help='If training/testing on RGB images')  #


opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
cudnn.benchmark = True
print(opt)





def train(epoch):
    epoch_loss = 0
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        # target = Variable(batch[1])  # 0 噪声   1 干净
        # input = Variable(batch[0])  # 输入为噪声图片
        # zaosheng = Variable(batch[0]) - Variable(batch[1])  # 噪声 - 干净
        # input = input.cuda()
        # target = target.cuda()
        # model.zero_grad()
        # optimizer.zero_grad()
        # t0 = time.time()
        #
        # prediction = model(input)
        #
        # fankui = input - prediction  # 噪声 - 预测
        #
        # # Corresponds to the Optimized Scheme
        # loss = criterion(fankui, zaosheng.cuda())/(input.size()[0]*2)
        target = Variable(batch[1])  # 0 噪声   1 干净
        input = Variable(batch[0])  # 输入为噪声图片
        zaosheng = Variable(batch[0]) - Variable(batch[1])  # 噪声 - 干净
        input = input.cuda()
        target = target.cuda()
        model.zero_grad()
        optimizer.zero_grad()
        t0 = time.time()

        prediction = model(input)  # 输出噪声

        # fankui = input - prediction  # 噪声 - 预测

        # Corresponds to the Optimized Scheme
        loss = criterion(prediction, zaosheng.cuda())/(input.size()[0]*2)
        # target = Variable(batch[1])  # 0 噪声   1 干净
        # input = Variable(batch[0])  # 输入为噪声图片
        # input = input.cuda()
        # target = target.cuda()
        # model.zero_grad()
        # optimizer.zero_grad()
        # t0 = time.time()
        #
        # prediction = model(input)
        #
        # # Corresponds to the Optimized Scheme
        # loss = criterion(prediction, target) / (input.size()[0] * 2)
        t1 = time.time()
        epoch_loss += loss.data
        loss.backward()
        optimizer.step()

        if (iteration+1) % 50 == 0:
            model.eval()
            model.train()
        print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration, len(training_data_loader), loss.data, (t1 - t0)))
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
    if not os.path.exists(opt.save_folder):  # 结果路径
        os.mkdir(opt.save_folder)  # 创造目录
    model.eval()
    # SC = 'net_epoch_' + str(epoch) + '_' + str(iteration + 1) + '.pth'
    # torch.save(model.state_dict(), os.path.join(opt.save_folder, SC))
    # model.train()
    torch.save(model.state_dict(), os.path.join(opt.save_folder, 'Real6.pth'))  # 保存模型

def tensor_ndarray(data):
    data1 = data.data.cpu().numpy().astype(np.float32)
    data2 = data1[0, :, :, :]
    data3 = data2[:, :, ::-1].transpose((1, 2, 0))
    return data3


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

def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x)  # #interpolation 插值方法  #cmap: 颜色图谱（colormap), 默认绘制为RGB(A)颜色空间
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()  # 输出图片



def test(testing_data_loader):
    psnr_test = 0
    ssim_test = 0
    model.eval()
    for batch in testing_data_loader:
        target = Variable(batch[1])  # 1干净图片  0噪声图片
        input = Variable(batch[0])
        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
            prediction = model(input)
            prediction = torch.clamp(prediction, 0., 1.)
        p = tensor_ndarray(prediction)
        i = tensor_ndarray(input)
        t = tensor_ndarray(target)
        # show(np.hstack((p, i, t)))
        psnr, ssim = batch_PSNR(prediction, target, 1.)
        psnr_test += psnr
        ssim_test += ssim
    print("===> Avg. PSNR: {:.4f} dB".format(psnr_test / len(testing_data_loader)))
    print("===> Avg. PSNR: {:.4f} dB".format(ssim_test / len(testing_data_loader)))
    return psnr_test / len(testing_data_loader)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def checkpoint(epoch,psnr):
    model_out_path = opt.save_folder+hostname+opt.model_type+"_psnr_{}".format(psnr)+"_epoch_{}.pth".format(epoch)
    # torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == '__main__':
    print('===> Loading datasets')


    train_set = Dataset_h5_real(src_path=os.path.join(opt.data_dir, 'train', 'train.h5'), patch_size=opt.patch_size, train=True)
    training_data_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size, shuffle=True, num_workers=4,
                                drop_last=True)
    test_set = Dataset_h5_real(src_path=os.path.join(opt.data_dir, 'test', 'val.h5'), patch_size=opt.patch_size, train=False)
    testing_data_loader = DataLoader(dataset=test_set, batch_size=opt.testBatchSize, shuffle=False, num_workers=0, drop_last=True)
    print('ceshi')


    print('===> Building model ', opt.model_type)
    model = Deam(opt.Isreal)

    model = torch.nn.DataParallel(model, device_ids=gpus_list)
    criterion = nn.MSELoss()

    print('---------- Networks architecture -------------')
    print_network(model)
    print('----------------------------------------------')

    if opt.Ispretrained:
        model_name = os.path.join(opt.pretrained, opt.pretrained_sr)
        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        print(model_name + ' model is loaded.')

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

    PSNR = []
    for epoch in range(opt.start_iter, opt.nEpochs + 1):
        train(epoch)
        psnr = test(testing_data_loader)
        PSNR.append(psnr)
        data_frame = pd.DataFrame(
            data={'epoch': epoch, 'PSNR': PSNR}, index=range(1, epoch+1)
        )
        data_frame.to_csv(os.path.join(opt.statistics, 'training_logs.csv'), index_label='index')
        # learning rate is decayed by a factor of 10 every half of total epochs
        if (epoch + 1) % (opt.nEpochs / 2) == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10.0
            # print('Learning rate decay: lr={}'.param_group['lr'])