# 测试模型
# 命令行选项、参数和子命令的解析器
import io
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
import time, argparse, os, datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from imageio import imread
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
# 该Image模块提供了一个同名的类，用于表示 PIL 图像。该模块还提供了许多工厂功能，包括从文件加载图像和创建新图像的功能
import PIL.Image as pil_image
from model import REDNet10, REDNet20, REDNet30

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


if __name__ == '__main__':
    path = "C:/Users/Jerry/AppData/Roaming/SPB_16.6/REDNet-pytorch"
    # 容器参数规范，并具有将解析器作为一个整体应用的选项
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default='data', type=str, help='directory of test dataset')
    parser.add_argument('--arch', type=str, default='REDNet10', help='REDNet10, REDNet20, REDNet30')
    parser.add_argument('--weights_path', type=str, default=path + "/model/REDNet10_epoch_19.pth")
    parser.add_argument('--image_path', type=str, default=path + "/img_original")
    parser.add_argument('--outputs_dir', type=str, default=path + "/img_output")
    parser.add_argument('--jpeg_quality', type=int, default=10)
    parser.add_argument('--set_names', default=['Set68', 'Set12'], help='directory of test dataset')

    parser.add_argument('--patch_size', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--use_fast_loader', action='store_true')
    parser.add_argument('--sigma', default=25, type=int, help='noise level')
    opt = parser.parse_args()
    # 结果保存路径
    if not os.path.exists(opt.outputs_dir):
        os.makedirs(opt.outputs_dir)

    # 选择模型
    if opt.arch == 'REDNet10':
        model = REDNet10()
    elif opt.arch == 'REDNet20':
        model = REDNet20()
    elif opt.arch == 'REDNet30':
        model = REDNet30()

    # 一个 state_dict 只是一个 Python 字典对象，将每个层映射到其参数张量
    state_dict = model.state_dict()
    for n, p in torch.load(opt.weights_path, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            # 将模型的权重复制到另一个模型
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    # 加载模型
    model = model.to(device)
    model.eval()

    psnrs = []
    ssims = []

    for im in os.listdir(opt.image_path):  # 返回指定的文件夹包含的文件或文件夹的名字的列表
        if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):
            x = np.array(imread(os.path.join(opt.image_path, im)), dtype=np.float32) / 255.0
            #  dtype:数组中的数据类型
            np.random.seed(seed=0)  # for reproducibility
            y = x + np.random.normal(0, 25 / 255.0, x.shape)  # Add Gaussian noise without clipping
            y = y.astype(np.float32)
            y_ = torch.from_numpy(y).view(1, -1, y.shape[0], y.shape[1])

            torch.cuda.synchronize()
            start_time = time.time()
            #               y_ = y_.cuda()
            with torch.no_grad():
                x_ = model(y_)
            x_ = x_.view(y.shape[0], y.shape[1])
            x_ = x_.cpu()
            x_ = x_.detach().numpy().astype(np.float32)
            torch.cuda.synchronize()
            elapsed_time = time.time() - start_time
            print('%10s : %2.4f second' % (im, elapsed_time))

            psnr_x_ = compare_psnr(x, x_)
            ssim_x_ = compare_ssim(x, x_)
            # if args.save_result:
            name, ext = os.path.splitext(im)
            # show(np.hstack((y, x_)))  # show the image

            psnrs.append(psnr_x_)
            ssims.append(ssim_x_)
    psnr_avg = np.mean(psnrs)
    ssim_avg = np.mean(ssims)
    psnrs.append(psnr_avg)
    ssims.append(ssim_avg)
    # if args.save_result:
    log('PSNR = {1:2.2f}dB, SSIM = {2:1.4f}'.format(psnr_avg, ssim_avg))