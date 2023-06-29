# 测试模型
# 命令行选项、参数和子命令的解析器
import argparse
import os
import io
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
# 该Image模块提供了一个同名的类，用于表示 PIL 图像。该模块还提供了许多工厂功能，包括从文件加载图像和创建新图像的功能
import PIL.Image as pil_image
from model import REDNet10, REDNet20, REDNet30

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    path = "C:/Users/Jerry/AppData/Roaming/SPB_16.6/REDNet-pytorch"
    # 容器参数规范，并具有将解析器作为一个整体应用的选项
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='REDNet10', help='REDNet10, REDNet20, REDNet30')
    parser.add_argument('--weights_path', type=str, default = path+"/model/REDNet10_epoch_19.pth")
    parser.add_argument('--image_path', type=str, default= path + "/img_original/test001.png")
    parser.add_argument('--outputs_dir', type=str, default= path + "/img_output")
    parser.add_argument('--jpeg_quality', type=int, default=10)
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

    filename = os.path.basename(opt.image_path).split('.')[0]

    # 输入图片
    input = pil_image.open(opt.image_path).resize((768, 512)).convert('RGB')
    # 内存中二进制流也可用作 BytesIO 对象
    buffer = io.BytesIO()
    input.save(buffer, format='jpeg', quality=opt.jpeg_quality)
    input = pil_image.open(buffer)
    input.save(os.path.join(opt.outputs_dir, '{}_jpeg_q{}.png'.format(filename, opt.jpeg_quality)))
    # 将 PIL 图像或 ndarray 转换为张量并相应地缩放值
    input = transforms.ToTensor()(input).unsqueeze(0).to(device)

    # 不需要计算梯度，也不会进行反向传播
    with torch.no_grad():
        pred = model(input)

    # 矩阵相乘
    pred = pred.mul_(255.0).clamp_(0.0, 255.0).squeeze(0).permute(1, 2, 0).byte().cpu().numpy()

    # 实现array到image的转换
    output = pil_image.fromarray(pred, mode='RGB')
    # 保存
    output.save(os.path.join(opt.outputs_dir, '{}_{}.png'.format(filename, opt.arch)))
