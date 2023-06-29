# 训练模型
# 命令行选项、参数和子命令的解析器
import argparse
import os
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
# 显示智能进度表
from tqdm import tqdm
from model import REDNet10, REDNet20, REDNet30
from dataset import Dataset
from utils import AverageMeter

# 允许您启用内置的 cudnn 自动调谐器以找到用于您的硬件的最佳算法
cudnn.benchmark = True
# 指定设备序号
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # 容器参数规范，并具有将解析器作为一个整体应用的选项
    parser = argparse.ArgumentParser()
    path = "C:/Users/Jerry/AppData/Roaming/SPB_16.6/REDNet-pytorch"
    parser.add_argument('--arch', type=str, default='REDNet10', help='REDNet10, REDNet20, REDNet30')
    parser.add_argument('--images_dir', type=str, default = path+"/img")
    parser.add_argument('--outputs_dir', type=str, default = path+"/model")
    parser.add_argument('--jpeg_quality', type=int, default=10)
    parser.add_argument('--patch_size', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--threads', type=int, default=0)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--use_fast_loader', action='store_true')
    # 运行解析器并放置提取的数据在 argparse.Namespace 对象
    opt = parser.parse_args()

    # 模型保存路径
    if not os.path.exists(opt.outputs_dir):
        os.makedirs(opt.outputs_dir)

    # 设置随机数种子
    torch.manual_seed(opt.seed)

    # 加载模型
    if opt.arch == 'REDNet10':
        model = REDNet10()
    elif opt.arch == 'REDNet20':
        model = REDNet20()
    elif opt.arch == 'REDNet30':
        model = REDNet30()

    # 将模型加载到指定设备上
    model = model.to(device)
    # 创建一个条件，用于测量 L2 范数之间的均方误差
    criterion = nn.MSELoss()

    # 指定特定于优化器的选项，优化的参数、学习率
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    # 加载数据集
    dataset = Dataset(opt.images_dir, opt.patch_size, opt.jpeg_quality, opt.use_fast_loader)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=opt.threads,
                            pin_memory=True,
                            drop_last=True)

    # 训练模型
    for epoch in range(opt.num_epochs):
        epoch_losses = AverageMeter()

        with tqdm(total=(len(dataset) - len(dataset) % opt.batch_size)) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, opt.num_epochs))
            for data in dataloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)

                loss = criterion(preds, labels)
                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _tqdm.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                _tqdm.update(len(inputs))

        torch.save(model.state_dict(), os.path.join(opt.outputs_dir, '{}_epoch_{}.pth'.format(opt.arch, epoch)))
