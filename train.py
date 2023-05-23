# 官方库
import os
import torch
import time
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
# 私人库
from model import DnCNN
from public import findLastCheckpoint, parse_args, sum_squared_error, log
import data_process as dp
from data_process import DenoisingDataset

if __name__ == '__main__':
    # 加载模型
    model = DnCNN()
    # 参数初始化
    # 运行解析器并放置提取的数据在 argparse.Namespace 对象
    args = parse_args(0)
    batch_size = args.batch_size  # 一次训练所抓取的数据样本数量 BATCH_SIZE的大小影响训练速度和模型优化。
    cuda = torch.cuda.is_available()  # 函数返回值为bool值
    n_epoch = args.epoch  # 当一个完整的数据集通过了神经网络一次并且返回了一次，为一个epoch
    sigma = args.sigma  # 高斯噪声级别
    # 保存模型路径
    save_dir = os.path.join('models', args.model + '_' + 'sigma' + str(sigma))
    if not os.path.exists(save_dir):  # 若不存在路径则添加路径
        os.mkdir(save_dir)

    initial_epoch = findLastCheckpoint(save_dir=save_dir)  # load the last model in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        # model.load_state_dict(torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch)))
        model = torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch))  # 加载模型
    model.train()
    # model.train()的作用是启用 Batch Normalization 和 Dropout。

# 如果模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加model.train()。
    # model.train()是保证BN层能够用到每一批数据的均值和方差。对于Dropout，model.train()是随机取一部分网络连接来训练更新参数。
    # criterion = nn.MSELoss(reduction = 'sum')  # PyTorch 0.4.1
    criterion = sum_squared_error()  # 测量元素均方误差
    # GPU or CPU
    if cuda:
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # adam优化算法
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)  # learning rates
    # 按设定的间隔调整学习率。这个方法适合后期调试使用，观察 loss 曲线，为每个实验定制学习率调整时机
    # milestones(list)- 一个 list，每一个元素代表何时调整学习率， list 元素必须是递增的。如 milestones=[30,80,120]
    # gamma(float)- 学习率调整倍数，默认为 0.1 倍，即下降 10 倍。

    # 训练
    # epoch = 1
    for epoch in range(initial_epoch, n_epoch):
        scheduler.step(epoch)  # step to the learning rate in this epcoh 恢复optimizer的学习率
        xs = dp.datagenerator(data_dir=args.train_data)   # 路径名 生成多批次的数据集
        xs = xs.astype('float32') / 255.0  # 数组转为float32 , 归一化
        xs = torch.from_numpy(xs.transpose((0, 3, 1, 2)))  # tensor of the clean patches, NXCXHXW
        # 变为张量， 238336 ， 1， 40 ， 40
        DDataset = DenoisingDataset(xs, sigma)    # 返回batch_y, batch_x
        # 加噪声
        # 238336 / batch_size
        DLoader = DataLoader(dataset=DDataset, num_workers=1, drop_last=True, batch_size=batch_size, shuffle=True)
        # DLoader = DataLoader(dataset=DDataset, num_workers=1, drop_last=True, batch_size=batch_size, shuffle=False)
        # dataset (Dataset) – 决定数据从哪读取或者从何读取；
        # num_workers (python:int, optional) – 是否多进程读取数据（默认为０);
        # drop_last (bool, optional) – 当样本数不能被batchsize整除时，最后一批数据是否舍弃（default: False)
        # batchszie：批大小，决定一个epoch有多少个Iteration；
        # shuffle (bool, optional) –每一个 epoch是否为乱序 (default: False)；

        epoch_loss = 0
        start_time = time.time()
        for n_count, batch_yx in enumerate(DLoader):
            # batch_yx 为两个列表， 列表的第一个和第二个元素均是64，1，40，40的张量
            # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中
            optimizer.zero_grad()  # 清空过往梯度；
            if cuda:
                batch_x, batch_y = batch_yx[1].cuda(), batch_yx[0].cuda()
            # loss = criterion(model(batch_yx[0]), batch_yx[1])
            loss = criterion(model(batch_y), batch_x)  # 计算损失函数 测量预测值与实际值之间差异的函数
            epoch_loss += loss.item()
# .item()方法是，取一个元素张量里面的具体元素值并返回该值，可以将一个零维张量转换成int型或者float型，在计算loss，accuracy时常用到。
# 作用：
# 1.item（）取出张量具体位置的元素元素值
# 2.并且返回的是该位置元素值的高精度值
# 3.保持原元素类型不变；必须指定位置
# 4.节省内存（不会计入计算图）


            loss.backward()  # 反向传播，计算当前梯度
            optimizer.step()  # 根据梯度更新网络参数

            ceshi1 = xs.size(0)
            ceshi = xs.size(0) // batch_size
            if n_count % 10 == 0:  # 每循环10遍，输出一次结果
                print('%4d %4d / %4d loss = %2.4f' % (
                epoch + 1, n_count, xs.size(0) // batch_size, loss.item() / batch_size))
        elapsed_time = time.time() - start_time  # 结束时间

        log('epcoh = %4d , loss = %4.4f , time = %4.2f s' % (epoch + 1, epoch_loss / n_count, elapsed_time))
        np.savetxt('train_result.txt', np.hstack((epoch + 1, epoch_loss / n_count, elapsed_time)), fmt='%2.4f')
        # 保存
        torch.save(model, os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))  # 保存模型
    # torch.save(model, os.path.join(save_dir, 'model_001.pth'))
    # torch.save(model, os.path.join(save_dir, 'model_%03d.pth' % (epoch + 1)))
