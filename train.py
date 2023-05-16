# 官方库
import torch, os, time
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
    batch_size = args.batch_size
    cuda = torch.cuda.is_available()
    n_epoch = args.epoch
    sigma = args.sigma
    # 保存模型路径
    save_dir = os.path.join('models', args.model + '_' + 'sigma' + str(sigma))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    initial_epoch = findLastCheckpoint(save_dir=save_dir)  # load the last model in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        # model.load_state_dict(torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch)))
        model = torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch))
    model.train()
    # criterion = nn.MSELoss(reduction = 'sum')  # PyTorch 0.4.1
    criterion = sum_squared_error()
    # GPU or CPU
    if cuda:
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)  # learning rates
    # 训练
    for epoch in range(initial_epoch, n_epoch):
        scheduler.step(epoch)  # step to the learning rate in this epcoh
        xs = dp.datagenerator(data_dir=args.train_data)
        xs = xs.astype('float32') / 255.0
        xs = torch.from_numpy(xs.transpose((0, 3, 1, 2)))  # tensor of the clean patches, NXCXHXW
        DDataset = DenoisingDataset(xs, sigma)
        DLoader = DataLoader(dataset=DDataset, num_workers=1, drop_last=True, batch_size=batch_size, shuffle=True)
        epoch_loss = 0
        start_time = time.time()

        for n_count, batch_yx in enumerate(DLoader):
            optimizer.zero_grad()
            if cuda:
                batch_x, batch_y = batch_yx[1].cuda(), batch_yx[0].cuda()
            # loss = criterion(model(batch_yx[0]), batch_yx[1])
            loss = criterion(model(batch_y), batch_x)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            if n_count % 10 == 0:
                print('%4d %4d / %4d loss = %2.4f' % (
                epoch + 1, n_count, xs.size(0) // batch_size, loss.item() / batch_size))
        elapsed_time = time.time() - start_time

        log('epcoh = %4d , loss = %4.4f , time = %4.2f s' % (epoch + 1, epoch_loss / n_count, elapsed_time))
        np.savetxt('train_result.txt', np.hstack((epoch + 1, epoch_loss / n_count, elapsed_time)), fmt='%2.4f')
        # torch.save(model.state_dict(), os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))
        torch.save(model, os.path.join(save_dir, 'model_%03d.pth' % (epoch + 1)))