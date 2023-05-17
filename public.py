import argparse
import os, glob, datetime
import re
import torch
from torch.nn.modules.loss import _Loss

# 参数设置
def parse_args(flag):
    if flag == 1:
        # 测试
        # 容器参数规范，并具有将解析器作为一个整体应用的选项
        test_parser = argparse.ArgumentParser()
        test_parser.add_argument('--set_dir', default='data/Test', type=str, help='directory of test dataset')
        test_parser.add_argument('--set_names', default=['Set68', 'Set12'], help='directory of test dataset')
        test_parser.add_argument('--sigma', default=25, type=int, help='noise level')
        test_parser.add_argument('--model_dir', default=os.path.join('models', 'DnCNN_sigma25'), help='directory of the model')
        # os.path.join  路径拼接
        test_parser.add_argument('--model_name', default='model_001.pth', type=str, help='the model name')
        test_parser.add_argument('--result_dir', default='results', type=str, help='directory of test dataset')
        test_parser.add_argument('--save_result', default=0, type=int, help='save the denoised image, 1 or 0')
        return test_parser.parse_args()
    else:
        # 训练
        train_parser = argparse.ArgumentParser(description='PyTorch DnCNN')
        train_parser.add_argument('--model', default='DnCNN', type=str, help='choose a type of model')
        train_parser.add_argument('--batch_size', default=128, type=int, help='batch size')
        train_parser.add_argument('--train_data', default='data/Train400', type=str, help='path of train data')
        train_parser.add_argument('--sigma', default=25, type=int, help='noise level')
        train_parser.add_argument('--epoch', default=180, type=int, help='number of train epoches')
        train_parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
        return train_parser.parse_args()


# 测试点
def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch

# 日志
def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)

# 计算损失
class sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)