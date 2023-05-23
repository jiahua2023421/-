import argparse
import os, glob, datetime
import re
import torch
from torch.nn.modules.loss import _Loss

# 参数设置
def parse_args(flag):
    if flag == 1:
        # 测试
        # 容器参数规范，并具有将解析器作为一个整体应用选项
        test_parser = argparse.ArgumentParser()#创建测试解析器
        test_parser.add_argument('--set_dir', default='data/Test', type=str, help='directory of test dataset')  #测试数据目录data/Test
        test_parser.add_argument('--set_names', default=['Set68', 'Set12'], help='directory of test dataset')  #测试数据目录'Set68'，'Set12'
        test_parser.add_argument('--sigma', default=25, type=int, help='noise level') #噪声水平 整型 默认25
        test_parser.add_argument('--model_dir', default=os.path.join('models', 'DnCNN_sigma25'), help='directory of the model')#model模型目录
        # os.path.join  路径拼接
        test_parser.add_argument('--model_name', default='model_001.pth', type=str, help='the model name')#保存训练模型model_001.pth
        test_parser.add_argument('--result_dir', default='results', type=str, help='directory of test dataset')#测试结果目录
        test_parser.add_argument('--save_result', default=0, type=int, help='save the denoised image, 1 or 0')#保存测试结果
        return test_parser.parse_args()
    else:
        # 训练
        train_parser = argparse.ArgumentParser(description='PyTorch DnCNN')#创建训练解析器
        train_parser.add_argument('--model', default='DnCNN', type=str, help='choose a type of model')#选择训练模型
        train_parser.add_argument('--batch_size', default=64, type=int, help='batch size')#批量大小  整型   默认大小128
        train_parser.add_argument('--train_data', default='data/Train400', type=str, help='path of train data')#训练数据  字符串型  默认 data/Train400  路径
        train_parser.add_argument('--sigma', default=25, type=int, help='noise level')#噪声水平 整型 默认25
        train_parser.add_argument('--epoch', default=180, type=int, help='number of train epoches')#epoch 整型  默认180
        train_parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')#学习率  float 0.001  adam优化算法
        return train_parser.parse_args()


# 测试点
def findLastCheckpoint(save_dir):
    #返回所有匹配的文件路径列表。定义了文件路径匹配规则，这里可以是绝对路径，也可以是相对路径。
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            # re.findall（返回string中所有与pattern相匹配的全部字串，返回形式为数组）
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))#append() 方法用于在列表末尾添加新的对象。
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0#返回值要么是最大的，要么是0
    return initial_epoch

# 日志

def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)#strftime()方法使用日期，时间或日期时间对象返回表示日期及时间的字符串

# 损失函数类
class sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    误差平方和= 1/2 * nn.mseloss(reduction = ' sum ')
    反向定义为:输入--目标
    """
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    # mse_loss计算input与target之差的平方
    # reduce(bool)- 返回值是否为标量，默认为True size_average(bool)- 当reduce=True时有效。为True时，返回的loss为平均值；为False时，返回的各样本的loss之和。
    def forward(self, input, target):
        # 返回torch.sum(torch.pow(input-target,2),(0,1,2,3)).div_(2)
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)
                # 测量元素均方误差