# 官方库
import torch, os, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from imageio import imread
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
 # 私人库

from public import parse_args, log
from data_process import show, save_result
from model import DnCNN



if __name__ == '__main__':
    # 参数
    args = parse_args(1)  #测试集

    # model = DnCNN()
    if not os.path.exists(os.path.join(args.model_dir, args.model_name)):  # model_001若不存在，则加载model

        model_cpu = torch.load(os.path.join(args.model_dir, 'model.pth'), map_location='cpu')  # 映射到CPU，此模型不会被加载到cuda
        model = torch.load(os.path.join(args.model_dir, 'model.pth'), map_location='cpu')  # 映射到CPU
        # load weights into new model
        log('load trained model on Train400 dataset by kai')  #张凯的模型
    else:
        # model.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_name)))
        model_cpu = torch.load(os.path.join(args.model_dir, args.model_name), map_location='cpu')
        model = torch.load(os.path.join(args.model_dir, args.model_name), map_location='cpu')
        log('load trained model')   #本地训练的模型

    #    params = model.state_dict()
    #    print(params.values())
    #    print(params.keys())
    #
    #    for key, value in params.items():
    #        print(key)    # parameter name
    #    print(params['dncnn.12.running_mean'])
    #    print(model.state_dict())

    model_cpu.eval()  # evaluation mode  设置为预测模式
    #    model.train()

    if torch.cuda.is_available():  # GPU
        model = model.cuda()  # 加载到GPU

    if not os.path.exists(args.result_dir):  # 结果路径
        os.mkdir(args.result_dir)  # 创造目录

    for set_cur in args.set_names:  # 测试图片的文件名

        if not os.path.exists(os.path.join(args.result_dir, set_cur)):  # 未找到保存文件的路径，则创造路径
            os.mkdir(os.path.join(args.result_dir, set_cur))
        psnrs = []  # 计算psnr与ssim的数组
        ssims = []

        for im in os.listdir(os.path.join(args.set_dir, set_cur)):  # 返回指定的文件夹包含的文件或文件夹的名字的列表
            if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):
                # 判断字符串是否以指定后缀结尾，如果以指定后缀结尾返回True，否则返回False
                x = np.array(imread(os.path.join(args.set_dir, set_cur, im)), dtype=np.float32) / 255.0
                #  dtype:数组中的数据类型

                np.random.seed(seed=0)  # for reproducibility
                y = x + np.random.normal(0, args.sigma / 255.0, x.shape)  # Add Gaussian noise without clipping
                y = y.astype(np.float32)
                y_ = torch.from_numpy(y).view(1, -1, y.shape[0], y.shape[1])

                # torch.cuda.synchronize()
                start_time = time.time()
                # ceshi = y_.nelement
                # ceshi2 = y_.squeeze(2)
                ceshi = y.size
                if y.size < 154402:
                    y_ = y_.cuda()
                    x_ = model(y_)  # inference
                else:
                    x_ = model_cpu(y_)
                x_ = x_.view(y.shape[0], y.shape[1])
                x_ = x_.cpu()
                x_ = x_.detach().numpy().astype(np.float32)
                # torch.cuda.synchronize()
                # elapsed_time = time.time() - start_time
                # print('%10s : %10s : %2.4f second' % (set_cur, im, elapsed_time))

                psnr_x_ = compare_psnr(x, x_)
                ssim_x_ = compare_ssim(x, x_)

                ssim_x_ = compare_ssim(x, x_)

                # if args.save_result:
                name, ext = os.path.splitext(im)
                # show(np.hstack((y, x_)))  # show the image
                save_result(x_, path=os.path.join(args.result_dir, set_cur,
                                                  name + '_dncnn' + ext))  # save the denoised image
                psnrs.append(psnr_x_)
                ssims.append(ssim_x_)
        psnr_avg = np.mean(psnrs)
        ssim_avg = np.mean(ssims)
        psnrs.append(psnr_avg)
        ssims.append(ssim_avg)
        if args.save_result:
         save_result(np.hstack((psnrs, ssims)), path=os.path.join(args.result_dir, set_cur, 'results.txt'))
        log('Datset: {0:10s} \n  PSNR = {1:2.2f}dB, SSIM = {2:1.4f}'.format(set_cur, psnr_avg, ssim_avg))
