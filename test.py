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
    args = parse_args(1)

    # 加载模型
    if not os.path.exists(os.path.join(args.model_dir, args.model_name)):  # model_001若不存在，则加载model
        model = torch.load(os.path.join(args.model_dir, 'model.pth'), map_location='cpu')  # 映射到CPU
        log('load trained model on Train400 dataset by kai')
    else:
        model = torch.load(os.path.join(args.model_dir, args.model_name), map_location='cpu')
        log('load trained model')

    model.eval()  # evaluation mode  设置为预测模式

    # GPU or CPU
    #    if torch.cuda.is_available():  #GPU
    #        model = model.cuda()

    # 保存路径
    if not os.path.exists(args.result_dir):  # 结果路径
        os.mkdir(args.result_dir)  # 创造目录

    # 测试图片保存
    for set_cur in args.set_names:
        if not os.path.exists(os.path.join(args.result_dir, set_cur)):
            os.mkdir(os.path.join(args.result_dir, set_cur))
        psnrs = []
        ssims = []

        for im in os.listdir(os.path.join(args.set_dir, set_cur)):  # 返回指定的文件夹包含的文件或文件夹的名字的列表
            if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):
                x = np.array(imread(os.path.join(args.set_dir, set_cur, im)), dtype=np.float32) / 255.0
                #  dtype:数组中的数据类型
                np.random.seed(seed=0)  # for reproducibility
                y = x + np.random.normal(0, args.sigma / 255.0, x.shape)  # Add Gaussian noise without clipping
                y = y.astype(np.float32)
                y_ = torch.from_numpy(y).view(1, -1, y.shape[0], y.shape[1])

                torch.cuda.synchronize()
                start_time = time.time()
                #               y_ = y_.cuda()
                x_ = model(y_)  # inference
                x_ = x_.view(y.shape[0], y.shape[1])
                x_ = x_.cpu()
                x_ = x_.detach().numpy().astype(np.float32)
                torch.cuda.synchronize()
                elapsed_time = time.time() - start_time
                print('%10s : %10s : %2.4f second' % (set_cur, im, elapsed_time))

                psnr_x_ = compare_psnr(x, x_)
                ssim_x_ = compare_ssim(x, x_)
                # if args.save_result:
                name, ext = os.path.splitext(im)
                show(np.hstack((y, x_)))  # show the image
                save_result(x_, path=os.path.join(args.result_dir, set_cur,
                                                  name + '_dncnn' + ext))  # save the denoised image
                psnrs.append(psnr_x_)
                ssims.append(ssim_x_)
        psnr_avg = np.mean(psnrs)
        ssim_avg = np.mean(ssims)
        psnrs.append(psnr_avg)
        ssims.append(ssim_avg)
        # if args.save_result:
        save_result(np.hstack((psnrs, ssims)), path=os.path.join(args.result_dir, set_cur, 'results.txt'))
        log('Datset: {0:10s} \n  PSNR = {1:2.2f}dB, SSIM = {2:1.4f}'.format(set_cur, psnr_avg, ssim_avg))