import h5py
from PIL import Image
import os
import numpy as np
import glob
import math

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def crop_patch(img, img_size=(512, 512), patch_size=(150, 150), stride=150, random_crop=False):
    count = 0
    patch_list = []
    if random_crop == True:
        crop_num = 100
        pos = [(np.random.randint(patch_size[0], img_size[0] - patch_size[0]),
                np.random.randint(patch_size[1], img_size[1] - patch_size[1]))
               for i in range(crop_num)]
    else:
        pos = [(x, y) for x in range(patch_size[1], img_size[1] - patch_size[1], stride) for y in
               range(patch_size[0], img_size[0] - patch_size[0], stride)]

    for (xt, yt) in pos:
        cropped_img = img[yt - patch_size[0]:yt + patch_size[0], xt - patch_size[1]:xt + patch_size[1]]
        patch_list.append(cropped_img)
        count += 1

    return patch_list


def gen_dataset(src_files, dst_path):
    create_dir(dst_path)
    h5py_name = dst_path + "train.h5"
    h5f = h5py.File(h5py_name, 'w')

    count = 0
    for src_path in src_path_list:
        files_source = os.listdir(src_path)
        file_path = glob.glob(src_path + '*')

        for file_name in files_source:

                # gt_imgs = glob.glob(file_name + '/*GT*.PNG')
                # gt_imgs = glob.glob(file_name)
                # ceshi = src_path + 'GT*.png'
                # ceshi2 = glob.glob(ceshi)
            gt_imgs = glob.glob(src_path + '/GT*.PNG')
            gt_imgs.sort()
                # noisy_imgs = glob.glob(file_name + '/*NOISY*.PNG')
                # noisy_imgs = glob.glob(file_name)
            noisy_imgs = glob.glob(src_path + '/NOISY*.PNG')
            noisy_imgs.sort()
            print('SIDD processing...' + str(count))
            for i in range(len(noisy_imgs)):
                gt = np.array(Image.open(gt_imgs[i]))
                noisy = np.array(Image.open(noisy_imgs[i]))
                for i in range(0, math.floor((gt.shape[0]) / 256)):
                    for j in range(0, math.floor((gt.shape[1]) / 256)):
                        cropped = gt[0 + i * 256:256 + i * 256, 0 + j * 256:256 + j * 256]  # 裁剪坐标为[y0:y1, x0:x1]
                        cropped2 = noisy[0 + i * 256:256 + i * 256, 0 + j * 256:256 + j * 256]  # 裁剪坐标为[y0:y1, x0:x1]
                img = np.concatenate([cropped2, cropped], 2)
                [h, w, c] = img.shape
                patch_list = crop_patch(img, (h, w), (64, 64), 64, True)
                for num in range(len(patch_list)):
                    data = patch_list[num].copy()
                    h5f.create_dataset(str(count), shape=(128, 128, 6), data=data)
                    count += 1

            # if 'RENOIR' in file_name:
            #     ref_imgs = glob.glob(file_name + '/*Reference.bmp')
            #     full_imgs = glob.glob(file_name + '/*full.bmp')
            #     noisy_imgs = glob.glob(file_name + '/*Noisy.bmp')
            #     noisy_imgs.sort()
            #
            #     ref = np.array(Image.open(ref_imgs[0])).astype(np.float32)
            #     full = np.array(Image.open(full_imgs[0])).astype(np.float32)
            #     gt = (ref + full) / 2
            #     gt = np.clip(gt, 0, 255).astype(np.uint8)
            #     print('RENOIR processing...' + str(count))
            #     for i in range(len(noisy_imgs)):
            #         noisy = np.array(Image.open(noisy_imgs[i]))
            #         img = np.concatenate([noisy, gt], 2)
            #         [h, w, c] = img.shape
            #         patch_list = crop_patch(img, (h, w), (150, 150), 150, False)
            #         for num in range(len(patch_list)):
            #             data = patch_list[num].copy()
            #             h5f.create_dataset(str(count), shape=(300, 300, 6), data=data)
            #             count += 1

    h5f.close()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    src_path_list = ["./data/data/0001_001_S6_00100_00060_3200_L",
                     "./data/data/0002_001_S6_00100_00020_3200_N",
                     "./data/data/0003_001_S6_00100_00060_3200_H",
                     "./data/data/0004_001_S6_00100_00060_4400_L",
                     "./data/data/0005_001_S6_00100_00060_4400_N",
                     "./data/data/0006_001_S6_00100_00060_4400_H",
                     "./data/data/0007_001_S6_00100_00100_5500_L",
                     "./data/data/0008_001_S6_00100_00100_5500_N",
                     "./data/data/0010_001_S6_00800_00350_3200_N",
                     "./data/data/0011_001_S6_00800_00500_5500_L",
                     "./data/data/0012_001_S6_00800_00500_5500_N",
                     "./data/data/0013_001_S6_03200_01250_3200_L",
                     "./data/data/0014_001_S6_03200_01250_3200_N",
                     "./data/data/0015_001_S6_03200_01600_5500_L",
                     "./data/data/0016_001_S6_03200_01600_5500_N",
                     "./data/data/0017_001_GP_00100_00060_5500_N",
                     "./data/data/0018_001_GP_00100_00160_5500_L",
                     "./data/data/0019_001_GP_00800_00640_5500_L",
                     "./data/data/0020_001_GP_00800_00350_5500_N",
                     "./data/data/0022_001_N6_00100_00060_5500_N",
                     "./data/data/0023_001_N6_00800_00350_5500_N",
                     "./data/data/0025_001_G4_00100_00060_5500_L",
                     "./data/data/0027_001_G4_00800_00350_5500_L",
                     "./data/data/0028_001_IP_00100_00160_5500_N",
                     "./data/data/0029_001_IP_00800_01000_5500_N",
                     "./data/data/0030_001_IP_01600_02000_5500_N",
                     "./data/data/0032_001_IP_00800_01000_3200_N",
                     "./data/data/0033_001_IP_00100_00160_3200_N",
                     "./data/data/0034_002_GP_00100_00160_3200_N",
                     "./data/data/0035_002_GP_00800_00350_3200_N",
                     "./data/data/0036_002_GP_06400_03200_3200_N",
                     "./data/data/0038_002_GP_00800_00640_3200_L",
                     "./data/data/0039_002_IP_00100_00180_5500_L",
                     "./data/data/0040_002_IP_00800_02000_5500_L",
                     "./data/data/0042_002_IP_01600_03100_5500_N",
                     "./data/data/0043_002_IP_00800_01520_5500_N",
                     "./data/data/0044_002_IP_00100_00180_5500_N",
                     "./data/data/0045_002_G4_00100_00060_3200_L",
                     "./data/data/0047_002_G4_00800_00640_3200_L",
                     "./data/data/0048_002_N6_00100_00100_5500_L",
                     "./data/data/0050_002_N6_03200_03200_5500_L",
                     "./data/data/0051_002_S6_00100_00060_5500_N",
                     "./data/data/0052_002_S6_01600_01000_5500_N",
                     "./data/data/0054_003_N6_00100_00160_5500_N",
                     "./data/data/0055_003_N6_00800_01000_5500_N",
                     "./data/data/0057_003_G4_00100_00125_5500_L",
                     "./data/data/0059_003_G4_00800_01000_5500_L",
                     "./data/data/0060_003_S6_00100_00100_4400_L",
                     "./data/data/0062_003_S6_03200_02500_4400_L",
                     "./data/data/0063_003_GP_00100_00100_4400_N",
                     "./data/data/0064_003_GP_01600_01600_4400_N",
                     "./data/data/0065_003_GP_10000_08460_4400_N",
                     "./data/data/0066_003_GP_00100_00200_3200_L",
                     "./data/data/0068_003_IP_00200_00400_3200_N",
                     "./data/data/0069_003_IP_01000_02000_3200_N",
                     "./data/data/0070_003_IP_02000_04000_3200_N",
                     "./data/data/0072_003_IP_01000_02000_5500_L",
                     "./data/data/0073_003_IP_00200_01000_5500_L",
                     "./data/data/0075_004_N6_00800_00080_3200_L",
                     "./data/data/0076_004_N6_03200_00320_3200_L",
                     "./data/data/0077_004_G4_00100_00025_3200_N",
                     "./data/data/0078_004_G4_00200_00050_3200_N",
                     "./data/data/0080_004_S6_00200_00050_3200_N",
                     "./data/data/0081_004_S6_00800_00160_4400_L",
                     "./data/data/0083_004_GP_00050_00020_4400_N",
                     "./data/data/0084_004_GP_00200_00100_4400_N",
                     "./data/data/0086_004_GP_00100_00100_5500_L",
                     "./data/data/0087_004_GP_00800_00640_5500_L",
                     "./data/data/0088_004_IP_00100_00050_5500_N",
                     "./data/data/0089_004_IP_00500_00250_5500_N",
                     "./data/data/0090_004_IP_01600_00750_5500_N",
                     "./data/data/0091_004_IP_00320_00080_3200_L",
                     "./data/data/0092_004_IP_00640_00125_3200_L",
                     "./data/data/0094_005_N6_00100_00050_3200_L",
                     "./data/data/0096_005_N6_01600_01000_3200_L",
                     "./data/data/0097_005_N6_03200_02000_3200_L",
                     "./data/data/0098_005_G4_00100_00050_3200_N",
                     "./data/data/0099_005_G4_00400_00200_3200_N",
                     "./data/data/0101_005_S6_00100_00050_4400_L",
                     "./data/data/0102_005_S6_00400_00200_4400_L",
                     "./data/data/0104_005_S6_03200_01600_4400_L",
                     "./data/data/0105_005_GP_00100_00100_4400_N",
                     "./data/data/0106_005_GP_00400_00400_4400_N",
                     "./data/data/0107_005_GP_01600_01600_4400_N",
                     "./data/data/0108_005_GP_06400_06400_4400_N",
                     "./data/data/0110_005_IP_00100_00100_5500_L",
                     "./data/data/0111_005_IP_00400_00400_5500_L",
                     "./data/data/0113_005_IP_01600_01520_5500_L",
                     "./data/data/0114_005_IP_00100_00200_5500_N",
                     "./data/data/0115_005_IP_00400_00750_5500_N",
                     "./data/data/0116_005_IP_00800_01520_5500_N",
                     "./data/data/0117_005_IP_01600_04160_5500_N",
                     "./data/data/0118_006_N6_00100_00025_3200_L",
                     "./data/data/0120_006_N6_01600_00400_3200_L",
                     "./data/data/0121_006_N6_03200_01000_3200_L",
                     "./data/data/0122_006_G4_00100_00050_3200_N",
                     "./data/data/0123_006_G4_00400_00160_3200_N",
                     "./data/data/0125_006_S6_00100_00050_4400_L",
                     "./data/data/0126_006_S6_00400_00200_4400_L",
                     "./data/data/0127_006_S6_01600_00800_4400_L",
                     "./data/data/0129_006_GP_00100_00100_4400_N",
                     "./data/data/0130_006_GP_00400_00400_4400_N",
                     "./data/data/0132_006_GP_00100_00200_5500_L",
                     "./data/data/0133_006_GP_00800_01600_5500_L",
                     "./data/data/0134_006_IP_00100_00100_5500_N",
                     "./data/data/0135_006_IP_00400_00400_5500_N",
                     "./data/data/0136_006_IP_00800_00800_5500_N",
                     "./data/data/0137_006_IP_01600_01600_5500_N",
                     "./data/data/0138_006_IP_00100_00100_3200_L",
                     "./data/data/0139_006_IP_00200_00200_3200_L",
                     "./data/data/0140_006_IP_00800_00800_3200_L",
                     "./data/data/0142_007_N6_00100_00100_4400_N",
                     "./data/data/0144_007_N6_01600_01600_4400_N",
                     "./data/data/0145_007_N6_03200_03200_4400_N",
                     "./data/data/0146_007_N6_00400_00400_4400_N",
                     "./data/data/0147_007_G4_00100_00100_4400_L",
                     "./data/data/0149_007_G4_00800_00800_4400_L",
                     "./data/data/0150_007_S6_00100_00100_5500_L",
                     "./data/data/0151_007_S6_00800_00800_5500_L",
                     "./data/data/0152_007_S6_01600_01600_5500_L",
                     "./data/data/0154_007_S6_00400_00400_5500_L",
                     "./data/data/0155_007_GP_00100_00100_5500_N",
                     "./data/data/0156_007_GP_00800_00800_5500_N",
                     "./data/data/0157_007_GP_01600_01600_5500_N",
                     "./data/data/0159_007_IP_00100_00100_3200_L",
                     "./data/data/0160_007_IP_00400_00400_3200_L",
                     "./data/data/0161_007_IP_00800_00800_3200_L",
                     "./data/data/0163_007_IP_00100_00100_3200_N",
                     "./data/data/0164_007_IP_00400_00400_3200_N",
                     "./data/data/0165_007_IP_00800_00800_3200_N",
                     "./data/data/0166_007_IP_01600_01600_3200_N",
                     "./data/data/0167_008_N6_00100_00050_4400_L",
                     "./data/data/0168_008_N6_00400_00200_4400_L",
                     "./data/data/0169_008_N6_00800_00400_4400_L",
                     "./data/data/0170_008_N6_01600_00800_4400_L",
                     "./data/data/0172_008_G4_00100_00100_4400_N",
                     "./data/data/0173_008_G4_00400_00400_4400_N",
                     "./data/data/0175_008_S6_00100_00025_5500_L",
                     "./data/data/0177_008_S6_00800_00200_5500_L",
                     "./data/data/0178_008_S6_01600_00400_5500_L",
                     "./data/data/0179_008_S6_03200_00800_5500_L",
                     "./data/data/0180_008_GP_00100_00100_5500_N",
                     "./data/data/0181_008_GP_00800_00800_5500_N",
                     "./data/data/0182_008_GP_03200_03200_5500_N",
                     "./data/data/0184_008_IP_00100_00100_3200_L",
                     "./data/data/0185_008_IP_00400_00400_3200_L",
                     "./data/data/0186_008_IP_00800_00800_3200_L",
                     "./data/data/0188_008_IP_00100_00100_3200_N",
                     "./data/data/0189_008_IP_00400_00400_3200_N",
                     "./data/data/0190_008_IP_00800_00800_3200_N",
                     "./data/data/0191_008_IP_01600_01600_3200_N",
                     "./data/data/0192_009_IP_00100_00200_3200_N",
                     "./data/data/0193_009_IP_00800_02000_3200_N",
                     "./data/data/0194_009_IP_01600_04000_3200_N",
                     "./data/data/0195_009_IP_01600_04000_5500_L",
                     "./data/data/0196_009_IP_00800_02000_5500_L",
                     "./data/data/0197_009_IP_00100_00200_5500_L",
                     "./data/data/0198_010_GP_00100_00200_5500_N",
                     "./data/data/0199_010_GP_00800_01600_5500_N",
                     "./data/data/0200_010_GP_01600_03200_5500_N",
                     ]
    dst_path = "./data/SIDD_RENOIR_h5/"

    create_dir(dst_path)
    print("start...")
    gen_dataset(src_path_list, dst_path)
    print('end')
