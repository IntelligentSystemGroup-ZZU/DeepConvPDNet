import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


import scipy.io
import os
from prettytable import PrettyTable


# 设置学习率为初始学习率乘以给定lr_lambda函数的值
class LambdaLR:
    def __init__(self, num_epochs, start_epoch,
                 decay_start_epoch):  # (num_epochs = 50, offset = start_epoch, decay_start_epoch = 30)
        assert (
                       num_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"  # 断言，要让num_epochs > decay_start_epoch 才可以
        self.num_epochs = num_epochs
        self.start_epoch = start_epoch
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):  # return    1-max(0, epoch - 30) / (50 - 30)
        return 1.0 - max(0, epoch + self.start_epoch - self.decay_start_epoch) / (
                self.num_epochs - self.decay_start_epoch)


def psnr(img1, img2):
    img1 = img1 / 255.0
    img1 = img1.data.detach().cpu().numpy()
    # print(img1.shape)
    img2 = img2 / 255.0
    img2 = img2.data.detach().cpu().numpy()
    return compare_psnr(img1, img2)


def ssim(img1, img2):
    img1 = img1.data.detach().cpu().numpy()
    # img1 = np.squeeze(img1, axis=0)
    # print(img1.shape)
    img2 = img2.data.detach().cpu().numpy()
    # img2 = np.squeeze(img2, axis=0)
    # print(img2.shape)
    return compare_ssim(img1, img2, channel_axis=0, data_range=255, win_size=3)

def n_ssim(img1, img2):
    # img1 and img2: (N, C, H, W)
    N, C, H, W = img1.shape
    average_ssim = 0.0
    for i in range(N):
        temp1, temp2 = img1[i].reshape([C, H, W]), img2[i].reshape([C, H, W])
        temp_ssim = ssim(temp1, temp2)
        average_ssim += temp_ssim

    average_ssim /= N
    return average_ssim
# # ***********************************************************************************************************************************************************************************
# import numpy as np
# from skimage.metrics import structural_similarity as compare_ssim
# from skimage.metrics import peak_signal_noise_ratio as compare_psnr


# # import scipy.io
# # import os
# # from prettytable import PrettyTable


# # 设置学习率为初始学习率乘以给定lr_lambda函数的值
# class LambdaLR:
#     def __init__(self, num_epochs, start_epoch,
#                  decay_start_epoch):  # (num_epochs = 50, offset = start_epoch, decay_start_epoch = 30)
#         assert (
#                        num_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"  # 断言，要让num_epochs > decay_start_epoch 才可以
#         self.num_epochs = num_epochs
#         self.start_epoch = start_epoch
#         self.decay_start_epoch = decay_start_epoch

#     def step(self, epoch):  # return    1-max(0, epoch - 30) / (50 - 30)
#         return 1.0 - max(0, epoch + self.start_epoch - self.decay_start_epoch) / (
#                 self.num_epochs - self.decay_start_epoch)


# def psnr(img1, img2):
#     img1 = img1 / 255.0
#     img1 = img1.data.detach().cpu().numpy()
#     # print(img1.shape)
#     img2 = img2 / 255.0
#     img2 = img2.data.detach().cpu().numpy()
#     return compare_psnr(img1, img2)


# def ssim(img1, img2):
#     img1 = img1.data.detach().cpu().numpy()
#     img1 = np.squeeze(img1, axis=0)
#     # print(img1.shape)
#     img2 = img2.data.detach().cpu().numpy()
#     img2 = np.squeeze(img2, axis=0)
#     # print(img2.shape)
#     return compare_ssim(img1, img2, channel_axis=0, data_range=255)

# def n_ssim(img1, img2):
#     # img (N, C, H, W)
#     N, C, H, W = img1.shape
#     # N, _, _, _ = img2.shape
#     average_ssim = 0.0
#     for i in range(N):
#         temp1, temp2 = img1[i].reshape([1, C, H, W]), img2[i].reshape([1, C, H, W])

#         temp = ssim(temp1, temp2)
#         average_ssim += temp
#         # average_ssim += ssim(temp1, temp2)
#     average_ssim /= N
#     return average_ssim
