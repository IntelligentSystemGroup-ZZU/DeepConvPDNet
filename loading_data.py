import os

import hdf5storage
import torch
import torchvision.transforms as transforms
import torch.utils.data as td
from matplotlib import pyplot as plt
from torchvision.transforms.functional import to_tensor

import scipy.io
import os
import numpy as np
from PIL import Image
from skimage.util import random_noise
from skimage.filters import gaussian

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import scipy.io as sio
import h5py
import numpy as np
import os
import re

# 定义自然排序函数
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

from torch.utils.data import Dataset, DataLoader
# ####################################构建数据集，三通道（MNIST图像）###################################
# # 定义数据增强方法
# transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     transforms.RandomRotation(45),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])

# 设置环境变量 KMP_DUPLICATE_LIB_OK 为 TRUE
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# class NoisyMNISTDataset(td.Dataset):
#     def __init__(self, root_dir, mode='train_noise', image_size=(28, 28), blur_size=3, sigma=10):
#         super(NoisyMNISTDataset, self).__init__()
#         self.mode = mode
#         self.image_size = image_size
#         self.blur_size = blur_size
#         self.sigma = sigma
#         self.images_dir = os.path.join(root_dir, mode)
#         self.files = os.listdir(self.images_dir)
#
#     def __len__(self):
#         return len(self.files)
#
#     def __repr__(self):
#         return "NoisyIMGDataset(mode={}, image_size={}, blur_size={}, sigma={})". \
#             format(self.mode, self.image_size, self.blur_size, self.sigma)
#
#     def __getitem__(self, idx):
#         name = self.files[idx]
#         degraded_data = scipy.io.loadmat(
#             os.path.join(self.images_dir, os.path.splitext(name)[0] + '.mat'))
#         noisy = degraded_data['noisy']
#         clean = degraded_data['clean']
#         # noisy = (noisy * 255).astype(np.uint8)
#         # clean = (clean * 255).astype(np.uint8)
#         # Create RGB images from grayscale
#         rgb_noisy = np.stack((noisy,) * 3, axis=-1)
#         rgb_clean = np.stack((clean,) * 3, axis=-1)
#
#
#         rgb_noisy = rgb_noisy.transpose(2, 0, 1)
#         rgb_clean = rgb_clean.transpose(2, 0, 1)
#
#         return rgb_noisy, rgb_clean, name
# class NoisyMNISTDataset(td.Dataset):
#     def __init__(self, root_dir, mode='train_noise', image_size=(28, 28), blur_size=3, sigma=10):
#         super(NoisyMNISTDataset, self).__init__()
#         self.mode = mode
#         self.image_size = image_size
#         self.blur_size = blur_size
#         self.sigma = sigma
#         self.images_dir = os.path.join(root_dir, mode)
#         self.files = os.listdir(self.images_dir)
#
#     # 获取数据集长度
#     def __len__(self):
#         return len(self.files)
#
#     def __repr__(self):
#         return "NoisyIMGDataset(mode={}, image_size={}, blur_size={}, sigma={})". \
#             format(self.mode, self.image_size, self.blur_size, self.sigma)
#
#     def __getitem__(self, idx):
#         name = self.files[idx]
#         degraded_data = scipy.io.loadmat(
#             os.path.join(self.images_dir, os.path.splitext(name)[0] + '.mat'))
#         noisy = degraded_data['noisy']
#         clean = degraded_data['clean']
#         noisy = (noisy * 255).astype(np.uint8)
#         clean = (clean * 255).astype(np.uint8)
#         # 创建三通道图像
#         rgb_noisy = np.zeros((28, 28, 3), np.uint8)
#         rgb_clean = np.zeros((28, 28, 3), np.uint8)
#         # 复制单通道图像到三通道图像
#         rgb_noisy[:, :, 0] = noisy
#         rgb_noisy[:, :, 1] = noisy
#         rgb_noisy[:, :, 2] = noisy
#         rgb_clean[:, :, 0] = clean
#         rgb_clean[:, :, 1] = clean
#         rgb_clean[:, :, 2] = clean
#
#         rgb_noisy = rgb_noisy.transpose(2, 0, 1)  # (C, H, W)，从(28,28,3)变为(3,28,28)
#         rgb_clean = rgb_clean.transpose(2, 0, 1)
#
#         # # Convert grayscale images to RGB
#         # if len(noisy.shape) == 2:
#         #     noisy = np.stack((noisy,) * 3, axis=-1)
#         #     clean = np.stack((clean,) * 3, axis=-1)
#         # else:
#         #     noisy = noisy.transpose(2, 0, 1)  # (C, H, W)
#         #     clean = clean.transpose(2, 0, 1)  # (C, H, W)
#         #
#         # noisy_size = tuple(noisy.shape[1:])
#         # if noisy_size != self.image_size:
#         #     noisy = noisy.transpose(0, 2, 1)
#         #     clean = clean.transpose(0, 2, 1)
#
#         # print(rgb_noisy.shape)
#
#         # noisy = Image.fromarray(noisy)
#         # clean = Image.fromarray(clean)
#
#         return rgb_noisy, rgb_clean, name


# # ####################################原本使用mnist数据集，单通道（mnist黑白图像）######################################
class NoisyIMGDataset(td.Dataset):
    def __init__(self, root_dir, mode='train_noise', image_size=(28, 28), blur_size=3, sigma=30):
        super(NoisyIMGDataset, self).__init__()
        self.mode = mode
        self.image_size = image_size
        self.blur_size = blur_size
        self.sigma = sigma
        self.images_dir = os.path.join(root_dir, mode)
        self.files = sorted(os.listdir(self.images_dir), key=natural_sort_key)  # 使用自然排序

    # 获取数据集长度
    def __len__(self):
        return len(self.files)

    def __repr__(self):
        return "NoisyIMGDataset(mode={}, image_size={}, blur_size={}, sigma={})". \
            format(self.mode, self.image_size, self.blur_size, self.sigma)

    def __getitem__(self, idx):
        name = self.files[idx]
        # 读取 .mat 文件的数据
        degraded_data = scipy.io.loadmat(os.path.join(self.images_dir, os.path.splitext(name)[0] + '.mat'))
        noisy = degraded_data['noisy']
        clean = degraded_data['clean']

        # 调整数据形状 (1, H, W)
        noisy = noisy.reshape(1, *noisy.shape)
        clean = clean.reshape(1, *clean.shape)

        return noisy, clean, name
# # ####################################bsd数据集，单通道######################################


class NoisyBSD68Dataset(td.Dataset):
    def __init__(self, root_dir, mode='train_noise', image_size=(10, 10), field=10, blur_size=5, sigma=75):
        super(NoisyBSD68Dataset, self).__init__()
        self.mode = mode
        self.image_size = image_size
        self.field = field
        self.blur_size = blur_size
        self.sigma = sigma
        self.images_dir = os.path.join(root_dir, mode)
        self.files = os.listdir(self.images_dir)

        # 对文件名进行自然排序
        self.files.sort(key=natural_sort_key)

    # 获取数据集长度
    def __len__(self):
        return len(self.files)

    def __repr__(self):
        return "NoisyIMGDataset(mode={}, image_size={}, blur_size={}, sigma={})". \
            format(self.mode, self.image_size, self.blur_size, self.sigma)

    def __getitem__(self, idx):
        name = self.files[idx]
        degraded_data = hdf5storage.loadmat(os.path.join(self.images_dir, os.path.splitext(name)[0] + '.mat'))
        noisy = degraded_data['noisy']
        clean = degraded_data['clean']

        noisy = noisy.reshape(1, *noisy.shape)
        clean = clean.reshape(1, *clean.shape)

        return noisy, clean, name
