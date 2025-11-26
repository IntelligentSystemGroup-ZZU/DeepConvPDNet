import numpy as np
import scipy.io as sio
import os

import hdf5storage

# 噪声等级
blur_size = 3
noise_std = 25
# 图片被分割的大小
field = 10

# 分为训练集、验证集、测试集
# mode = ["train", "test", "val"]
# for m in mode:
#     # 加载.mat文件
#     clean_data = sio.loadmat('dataset/withoutboundary/mnist/mnist_data_filter3_sig30_210102.mat')['{0}_x'.format(m)]
#     noisy_data = sio.loadmat('dataset/withoutboundary/mnist/mnist_data_filter3_sig30_210102.mat')['{0}_nx'.format(m)]
#     # 确定输出目录
#     save_dir = 'data/noisy_images/blur%d_sig%d/{0}_noise'.format(m) % (blur_size, noise_std)
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#
#     # 分割数据并将其存储为单独的.mat文件
#     for i in range(clean_data.shape[0]):
#         file_name = os.path.join(save_dir, f'{i + 1}.mat')
#         clean = clean_data[i, :].reshape(28, 28)
#         noisy = noisy_data[i, :].reshape(28, 28)
#         sio.savemat(file_name, dict(noisy=noisy, clean=clean))



# # 整张321x481图片转换为clean noisy形式的数据集
# 测试图片大小是(321, 481)
# mode = ["test"]
# for m in mode:
#     # 加载.mat文件
#     mat_data = hdf5storage.loadmat('dataset/bsd68set12/{0}_bsd68set12_f%ds%d.mat'.format(m) % (blur_size, noise_std))
#     # 在.mat文件中获取需要的数据
#     bsd68_data = mat_data['bsd68_imageset']
#     # set12_test_data = test_data['set12_imageset']
#     # 确定输出目录
#     save_dir = 'data/bsd/field%d/blur%d_sig%d/{0}_noise'.format(m) % (field, blur_size, noise_std)
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#
#     # 分割数据并将其存储为单独的.mat文件
#     for i in range(bsd68_data.shape[0]):
#         file_name = os.path.join(save_dir, f'{i + 1}.mat')
#         row_data = bsd68_data[i, :]
#         row_mat_data = {'clean': row_data[0], 'noisy': row_data[1]}
#         hdf5storage.savemat(file_name, row_mat_data)

# 将bsd400分割成(field, field)大小后作为训练集
mode = ["train"]
for m in mode:
    # 加载.mat文件
    clean_data = hdf5storage.loadmat('dataset/bsd400/bsddata_field%d_filter%d_sig%d_new_0705.mat' % ( field, blur_size, noise_std))['{0}_x'.format(m)]
    noisy_data = hdf5storage.loadmat('dataset/bsd400/bsddata_field%d_filter%d_sig%d_new_0705.mat' % ( field, blur_size, noise_std))['{0}_nx'.format(m)]
    # 确定输出目录
    save_dir = 'data/bsd/field%d/blur%d_sig%d/{0}_noise'.format(m) % (field, blur_size, noise_std)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 分割数据并将其存储为单独的.mat文件
    for i in range(clean_data.shape[0]):
        file_name = os.path.join(save_dir, f'{i + 1}.mat')
        clean = clean_data[i, :].reshape(field, field)
        noisy = noisy_data[i, :].reshape(field, field)
        hdf5storage.savemat(file_name, dict(noisy=noisy, clean=clean))
