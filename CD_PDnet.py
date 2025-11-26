# _*_ coding: utf-8 _*_
import os
import glob
import numpy as np
import scipy.io as scio
import torch
from torch import nn
from torchvision.models import vgg16, VGG16_Weights
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from loading_data import *
import torch.nn.functional as F

"""
输入为 us = A*z
初始化参数: A, K, zs, P
各变量维度：
A -- M x N      A* -- N x M
z -- M x images_num 
us = A*z -- N x images_num

待学习的参数:
L -- (K, P, N), sigma --- (K, ), tao --- (K, )
Note: 每一层的数值都不相同。
"""
class InputLayer(nn.Module):
    def __init__(self) -> None:
        super(InputLayer, self).__init__()

        # A 和 A* 的卷积层
        self.conv_A = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(7, 7), stride=(1, 1), padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(7, 7), stride=(1, 1), padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=(7, 7), stride=(1, 1), padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=(7, 7), stride=(1, 1), padding=3, bias=True),
            nn.ReLU(inplace=True),
        )

        self.conv_A_conj = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(7, 7), stride=(1, 1), padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(7, 7), stride=(1, 1), padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=(7, 7), stride=(1, 1), padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=(7, 7), stride=(1, 1), padding=3, bias=True),
            nn.ReLU(inplace=True),
        )

        # 并行特征提取层（编码器）
        self.conv_L = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1), padding=2, bias=True),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(7, 7), stride=(1, 1), padding=3, bias=True),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(9, 9), stride=(1, 1), padding=4, bias=True),
                nn.ReLU(inplace=True),
            ),
            # nn.Sequential(
            #     nn.Conv2d(1, 32, kernel_size=(10, 10), stride=(1, 1), padding=5, bias=True),
            #     nn.ReLU(inplace=True)
            # )
        ])

        # 对应的解码器 (L*)
        self.conv_L_conj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(32, 1, kernel_size=(5, 5), stride=(1, 1), padding=2, bias=True),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(32, 1, kernel_size=(7, 7), stride=(1, 1), padding=3, bias=True),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(32, 1, kernel_size=(9, 9), stride=(1, 1), padding=4, bias=True),
                nn.ReLU(inplace=True),
            ),
        ])



        self.constrain_conv_layers()

    def constrain_conv_layers(self):


        for i, layer in enumerate(self.conv_A_conj):
            if isinstance(layer, nn.Conv2d):
                # 共轭和旋转Conv2d层的卷积核
                original_weights = self.conv_A[0].weight.data if i == 6 else self.conv_A[2].weight.data if i == 4 else \
                self.conv_A[4].weight.data if i == 2 else self.conv_A[6].weight.data
                # original_weights = self.conv_L[0].weight.data if i ==7  else self.conv_L[3].weight.data if i ==3  else self.conv_L[7].weight.data
                rotated_weights = torch.rot90(torch.conj(original_weights), k=2, dims=[2, 3])  # 180度旋转
                self.conv_A_conj[i].weight.data = rotated_weights.permute(1, 0, 2, 3)

    def forward(self, tau1, sigma1, zs, rho1):
        # zs = (batch_size, C=3, H=7, W=7)

        us = self.conv_A_conj(zs)  # x1 = A* zs
        x1 = us  # x1 __ (batch_size, 3, 7, 7)
        temp1 = self.conv_A(x1)  # temp1 = A x1  __ (batch_size, 3, 7, 7)
        temp2 = self.conv_A_conj(temp1)  # temp2 = A* A x1 __ (batch_size, C=3, 7, 7)

        # b1 和 b2 计算
        b1 = tau1 * self.conv_A_conj(zs)  # b1 = tao A* zs __ (batch_size, 1, 10, 10)

        # L 不需要合并操作，每个卷积层独立处理
        L_outs = [conv(self.conv_A_conj(zs)) for conv in self.conv_L]
        b2 = 2 * tau1 * sigma1 * torch.stack(L_outs, dim=1)
        # print(f'b2 shape: {b2.shape}')
        # 处理 D1_1 和 D2_1
        D1_1 = x1 - tau1 * temp2  # D1_1 __ (batch_size, 1, 10, 10)

        # 处理 D2_1
        L_outs_x1 = [conv(x1) for conv in self.conv_L]
        L_outs_temp2 = [conv(temp2) for conv in self.conv_L]
        # print(f'L_outs_x1 shape: {[x.shape for x in L_outs_x1]}')
        # print(f'L_outs_temp2 shape: {[x.shape for x in L_outs_temp2]}')

        D2_1 = sigma1 * torch.stack(L_outs_x1, dim=1) - 2 * tau1 * sigma1 * torch.stack(L_outs_temp2,
                                                                                        dim=1)
        # print(f'D2_1 shape: {D2_1.shape}')
        # 计算 xk_new 和 yk_new
        xk_new = D1_1 + b1  # xk __ (batch_size, 1, 10, 10)
        yk_new = torch.nn.functional.softshrink(D2_1 + b2, 0.5)
        # print(f'yk_new shape: {yk_new.shape}')
        # 更新 xk 和 yk
        xk = rho1 * xk_new + (1 - rho1) * b1
        yk = rho1 * yk_new + (1 - rho1) * b2

        return xk, yk



class MiddleLayer(nn.Module):
    def __init__(self, K, ) -> None:
        super(MiddleLayer, self).__init__()
        self.K = K

        self.conv_A = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(7, 7), stride=(1, 1), padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(7, 7), stride=(1, 1), padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=(7, 7), stride=(1, 1), padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=(7, 7), stride=(1, 1), padding=3, bias=True),
            nn.ReLU(inplace=True),
        )

        self.conv_A_conj = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(7, 7), stride=(1, 1), padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(7, 7), stride=(1, 1), padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=(7, 7), stride=(1, 1), padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=(7, 7), stride=(1, 1), padding=3, bias=True),
            nn.ReLU(inplace=True),
        )

        # 并行特征提取层（编码器）
        self.conv_L = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1), padding=2, bias=True),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(7, 7), stride=(1, 1), padding=3, bias=True),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(9, 9), stride=(1, 1), padding=4, bias=True),
                nn.ReLU(inplace=True),
            ),
        ])

        # 对应的解码器 (L*)
        self.conv_L_conj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(32, 1, kernel_size=(5, 5), stride=(1, 1), padding=2, bias=True),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(32, 1, kernel_size=(7, 7), stride=(1, 1), padding=3, bias=True),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(32, 1, kernel_size=(9, 9), stride=(1, 1), padding=4, bias=True),
                nn.ReLU(inplace=True),
            ),
        ])
        # 最后的合并层


        self.constrain_conv_layers()

    def constrain_conv_layers(self):
        for i, layer in enumerate(self.conv_L_conj):
            if isinstance(layer, nn.Conv2d):
                # 共轭和旋转Conv2d层的卷积核
                original_weights = self.conv_L[0].weight.data
                # original_weights = self.conv_L[0].weight.data if i ==7  else self.conv_L[3].weight.data if i ==3  else self.conv_L[7].weight.data
                rotated_weights = torch.rot90(torch.conj(original_weights), k=2, dims=[2, 3])  # 180度旋转
                self.conv_L_conj[i].weight.data = rotated_weights.permute(1, 0, 2, 3)
        for i, layer in enumerate(self.conv_A_conj):
            if isinstance(layer, nn.Conv2d):
                # 共轭和旋转Conv2d层的卷积核
                original_weights = self.conv_A[0].weight.data if i == 6 else self.conv_A[2].weight.data if i == 4 else \
                    self.conv_A[4].weight.data if i == 2 else self.conv_A[6].weight.data
                # original_weights = self.conv_L[0].weight.data if i ==7  else self.conv_L[3].weight.data if i ==3  else self.conv_L[7].weight.data
                rotated_weights = torch.rot90(torch.conj(original_weights), k=2, dims=[2, 3])  # 180度旋转
                self.conv_A_conj[i].weight.data = rotated_weights.permute(1, 0, 2, 3)

        # 对conv_A和conv_A_conj应用约束
        # rotated_conj_weights = torch.rot90(torch.conj(self.conv_A.weight.data), k=2, dims=[2, 3])
        # self.conv_A_conj.weight.data = rotated_conj_weights.permute(1, 0, 2, 3)

    def forward(self, xk, yk, tau, sigma, rho, zs):
        for k in range(1, self.K + 1):
            temp1 = self.conv_A(xk)
            temp2 = self.conv_A_conj(temp1)


            yk_5x5,yk_7x7,yk_9x9 = torch.split(yk, 1, dim=1)

            yk_5x5 = yk_5x5.squeeze(1)
            yk_7x7 = yk_7x7.squeeze(1)
            yk_9x9 = yk_9x9.squeeze(1)
            # 解码器部分
            out_5x5 = self.conv_L_conj[0](yk_5x5)
            out_7x7 = self.conv_L_conj[0](yk_7x7)
            out_9x9 = self.conv_L_conj[0](yk_9x9)


            temp3 = out_5x5+out_7x7+out_9x9

            b1 = tau[k] * self.conv_A_conj(zs)

            # L 特征提取并合并
            L_outs = [conv(self.conv_A_conj(zs)) for conv in self.conv_L]
            b2 = 2 * tau[k] * sigma[k] * torch.stack(L_outs, dim=1)



            # 计算 D1 和 D2
            D1_1 = xk - tau[k] * temp2
            D1_2 = - tau[k] * temp3
            L_outs_xk = [conv(xk) for conv in self.conv_L]
            L_outs_temp2 = [conv(temp2) for conv in self.conv_L]
            L_outs_temp3 = [conv(temp3) for conv in self.conv_L]
            D2_1 = sigma[k] * torch.stack(L_outs_xk, dim=1) - 2 * tau[k] * sigma[k] * torch.stack(L_outs_temp2, dim=1)

            D2_2 = yk - 2 * tau[k] * sigma[k] *  torch.stack(L_outs_temp3, dim=1)

            # 计算 xk_new 和 yk_new
            xk_new = D1_1 + D1_2 + b1
            yk_new = torch.nn.functional.softshrink(D2_1 + D2_2 + b2, 0.5)

            # 更新 xk 和 yk
            xk = rho[k] * xk_new + (1 - rho[k]) * xk
            yk = rho[k] * yk_new + (1 - rho[k]) * yk

        return xk, yk


class LastLayer(nn.Module):
    def __init__(self) -> None:
        super(LastLayer, self).__init__()
        # #bsd
        # self.conv_A = nn.Conv2d(1,1 , kernel_size=(5, 5), stride=(1, 1), padding=2, bias=True)
        # self.conv_A_conj = nn.Conv2d(1, 1, kernel_size=(5, 5), stride=(1, 1), padding=2, bias=True)
        # minist
        # self.conv_A = nn.Conv2d(1, 1, kernel_size=(7, 7), stride=(1, 1), padding=3, bias=True)
        # self.conv_A_conj = nn.Conv2d(1, 1, kernel_size=(7, 7), stride=(1, 1), padding=3, bias=True)
        self.conv_A = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(7, 7), stride=(1, 1), padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(7, 7), stride=(1, 1), padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=(7, 7), stride=(1, 1), padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=(7, 7), stride=(1, 1), padding=3, bias=True),
            nn.ReLU(inplace=True),
        )

        self.conv_A_conj = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(7, 7), stride=(1, 1), padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(7, 7), stride=(1, 1), padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=(7, 7), stride=(1, 1), padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=(7, 7), stride=(1, 1), padding=3, bias=True),
            nn.ReLU(inplace=True),
        )

        # 并行特征提取层（编码器）
        self.conv_L = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1), padding=2, bias=True),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(7, 7), stride=(1, 1), padding=3, bias=True),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(9, 9), stride=(1, 1), padding=4, bias=True),
                nn.ReLU(inplace=True),
            ),
            # nn.Sequential(
            #     nn.Conv2d(1, 32, kernel_size=(10, 10), stride=(1, 1), padding=5, bias=True),
            #     nn.ReLU(inplace=True)
            # )
        ])

        # 对应的解码器 (L*)
        self.conv_L_conj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(32, 1, kernel_size=(5, 5), stride=(1, 1), padding=2, bias=True),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(32, 1, kernel_size=(7, 7), stride=(1, 1), padding=3, bias=True),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(32, 1, kernel_size=(9, 9), stride=(1, 1), padding=4, bias=True),
                nn.ReLU(inplace=True),
            ),
        ])

        # 最后的合并层
        # self.merge_conv = nn.Sequential(
        #     nn.Conv2d(3, 1, kernel_size=(1, 1), stride=(1, 1), bias=True),  # 将 3 个输出通道合并成 1 个通道
        #     nn.ReLU(inplace=True),
        # )

        self.constrain_conv_layers()

    def constrain_conv_layers(self):
        # 定义 L 和 L* 之间的卷积层对应关系
        for i, layer in enumerate(self.conv_L_conj):
            if isinstance(layer, nn.Conv2d):
                # 共轭和旋转Conv2d层的卷积核
                original_weights = self.conv_L[0].weight.data
                # original_weights = self.conv_L[0].weight.data if i ==7  else self.conv_L[3].weight.data if i ==3  else self.conv_L[7].weight.data
                rotated_weights = torch.rot90(torch.conj(original_weights), k=2, dims=[2, 3])  # 180度旋转
                self.conv_L_conj[i].weight.data = rotated_weights.permute(1, 0, 2, 3)

        for i, layer in enumerate(self.conv_A_conj):
            if isinstance(layer, nn.Conv2d):
                # 共轭和旋转Conv2d层的卷积核
                original_weights = self.conv_A[0].weight.data if i == 6 else self.conv_A[2].weight.data if i == 4 else \
                    self.conv_A[4].weight.data if i == 2 else self.conv_A[6].weight.data
                # original_weights = self.conv_L[0].weight.data if i ==7  else self.conv_L[3].weight.data if i ==3  else self.conv_L[7].weight.data
                rotated_weights = torch.rot90(torch.conj(original_weights), k=2, dims=[2, 3])  # 180度旋转
                self.conv_A_conj[i].weight.data = rotated_weights.permute(1, 0, 2, 3)

        # # 对conv_A和conv_A_conj应用约束
        # rotated_conj_weights = torch.rot90(torch.conj(self.conv_A.weight.data), k=2, dims=[2, 3])
        # self.conv_A_conj.weight.data = rotated_conj_weights.permute(1, 0, 2, 3)

    # def forward(self, xk, yk, tau, sigma, rho, zs):
    #     temp1 = self.conv_A(xk)
    #     temp2 = self.conv_A_conj(temp1)
    #     temp3 = self.conv_L_conj(yk)
    #     b1 = tau[-1] * self.conv_A_conj(zs)
    #     b2 = 2 * tau[-1] * sigma[-1] * self.conv_L(self.conv_A_conj(zs))
    #     D1_1 = xk - tau[-1] * temp2
    #     D1_2 = - tau[-1] * self.conv_L_conj(yk)
    #
    #     D2_1 = sigma[-1] * self.conv_L(xk) - 2 * tau[-1] * sigma[-1] * self.conv_L(temp2)
    #     D2_2 = yk - 2 * tau[-1] * sigma[-1] * self.conv_L(temp3)
    #
    #     # xk = D1_1 + D1_2 + b1
    #
    #     xk_new = D1_1 + D1_2 + b1
    #     ykn_new = torch.nn.functional.softshrink(D2_1 + D2_2 + b2, 0.5)
    #     # 更新 xk
    #     xk = rho[-1] * xk_new + (1 - rho[-1]) * xk
    #     ykn = rho[-1] * ykn_new + (1 - rho[-1]) * yk
    #
    #
    #
    #     return xk
    def forward(self, xk, yk, tau, sigma, rho, zs):

            temp1 = self.conv_A(xk)
            temp2 = self.conv_A_conj(temp1)

            yk_5x5,yk_7x7,yk_9x9 = torch.split(yk, 1, dim=1)

            yk_5x5 = yk_5x5.squeeze(1)
            yk_7x7 = yk_7x7.squeeze(1)
            yk_9x9 = yk_9x9.squeeze(1)
            # 解码器部分
            out_5x5 = self.conv_L_conj[0](yk_5x5)
            out_7x7 = self.conv_L_conj[0](yk_7x7)
            out_9x9 = self.conv_L_conj[0](yk_9x9)


            temp3 = out_5x5+out_7x7+out_9x9

            b1 = tau[-1] * self.conv_A_conj(zs)

            # L 特征提取并合并
            L_outs = [conv(self.conv_A_conj(zs)) for conv in self.conv_L]
            b2 = 2 * tau[-1] * sigma[-1] * torch.stack(L_outs, dim=1)

            # 计算 D1 和 D2
            D1_1 = xk - tau[-1] * temp2
            D1_2 = - tau[-1] * temp3
            L_outs_xk = [conv(xk) for conv in self.conv_L]
            L_outs_temp2 = [conv(temp2) for conv in self.conv_L]
            L_outs_temp3 = [conv(temp3) for conv in self.conv_L]
            D2_1 = sigma[-1] * torch.stack(L_outs_xk, dim=1) - 2 * tau[-1] * sigma[-1] * torch.stack(L_outs_temp2, dim=1)

            D2_2 = yk - 2 * tau[-1] * sigma[-1] * torch.stack(L_outs_temp3, dim=1)

            # 计算 xk_new 和 yk_new
            xk_new = D1_1 + D1_2 + b1
            yk_new = torch.nn.functional.softshrink(D2_1 + D2_2 + b2, 0.5)

            # 更新 xk 和 yk
            xk = rho[-1] * xk_new + (1 - rho[-1]) * xk
            yk = rho[-1] * yk_new + (1 - rho[-1]) * yk

            return xk


class DeepPDNet(nn.Module):
    def __init__(self, K) -> None:
        super(DeepPDNet, self).__init__()
        self.inputlayer = InputLayer()
        self.middlelayer = MiddleLayer(K)
        self.outputlayer = LastLayer()

        tau_ini = 1
        self.tau = nn.parameter.Parameter((torch.ones(K + 1, 1)) * tau_ini)
        self.sigma = nn.parameter.Parameter(torch.as_tensor(torch.randn(K + 1), dtype=torch.float))
        self.rho = nn.parameter.Parameter(torch.full((K + 1,), 0.5), requires_grad=True)

        self.previous_rho = self.rho.data.clone()

    def forward(self, zs):
        x2, y2 = self.inputlayer(self.tau[0], self.sigma[0], zs, self.rho[0])

        xk, yk = self.middlelayer(x2, y2, self.tau, self.sigma, self.rho, zs)
        result = self.outputlayer(xk, yk, self.tau, self.sigma, self.rho, zs)

        return result

    def update_rho(self):
        with torch.no_grad():
            if torch.any((self.rho < 0) | (self.rho > 1)):
                self.rho.data.copy_(self.previous_rho)
            else:
                self.previous_rho.copy_(self.rho.data)


# MSE
def deeppdnet_loss(x_origin: torch.Tensor, z_restored: torch.Tensor) -> torch.Tensor:
    # x_origin -- (batch_size, C=3, H=28, W=28), z_restore -- (batch_size, C=3, H=28, W=28)
    images_num = x_origin.shape[0]
    return (1 / images_num) * torch.sum(torch.square(x_origin - z_restored))



# 定义参数初始化函数
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
