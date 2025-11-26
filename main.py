import argparse
from math import inf
import csv

from DeepPDNet import DeepPDNet, deeppdnet_loss, weights_init_normal
from utils import LambdaLR, psnr, n_ssim, ssim
from matplotlib import pyplot as plt
from torch import optim

import time
import torch.utils.data as td
from loading_data import *
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
import os
import scipy.io as sio

#  os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# torch.manual_seed(330)
# np.random.seed(330)

parser = argparse.ArgumentParser()
parser.add_argument("--start_epoch", type=int, default=0,help="epoch to start training from")
parser.add_argument("--num_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--root_dir", type=str, default="data/noisy_images/blur3_sig20", help="root_dir")  # mnist
parser.add_argument('--dataroot', type=str, default='data/noisy_images/blur3_sig20',
                    help='root directory of the dataset')  # 测试
# parser.add_argument("--root_dir", type=str, default="data/noisy_images_bsd/blur5_sig50",
#                     help="root_dir")  # BSDS500
parser.add_argument("--dataset_name", type=str, default="MNIST",
                    help="name of the dataset")  # parser.add_argument("--dataset_name", type=str, default="BSD", help="name of the dataset")
parser.add_argument("--K", type=int, default=9, help="layer size - 1")
# parser.add_argument("--P", type=int, default=100, help="a fixed embedded feature number")   # mnist数据集使用
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument('--test_batch_size', type=int, default=1, help='size of the batches for testing')
parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
# parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")

parser.add_argument("--decay_epoch", type=int, default=30, help="epoch from which to start lr decay")
parser.add_argument("--image_size", type=tuple, default=(28, 28))  # mnist
parser.add_argument('--test_image_size', type=tuple, default=(28, 28))  # mnist
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
# parser.add_argument("--image_size", type=tuple, default=(321, 481))  # BSDS500
# parser.add_argument('--test_image_size', type=tuple, default=(321, 481))  # BSDS500
# parser.add_argument("--channels", type=int, default=1, help="number of image channels")  # BSDS500
# parser.add_argument("--sample_interval", type=int, default=10000, help="interval between saving outputs")
parser.add_argument("--blur_size", type=int, default=3, help="blur size")
parser.add_argument("--sigma", type=int, default=20, help="Gaussian noise")

opt = parser.parse_args()
print(opt)

# 创建文件夹s
os.makedirs("train_h5l2/blur%d_sig%d/epoch%d_b%d/images" % (opt.blur_size, opt.sigma, opt.num_epochs, opt.batch_size),
            exist_ok=True)

os.makedirs("train_h5l2/blur%d_sig%d/epoch%d_b%d/data" % (opt.blur_size, opt.sigma, opt.num_epochs, opt.batch_size),
            exist_ok=True)

os.makedirs("save_h5l2/blur%d_sig%d/epoch%d_b%d" % (opt.blur_size, opt.sigma, opt.num_epochs, opt.batch_size), exist_ok=True)

train_image_dir = 'train_h5l2/blur%d_sig%d/epoch%d_b%d/images' % (
    opt.blur_size, opt.sigma, opt.num_epochs, opt.batch_size)

train_data_dir = 'train_h5l2/blur%d_sig%d/epoch%d_b%d/data' % (
    opt.blur_size, opt.sigma, opt.num_epochs, opt.batch_size)
# train_params_dir = 'train/blur%d_sig%d/epoch%d_b%d/data/params' % (
#     opt.blur_size, opt.sigma, opt.num_epochs, opt.batch_size)

save_model_dir = 'save_h5l2/blur%d_sig%d/epoch%d_b%d' % (opt.blur_size, opt.sigma, opt.num_epochs, opt.batch_size)

# datasets
train_set = NoisyIMGDataset(opt.root_dir, mode='train_noise', image_size=opt.image_size, blur_size=opt.blur_size,
                            sigma=opt.sigma)
val_set = NoisyIMGDataset(opt.root_dir, mode='val_noise', image_size=opt.image_size, blur_size=opt.blur_size,
                          sigma=opt.sigma)
test_set = NoisyIMGDataset(opt.dataroot, mode='test_noise', image_size=opt.image_size,
                           blur_size=opt.blur_size, sigma=opt.sigma)

# Define data loaders
train_loader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size, shuffle=True,
                                           drop_last=True, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=opt.batch_size, shuffle=False,
                                         drop_last=True, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=opt.test_batch_size, shuffle=False,
                                          drop_last=False, pin_memory=True)
# # 构建测试数据集
# test_set = NoisyMNISTDataset(opt.dataroot, mode='test_noise', image_size=opt.image_size,
#                              blur_size=opt.blur_size, sigma=opt.sigma)
# test_loader = torch.utils.data.DataLoader(test_set, batch_size=opt.test_batch_size, shuffle=False,
#                                           drop_last=False, pin_memory=True)
# # cpu
# # 模型实例化
# model = DeepPDNet(opt.K)

# gpu单卡运行
device = torch.device("cuda:0")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 模型实例化
model = DeepPDNet(opt.K)
# 使用cuda
model = model.to(device)

# # gpu多卡并行
# device_ids = [i for i in range(torch.cuda.device_count())]
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = DeepPDNet(opt.K)
# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUS!")
#     model = nn.DataParallel(model, device_ids=device_ids)
#     model = model.to(device)

# 定义损失函数和优化器
criterion = deeppdnet_loss
# def calculate_loss(clean, restored, model, lambda_reg=0.001):
#     mse_loss = criterion(clean, restored)  # 使用你之前定义的损失函数
#     l2_reg = torch.tensor(0., device=device)
#     for param in model.parameters():
#         l2_reg += torch.norm(param, p=2)  # 计算所有参数的 L2 范数
#     total_loss = mse_loss + lambda_reg * l2_reg
# optimizer = optim.Adam(model.parameters(), lr=opt.lr)
optimizer = optim.Adam(model.parameters(), lr=opt.lr)

# 学习率更新进程
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=LambdaLR(opt.num_epochs, opt.start_epoch, opt.decay_epoch).step
)

# 如果epoch == 0，初始化模型参数; 如果epoch == n, 载入训练到第n轮的预训练模型
if opt.start_epoch != 0:
    # Load pretrained models载入训练到第n轮的预训练模型
    model.load_state_dict(
        torch.load(os.path.join(save_model_dir, '%s_%d.pth' % (opt.dataset_name, opt.start_epoch))))
    # model.load_state_dict(
    #     torch.load("save/blur%d_sig%d/epoch%d_b%d/%s_%d.pth" % (
    #         opt.blur_size, opt.sigma, opt.num_epochs, opt.batch_size, opt.dataset_name, opt.start_epoch)))
else:
    # Initialize weights初始化模型参数
    model.apply(weights_init_normal)


# # 保存参数到csv文件中
# def save_params(epoch):
#     with open('train/blur%d_sig%d/epoch%d_b%d/data/train_result_para.csv' % (
#             opt.blur_size, opt.sigma, opt.num_epochs, opt.batch_size), 'a', newline='') as f:
#         csv_writer = csv.writer(f)
#         title = ['epoch', 'conv_A_params', 'conv_A_conj_params', 'conv_L_params', 'conv_L_conj_params', 'tau_params',
#                  'sigma_params']
#         if epoch == 0:
#             csv_writer.writerow(title)
#         conv_A_params = model.state_dict()['conv_A.weight']
#         conv_A_conj_params = model.state_dict()['conv_A_conj.weight']
#         conv_L_params = model.state_dict()['conv_L.weight']
#         conv_L_conj_params = model.state_dict()['conv_L_conj.weight']
#         tau_params = model.tau
#         sigma_params = model.sigma
#         # 将数据保存到csv文件中
#         csv_writer.writerow([epoch + 1, conv_A_params, conv_A_conj_params, conv_L_params, conv_L_conj_params,
#                              tau_params, sigma_params])

# 将参数提取出来
def save_params():
    conv_A_params = model.state_dict()['conv_A.weight'].data.detach().cpu().numpy()
    conv_A_conj_params = model.state_dict()['conv_A_conj.weight'].data.detach().cpu().numpy()
    conv_L_params = model.state_dict()['conv_L.weight'].data.detach().cpu().numpy()
    conv_L_conj_params = model.state_dict()['conv_L_conj.weight'].data.detach().cpu().numpy()
    tau_params = model.tau.data.detach().cpu().numpy()
    sigma_params = model.sigma.data.detach().cpu().numpy()
    rho_params = model.sigma.data.detach().cpu().numpy()
    return conv_A_params, conv_A_conj_params, conv_L_params, conv_L_conj_params, tau_params, sigma_params, rho_params


# 每一轮打印一次图片
# def sample_images(clean, noisy, restored, epoch):
#     """Saves a generated sample from the test set"""
#     # Arange images along x-axis
#     # make_grid():用于把几个图像按照网格排列的方式绘制出来
#     clean = make_grid(clean, nrow=16, normalize=True)
#     noisy = make_grid(noisy, nrow=16, normalize=True)
#     restored = make_grid(restored, nrow=16, normalize=True)
#     # Arange images along y-axis
#     # 把以上图像都拼接起来，保存为一张大图片
#     image_grid = torch.cat((clean, noisy, restored), 1)
#     save_image(image_grid, os.path.join(train_image_dir, '%s.png' % (epoch + 1)), normalize=False)

def sample_images(clean, noisy, restored, yk, epoch, image_folder):
    """Saves images from the batch."""
    # Make grids for clean, noisy, restored, and yk images
    clean_grid = make_grid(clean, nrow=8, normalize=True)
    noisy_grid = make_grid(noisy, nrow=8, normalize=True)
    restored_grid = make_grid(restored, nrow=8, normalize=True)
    yk_grid = make_grid(yk, nrow=8, normalize=True)

    # Save images for each type
    save_image(clean_grid, os.path.join(image_folder, f"{epoch}_clean.png"))
    save_image(noisy_grid, os.path.join(image_folder, f"{epoch}_noisy.png"))
    save_image(restored_grid, os.path.join(image_folder, f"{epoch}_restored.png"))
    save_image(yk_grid, os.path.join(image_folder, f"{epoch}_yk.png"))


def calculate_norm_difference(tensor1, tensor2):
    """
    计算两个张量之间的Frobenius范数差异。

    参数:
    tensor1 (torch.Tensor): 第一个输入张量。
    tensor2 (torch.Tensor): 第二个输入张量。

    返回:
    torch.Tensor: 两个张量之间的Frobenius范数差异。
    """
    norm1 = torch.norm(tensor1 - tensor2, p='fro')  # 计算tensor1的Frobenius范数
    return torch.abs(norm1)  # 返回范数差异的绝对值


def train():
    # 将训练过程中的计算结果输出
    # with open(os.path.join(train_data_dir, 'train.csv'), 'w', newline='') as f:
    with open(os.path.join(train_data_dir, 'train_val_result.csv'), 'w', newline='') as f:
        csv_writer = csv.writer(f)
        # title = ['Epoch', 'train_time', 'train_loss', 'tr_PSNR', 'tr_SSIM']
        title = ['Epoch', 'train_time', 'train_loss', 'val_loss', 'tr_PSNR', 'tr_SSIM', 'val_PSNR', 'val_SSIM',
                 'test_PSNR', 'test_SSIM']
        csv_writer.writerow(title)
        # 防止过拟化
        Earlystop = True
        DecideEarlystop = []
        expect = inf
        # train_loss, val_loss存储到列表，画图
        train_loss_list = []
        val_loss_list = []
        train_loss = 0
        # # 将训练过程中的参数保存至列表
        # conv_A_params_list, conv_A_conj_params_list, conv_L_params_list = [], [], []
        # conv_L_conj_params_list, tau_params_list, sigma_params_list = [], [], []
        # 计算评估指标
        tr_PSNR = 0
        tr_SSIM = 0
        print("Start/Continue training from epoch {}".format(opt.start_epoch))
        for epoch in range(opt.start_epoch, opt.num_epochs):
            prev_time = time.time()
            for noisy, clean, name_im in train_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                noisy = torch.as_tensor(noisy, dtype=torch.float32)
                clean = torch.as_tensor(clean, dtype=torch.float32)

                model.train()
                restored = model(noisy)  # Assuming the model returns both 'restored' and 'yk'

                # Clamp values to range [0, 255]
                clean = torch.clamp(clean, 0, 255)
                restored = torch.clamp(restored, 0, 255)

                # Calculate PSNR and SSIM
                tr_PSNR = psnr(clean, restored)
                tr_SSIM = n_ssim(clean, restored)

                loss = criterion(clean, restored)
                train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                model.update_rho()
            norm_difference_restored = calculate_norm_difference(restored, clean)

            # sample_images(clean, noisy, restored, yk, epoch, train_image_dir)

            train_loss /= len(train_loader)
            if train_loss < expect:
                expect = train_loss
                expect = train_loss
                # torch.save(model.state_dict(), "save/%s/deeppdnet_%d.pth" % (opt.dataset_name, epoch))
                torch.save(model.state_dict(),
                           os.path.join(save_model_dir, '%s_%d.pth' % (opt.dataset_name, epoch + 1)))
                # torch.save(model.state_dict(),
                #            "save/blur%d_sig%d/epoch%d_b%d/%s.pth" % (
                #                opt.blur_size, opt.sigma, opt.num_epochs, opt.batch_size, opt.dataset_name))
            # Early stop 防止模型过拟合
            if (epoch + 1) >= 10:
                if Earlystop:
                    back = DecideEarlystop[epoch - 8:]
                    front = DecideEarlystop[epoch - 9:epoch]
                    decidevar = np.array(front) - np.array(back)
                    if sum(decidevar >= 0) < sum(decidevar < 0):
                        break

            # train_time = time.time() - prev_time

            # # 保存每一轮的参数到csv文件中
            # save_params(epoch)

            # 验证部分
            val_loss, val_PSNR, val_SSIM = val(model, val_loader, criterion)
            test_PSNR, test_SSIM = test(model, test_loader)
            train_time = time.time() - prev_time

            print(
                "\r[Epoch %d/%d] [train_time: %.4f] [train loss: %.6f] [val loss: %.6f] [tr_PSNR: %.2f] [tr_SSIM: %.4f]"
                " [val_PSNR: %.2f] [val_SSIM: %.4f] [test_PSNR: %.2f] [test_SSIM: %.4f][norm: %.6f]" % (
                    epoch + 1,
                    opt.num_epochs,
                    train_time,
                    train_loss,
                    val_loss,
                    tr_PSNR,
                    tr_SSIM,
                    val_PSNR,
                    val_SSIM,
                    test_PSNR,
                    test_SSIM, norm_difference_restored
                )
            )
            csv_writer.writerow(
                [epoch + 1, train_time, train_loss, val_loss, tr_PSNR, tr_SSIM, val_PSNR, val_SSIM, test_PSNR,
                 test_SSIM, norm_difference_restored
                 ])
            # 更新学习率
            lr_scheduler.step()

            # # 在每个epoch结束后，将当前模型的参数存储到.mat文件中
            # conv_A_params, conv_A_conj_params, conv_L_params, conv_L_conj_params, tau_params, sigma_params = save_params()
            # params = {'A': conv_A_params, 'A*': conv_A_conj_params, 'L': conv_L_params,
            #           'L*': conv_L_conj_params, 'tau': tau_params, 'sigma': sigma_params}
            # sio.savemat(os.path.join(train_params_dir, 'model_params%d.mat' % (epoch + 1)), params, appendmat=True)

            # # 将参数存入列表
            # conv_A_params, conv_A_conj_params, conv_L_params, conv_L_conj_params, tau_params, sigma_params = save_params()
            # conv_A_params_list.append(conv_A_params)
            # conv_A_conj_params_list.append(conv_A_conj_params)
            # conv_L_params_list.append(conv_L_params)
            # conv_L_conj_params_list.append(conv_L_conj_params)
            # tau_params_list.append(tau_params)
            # sigma_params_list.append(sigma_params)

            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)

        # # 将列表中的参数最后统一存放到.mat文件中
        # sio.savemat(os.path.join(train_data_dir, f'train_params.mat'),
        #             dict(conv_A_params=conv_A_params_list, conv_A_conj_params=conv_A_conj_params_list,
        #                  conv_L_params=conv_L_params_list,
        #                  conv_L_conj_params=conv_L_conj_params_list, tau_params=tau_params_list,
        #                  sigma_params=sigma_params_list))

        # # 训练结束后，保存模型
        torch.save(model.state_dict(), os.path.join(save_model_dir, '%s_%d.pth' % (opt.dataset_name, epoch + 1)))
        # torch.save(model.state_dict(),
        #            "save/blur%d_sig%d/epoch%d_b%d/%s_%d.pth" % (
        #                opt.blur_size, opt.sigma, opt.num_epochs, opt.batch_size, opt.dataset_name, epoch + 1))
        print("\nsave my model finished !!")
        plt.plot(train_loss_list, label='Train Loss')
        plt.plot(val_loss_list, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


# 定义验证函数
def val(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    val_PSNR = 0
    val_SSIM = 0
    # prev_time = time.time()  # 开始时间
    with torch.no_grad():
        for noisy, clean, name_im in val_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            noisy = torch.as_tensor(noisy, dtype=torch.float32)
            clean = torch.as_tensor(clean, dtype=torch.float32)

            # restored, ykn, real_output, fake_output = model(noisy)
            restored = model(noisy)
            # real_xk_pred = D(clean)
            # real_yk_pred = D(noisy)
            restored = torch.clamp_(restored, 0, 255)
            # yk = torch.clamp(ykn, 0, 255)
            # loss = criterion(clean, restored, noisy, yk, real_xk_pred, real_output, real_yk_pred, fake_output)
            loss = criterion(clean, restored)
            val_loss += loss.item()
            # 计算评估指标
            val_PSNR = psnr(clean, restored)
            val_SSIM = n_ssim(clean, restored)
        val_loss /= len(val_loader)
    # val_time = time.time() - prev_time
    # print("Epoch {} | Time: {:.2f}s | Val Loss: {:.6f}".format(epoch, test_time, val_loss))
    return val_loss, val_PSNR, val_SSIM


# 测试函数
def test(model, test_loader):
    model.eval()
    with torch.no_grad():
        total_PSNR = 0.0
        total_SSIM = 0.0
        total_images = 0

        for i, data in enumerate(test_loader):
            noisy, clean, name_im = data
            noisy, clean = noisy.to(device), clean.to(device)
            noisy = torch.as_tensor(noisy, dtype=torch.float32)
            clean = torch.as_tensor(clean, dtype=torch.float32)
            restored = model(noisy)

            # restored, ykn, real_output, fake_output = model(noisy)
            restored = torch.clamp_(restored, 0, 255)

            norm_difference_restored = calculate_norm_difference(restored, clean)
            # 计算评估指标
            PSNR = psnr(clean, restored)
            SSIM = ssim(clean, restored)

            total_PSNR += PSNR.item()
            total_SSIM += SSIM.item()
            total_images += noisy.size(0)

        # 计算平均值
        average_PSNR = total_PSNR / total_images
        average_SSIM = total_SSIM / total_images

    return average_PSNR, average_SSIM


if __name__ == '__main__':
    train()