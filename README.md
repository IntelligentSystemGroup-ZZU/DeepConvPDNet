# Deep Convolutional Primal-Dual Network for Image Deblurring

Mingyuan Jiu, Mingjing Peng, Fanfan Zhang, Shupan Li, Hongru Zhao, Rongrong Ji, Mingliang Xu


School of Computer and Artificial Intelligence, Zhengzhou University


## Abstract
Image deblurring is a challenging image task, which
 is regarded as a classical inverse problem. Deep primal-dual
 proximal network (DeepPDNet) is recently proposed which un
rolls the Condat-V˜ u primal-dual splitting algorithm as a feed
forward network and it has demonstrated excellent restoration
 performance. However, the feature patterns in the DeepPDNet are
 well manually designed and thus the network is not implemented
 in an efficient convolutional fashion. In this work, we revisit the
 DeepPDNet and extend it in three respects: i) the convolution and
 pooling operators as well as their associating adjoint operations
 are studied in the primal-dual algorithm, and then a deep
 convolutional primal-dual network (DeepConvPDNet) and its full
 variant with skips are proposed to preserve the optimization
 consistence of primal-dual Conda-V˜ u algorithm; ii) two (cascade
 vs parallel) variants of the networks are designed according to the
 structure of convolutional kernels; iii) rather than that the blur
 kernels are given as prior knowledge, they can be encoded by a
 set of convolutional layers and deconvolutional layers for their
 conjugate, resulting to a full learnable deep convolutional primal
dual neural network. We investigate the proposed networks on the
 MNIST dataset, the grayscale and color version of BSD dataset
 and GoPro dataset for image deblurring. Extensive experiments
 are conducted to validate the performance of the proposed
 networks, and promising results in term of PSNR and SSIM
 are obtained in comparison with twelve methods including state
of-the-art methods (e.g. Restormer, DRUNet, and DeblurGAN),
 which validated its effectiveness.
