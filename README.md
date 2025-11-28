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


## Network Architecture

<img width="1019" height="571" alt="image" src="https://github.com/user-attachments/assets/bc747478-5f2a-45aa-844d-4977b0f1b567" />



## Experimental Results

### MNIST

COMPARISON RESULTS (PSNR/SSIM) OF DIFFERENT METHODS ON MNIST DATASETS FROM DIFFERENT DEGRADATION CONFIGURATIONS.

<img width="854" height="420" alt="image" src="https://github.com/user-attachments/assets/48f2f20c-2c3a-4c35-9cdc-55be98fe3eae" />

<img width="1106" height="485" alt="image" src="https://github.com/user-attachments/assets/b03de3ad-3dd9-458e-b882-fdc35e072f44" />


### BSD68

COMPARISON RESULTS (PSNR/SSIM) OF DIFFERENT METHODS ON BSD68 DATASETS FROM DIFFERENT DEGRADATION CONFIGURATIONS.

<img width="1095" height="329" alt="image" src="https://github.com/user-attachments/assets/116e93d2-58d6-4bff-b482-b271ba4ad038" />

<img width="1170" height="689" alt="image" src="https://github.com/user-attachments/assets/d1017197-c6fa-4983-ab3a-858c9c0d7368" />


### BSD100

COMPARISON RESULTS (PSNR/SSIM) OF DIFFERENT METHODS ON BSD100 DATASETS FROM DIFFERENT DEGRADATION CONFIGURATIONS.

<img width="1094" height="136" alt="image" src="https://github.com/user-attachments/assets/d2685784-5ffe-48c5-a220-c2d3d63367d3" />

<img width="1165" height="569" alt="image" src="https://github.com/user-attachments/assets/2473df89-665a-40e0-a169-f365ed53fdc0" />
