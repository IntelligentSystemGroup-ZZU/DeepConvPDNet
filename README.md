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

| Method               | Blur 3×3 (α=10) | Blur 3×3 (α=20) | Blur 3×3 (α=30) | Blur 5×5 (α=20) | Blur 7×7 (α=20) |
|----------------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| EPLL                 | 20.340/0.7923   | 19.990/0.7008   | 19.060/0.6871   | 16.420/0.5645   | 13.970/0.3268   |
| TV                   | 22.770/0.7964   | 20.570/0.7623   | 18.860/0.6881   | 18.430/0.6728   | 16.330/0.5678   |
| NLTv                 | 22.030/0.7942   | 21.980/0.7737   | 20.740/0.7354   | 19.530/0.7337   | 16.800/0.6017   |
| MWCNN                | 19.320/0.7007   | 19.260/0.6969   | 18.730/0.6666   | 15.150/0.4327   | 13.360/0.2316   |
| IRCNN                | 23.400/0.8049   | 24.710/0.8448   | 22.330/0.7668   | 21.290/0.7645   | 18.290/0.6551   |
| DeblurGAN-v3         | 23.830/0.9836   | 23.610/0.9124   | 21.760/0.9171   | 19.570/0.8636   | 15.930/0.7018   |
| DRUNet               | 27.460/0.9781   | 27.25/0.9770    | 25.490/0.9651   | 24.260/0.9522   | 22.01/0.9176    |
| Restormer            | 27.690/0.9656   | 27.460/0.9645   | 25.680/0.9528   | 24.130/0.9420   | 21.120/0.9105   |
| DeepScP-GD           | 25.340/0.9313   | 25.210/0.9328   | 23.450/0.9045   | 20.800/0.8423   | 17.460/0.7065   |
| DeepPDNet            | 26.390/0.9502   | 26.190/0.9483   | 24.650/0.9291   | 23.180/0.9008   | 21.110/0.8417   |
| Full DeepPDNet       | 26.500/0.9527   | 26.200/0.9504   | 24.660/0.9308   | 23.210/0.8993   | 21.170/0.8459   |
| DeepCPNet            | 26.530/0.9569   | 26.360/0.9556   | 24.790/0.9375   | 23.280/0.9072   | 21.140/0.8480   |
| Full DeepCPNet       | 26.550/0.9527   | 26.380/0.9548   | 24.760/0.9322   | 23.270/0.9108   | 21.100/0.8554   |
| ED_PDNet             | 26.830/0.9717   | 26.590/0.9702   | 25.010/0.9580   | 23.760/0.9461   | 21.710/0.9159   |
| DeepParConvPDNet     | 26.720/0.9711   | 26.570/0.9702   | 25.050/0.9582   | 23.750/0.9455   | 21.600/0.9129   |
| DeepConvPDNet        | 27.300/0.9746   | 27.100/0.9733   | 25.390/0.9608   | 23.840/0.9464   | 21.330/0.9082   |
| Blind DeepConvPDNet  | 27.430/0.9749   | 27.200/0.9736   | 25.520/0.9619   | 24.140/0.9500   | 22.030/0.9212   |
| Blind DeepParConvPDNet | 27.500/0.9753 | 27.370/0.9737   | 25.590/0.9627   | 24.310/0.9518   | 22.130/0.9232   |
