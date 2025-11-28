# Deep Convolutional Primal-Dual Network for Image Deblurring

Mingyuan Jiu, Mingjing Peng, Fanfan Zhang

School of Computer and Artificial Intelligence, Zhengzhou University


## Overview

Based on the classical primal-dual splitting method, we propose a deep convolutional primal-dual network (DeepConvPDNet) for image deblurring. The network is constructed by unfolding the iterative optimization procedure into an encoder-decoder architecture, where the linear operator and its adjoint in the original algorithm are substituted with convolutional and transposed convolutional layers, respectively. To enhance feature propagation, skip connections are incorporated across layers, resulting in a fully connected convolutional primal-dual network. Furthermore, two structural variants—cascaded and parallel—are developed based on different convolutional kernel designs. By treating the blur kernel as a learnable convolutional layer and its conjugate as the corresponding deconvolutional layer, the proposed model extends to a blind deblurring framework, eliminating the need for explicit kernel knowledge. Experimental results on the MNIST and BSD datasets demonstrate the effectiveness of the proposed approach, achieving competitive performance compared to state-of-the-art methods.
