# :page_facing_up: Domain and Content Adaptive Convolution for Domain Generalization in Medical Image Segmentation (DCAC)
#### [Paper](https://arxiv.org/abs/2003.09005), [Project Page](https://shishuaihu.github.io/DCAC/)

This repo contains the official implementation of our paper: Domain and Content Adaptive Convolution for Domain Generalization in Medical Image Segmentation, which adapts dynamic convolution for domain generalization.

<p align="center"><img src="https://shishuaihu.github.io/DCAC/page_files/method.png" width="90%"></p>

### Highlights

**(1) Multi-scale features based domain relationship modeling.**

We use the domain-discriminative information embedded in the encoder feature maps to generate the domain code of each input image, which establishes the relationship between multiple source domains and the unseen target domain.

**(2) Domain and Content Adaptive Convolution.**

We design the dynamic convolution-based domain adaptive convolution (DAC) module and content adaptive convolution (CAC) module to enable our DCAC model to adapt not only to the unseen target domain but also to each test image.

**(3) Competitive results on three benchmarks.**

We present extensive experiments, which demonstrate the effectiveness of our DCAC model against the state-of-the-art in three medical image segmentation benchmarks with different imaging modalities.

### Requirements

This repo was tested with Ubuntu 20.04.3 LTS, Python 3.8, PyTorch 1.8.0, and CUDA 10.1. But it should be runnable with Ubuntu 16.04 and Ubuntu 18.04.

We suggest using virtual env to configure the experimental environment. Compiling PyTorch on your own workstation is suggested but not needed.

1. Clone this repo:

```bash
git clone https://github.com/ShishuaiHu/DCAC.git
```

2. Create experimental environment using virtual env:

```bash
cd DCAC/nnUNet
virtualenv .env --python=3.8 # create
source .env/bin/activate # activate
bash ./install.sh # install torch and nnUNet (equipped with DCAC)
```

3. Configure the paths in `.envrc` to the proper path:

```bash
echo -e '
export nnUNet_raw_data_base="nnUNet raw data path you want to store in"
export nnUNet_preprocessed="nnUNet preprocessed data path you want to store in, SSD is prefered"
export RESULTS_FOLDER="nnUNet trained models path you want to store in"' > .envrc

source .envrc # make the variables take effect
```

### Dataset

The dataset details and the download link can be found in the [Project Page](https://shishuaihu.github.io/DCAC/).

### Data Preprocessing

```bash
```




### Training

```bash
```




### Inference

```bash
```



### Pre-trained models

Pre-trained models can be downloaded [here](https://github.com/yassouali/CCT/releases).

```bash
```



### Citation ✏️ 📄

If you find this repo useful for your research, please consider citing the paper as follows:

```
@article{hu2021dcac,
        title={Domain and Content Adaptive Convolution for Domain Generalization in Medical Image Segmentation},
        author={Shishuai Hu and Zehui Liao and Jianpeng Zhang and Yong Xia},
        journal={arXiv},
        year={2021},
}
```

For any questions, please feel free to open an issue [here](https://github.com/ShishuaiHu/DCAC/issues/new).

### Acknowledgements

- The whole framework is based on [nnUNet](https://github.com/MIC-DKFZ/nnUNet).
- The code of the dynamic convolution is adopted from [DoDNet](https://github.com/jianpengz/DoDNet)

