# Negative Margin Matters: Understanding Margin in Few-shot Classification

By [Bin Liu](https://scholar.google.com/citations?user=-RYlJvYAAAAJ&hl=zh-CN), [Yue Cao](http://yue-cao.me), Yutong Lin, Qi Li, [Zheng Zhang](https://www.microsoft.com/en-us/research/people/zhez/), [Mingsheng Long](http://ise.thss.tsinghua.edu.cn/~mlong/), [Han Hu](https://ancientmooner.github.io/).

This repo is an official implementation of ["Negative Margin Matters: Understanding Margin in Few-shot Classification"](https://arxiv.org/abs/2003.12060) on PyTorch.

*Update on 2020/07/01*

Our paper was accepted by ECCV 2020 as spotlight!

## Introduction

This paper is initially described in [arxiv](https://arxiv.org/abs/2003.12060), which introduces a negative margin loss to metric learning based few-shot learning methods. The negative margin loss significantly outperforms regular softmax loss, and achieves state-of-the-art accuracy on three standard few-shot classification benchmarks with few bells and whistles. These results are contrary to the common practice in the metric learning field, that the margin is zero or positive. To understand why the negative margin loss performs well for the few-shot classification, the authors analyze the discriminability of learned features w.r.t different margins for training and novel classes, both empirically and theoretically. They find that although negative margin reduces the feature discriminability for training classes, it may also avoid falsely mapping samples of the same novel class to multiple peaks or clusters, and thus benefit the discrimination of novel classes. 

## Citation

```
@article{liu2020negative,
  title={Negative Margin Matters: Understanding Margin in Few-shot Classification},
  author={Liu, Bin and Cao, Yue and Lin, Yutong and Li, Qi and Zhang, Zheng and Long, Mingsheng and Hu, Han},
  journal={arXiv preprint arXiv:2003.12060},
  year={2020}
}
```

## Main Results

The few-shot classification accuracy on the novel classes with ResNet-18 as the backbone is listed bellowing:.

| Method     | Mini-ImageNet<br/>1 - shot | Mini-ImageNet<br/>5 - shot | CUB<br/>1 - shot | CUB<br/>5 - shot | Mini-ImageNet -> CUB<br/>5-shot |
| :---------------------: | :-------------------------------------------------: | :-------------------------------------------------: | :--------------------------------------: | :--------------------------------------: | :-------------------------------------------------------: |
| Softmax    | 51.75+-0.80                                       | 74.27+-0.63                                       | 65.51+-0.87                            | 82.85+-0.55                            | 65.57+-0.70                                             |
| Cosine     | 51.87+-0.77                                       | 75.68+-0.63                                       | 67.02+-0.90                            | 83.58+-0.54                            | 62.04+-0.76                                             |
| Neg-Sofmax | 59.02+-0.81                                       | 78.80+-0.61                                       | 71.48+-0.83                            | 87.30+-0.48                            | 69.30+-0.73                                             |
| Neg-Cosine | 62.33+-0.82                                       | 80.94+-0.59                                       | 72.66+-0.85                            | 89.40+-0.43                            | 67.03+-0.76                                             |

 You can download the pre-trained model checkpoints of `resnet-18` from [OneDrive](https://1drv.ms/u/s!AsaPPmtCAq08pRM54_CuGPFbfgUz?e=ydjBfW).

## Getting started

### Environment

 - `Anaconda` with `python >= 3.6`
 - `pytorch=1.2.0, torchvison, cuda=9.2`
 - others: `pip install yacs`

### Datasets

#### CUB

* Change directory to `./data/CUB`
* run `bash ./download_CUB.sh`

#### mini-ImageNet

* Change directory to `./data/miniImagenet`
* run `bash ./download_miniImagenet.sh` 

(WARNING: This would download the 155G ImageNet dataset. You can comment out correponded line 5-6 in `download_miniImagenet.sh` if you already have one.) 

#### mini-ImageNet->CUB

* Finish preparation for CUB and mini-ImageNet and you are done!

## Train and eval

Run the following commands to train and evaluate:

```
python main.py --config [CONFIGFILENAME] \
    --supp [SUPPLEMENTSTRING] \
    method.backbone [BACKBONE] \
    method.image_size [IMAGESIZE] \
    method.metric_params.margin [MARGIN] \
    [OPTIONARG]
```

 For additional options, please refer to `./lib/config.py`.

We have also provided some scripts to reproduce the results in the paper. Pleas check `./script` for details.

## References
Our testbed builds upon several existing publicly available code. Specifically, we have modified and integrated the following code into this project:

* Framework, Backbone, Method: Closer Look at Few Shot
https://github.com/wyharveychen/CloserLookFewShot
* Backbone(resnet12): MetaOpt
https://github.com/kjunelee/MetaOptNet
* Backbone(WideResNet28_10): S2M2_R
https://github.com/nupurkmr9/S2M2_fewshot
