# From General to Specific: Informative Scene Graph Generation via Balance Adjustment

![LICENSE](https://img.shields.io/badge/license-MIT-green)
[![Python](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.2.0-%237732a8)

This repository contains code for the paper "From General to Specific: Informative Scene Graph Generation via Balance Adjustment". This code is based on [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch). The scene graph generation task aims to generate a set of triplet <subject, predicate, object> and construct a graph structure for an image.
<div align=center><img width="672" height="480" src=demo/framework_generalSGG.png/></div>

## Framework
<div align=center><img width="672" height="508" src=demo/framework_G2ST.png/></div>

## Abstract
The scene graph generation (SGG) task aims to detect visual relationship triplets, i.e., subject, predicate, object, in an image, providing a structural vision layout for scene understanding. However, current models are stuck in common predicates, e.g., ``on'' and ``at'', rather than informative ones, e.g., ``standing on'' and ``looking at'', resulting in the loss of precise information and overall performance. If a model only uses ``stone on road'' rather than ``blocking'' to describe an image, it is easy to misunderstand the scene. We argue that this phenomenon is caused by two key imbalances between informative predicates and common ones, i.e., semantic space level imbalance and training sample level imbalance. To tackle this problem, we propose BA-SGG, a simple yet effective SGG framework based on balance adjustment but not the conventional distribution fitting. It integrates two components: Semantic Adjustment (SA) and Balanced Predicate Learning (BPL), respectively for adjusting these imbalances. Benefited from the model-agnostic process, our method is easily applied to the state-of-the-art SGG models and significantly improves the SGG performance. Our method achieves 14.3%, 8.0%, and 6.1% higher Mean Recall (mR) than that of the Transformer model at three scene graph generation sub-tasks on Visual Genome, respectively. 

## Visualization
<div align=center><img width="994" height="774" src=demo/vis_res_supp1.png/></div>

## Setup 
Follow the [instructions](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) to install and use the code. We write some [scripts](https://github.com/ZhuGeKongKong/SSG-G2S/tree/main/scripts) for training and testing.
The training process is divided into two stages:
### Training the common SGG model
The training script should be set up as follows: \
    MODEL.PRETRAINED_MODEL_CKPT '' \
    MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER False \
    MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER False  
### Finetuning the informative SGG model
The training script should be set up as follows: \
    MODEL.PRETRAINED_MODEL_CKPT 'path to the common SGG model' \
    MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER True \
    MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER True  

## Help

Feel free to ping me if you encounter trouble getting it to work!

## Bibtex

```
@inproceedings{sgg:g2s,
  author    = {Yuyu Guo and
               Lianli Gao and
               Xuanhan Wang and
               Yuxuan Hu and
               Xing Xu and
               Xu Lu and
               Heng Tao Shen and
               Jingkuan Song},
  title     = {From General to Specific: Informative Scene Graph Generation via Balance Adjustment},
  booktitle = {ICCV},
  year      = {2021}
}
```
