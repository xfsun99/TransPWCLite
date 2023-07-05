# TransPWCLite
## Code for the article "TransPWCLite: A Lightweight Transformer-Encoded Optical Flow Pyramidal Network for Motion Estimation in Ultrasound Imaging"
## TransPWCLite &mdash; Official PyTorch Implementation

![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg?style=plastic) ![PyTorch 1.8.0](https://img.shields.io/badge/pytorch-1.8.0-green.svg?style=plastic)

This repository contains the official PyTorch implementation of the paper "TransPWCLite: A Lightweight Transformer-Encoded Optical Flow Pyramidal Network for Motion Estimation in Ultrasound Imaging".


## Using the Code

### Requirements

This code has been developed under Python3, PyTorch 1.8.0 and CUDA 11.1 on Ubuntu 18.04. 

You can build the environment by following:

```shell
# Install python packages
pip3 install -r requirements.txt
```
### Training

Here we provide the complete training pipeline for pre-training on Sintel datasets:

#### Sintel dataset

1. Pre-train on the Sintel raw movie. 

   ```shell
   python3 train.py -c sintel_pre.json
   ```

2. Fine-tune on the ultrasound elastography imaging training set.

   ```shell
   python3 train.py -c ultra_ft.json
   ```
   
> The default configuration uses the whole training set for pre-training.
  Please refer to [configs](./configs) for more details.   

### Evaluation

Coming Soon!

## Datasets in the paper

Due to copyright issues, please download the dataset from the official websites.

- [MPI Sintel Dataset](http://sintel.is.tue.mpg.de/downloads).
- [Ultasound Elastography Simulation Database](https://www.dropbox.com/sh/3qft4y765tkhu91/AADlMzFP1y1-kLUd0xNvR6hAa?dl=0). 

We thank for portions of the source code from some great works such as [ARFlow](https://github.com/lliuz/ARFlow), [IRR](https://github.com/visinf/irr) and [ViT](https://github.com/lucidrains/vit-pytorch).
