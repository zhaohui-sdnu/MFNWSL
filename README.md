# MFNWSL: Multi-Feature Fusion Network with Weakly Supervised Localization
The official implementation of the paper "Multi-Feature Fusion Network with Weakly Supervised Localization for Gastric Intestinal Metaplasia Grading".  
Authors: Zhaohui Wang, Xiangwei Zheng, Rui Li, Mingzhe Zhang
## Introduction
![Framework diagram](https://github.com/zhaohui-sdnu/MFNWSL/docs/MFNWSL.png)
## Dataset acquisition
Data sharing is not applicable to this article due to medical data privacy. If you need the dataset, please contact the corresponding author.  
Contact email: xwzhengcn@163.com
## Train
- Create environment & install required packages
- Preparing the dataset
- Pre-train the lesion activation map generation network and save the weights
```python
python pretrain.py
```
- Training and testing MFNWSL
```python
python train_test.py
```
