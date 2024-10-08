# MFNWSL: Multi-Feature Fusion Network with Weakly Supervised Localization
The official implementation of the paper "Multi-Feature Fusion Network with Weakly Supervised Localization for Gastric Intestinal Metaplasia Grading".  
Authors: Zhaohui Wang, Xiangwei Zheng, Rui Li, Mingzhe Zhang
## Introduction
Framework diagram
![Framework diagram](https://github.com/zhaohui-sdnu/MFNWSL/blob/main/docs/MFNWSL.png)
## Dataset acquisition
Data sharing is not applicable to this article due to medical data privacy. If you need the dataset, please contact the corresponding author.  
Contact email: xwzhengcn@163.com
## Train
- Create environment & install required packages
- Prepare dataset folder (a parent directory containing three sub-folders '0', '1' and '2' like below):
```python
.../path/to/data
            | 0 (containing GIM grade 0 images)
            | 1 (containing GIM grade 1 images)
            | 2 (containing GIM grade 2 images)
```
- Configure training parameters in pretrain.py, the default settings are as below:
```python
    # Train Parameters:
    epochs = 40
    batch_size = 64
    learning_rate = 0.0001
    num_classes = 3
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset_path = 'path_to_dataset'
```
- Pre-train the lesion activation map generation network and save the weights
```python
python pretrain.py
```
- Training and testing MFNWSL
```python
python train_test.py
```
