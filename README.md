# MFNWSL: Multi-Feature Fusion Network with Weakly Supervised Localization
The official implementation of the paper "Multi-Feature Fusion Network with Weakly Supervised Localization for Gastric Intestinal Metaplasia Grading".  
Authors: Zhaohui Wang, Xiangwei Zheng, Rui Li, Mingzhe Zhang
## Introduction
Framework diagram
![Framework diagram](https://github.com/zhaohui-sdnu/MFNWSL/blob/main/docs/MFNWSL.png)
## Dataset acquisition
Data sharing is not applicable to this article due to medical data privacy. If you need the dataset, please contact the corresponding author.  
Contact email: xwzhengcn@163.com
## Pre-training lesion activation map generation network
- Create environment & install required packages
- Prepare dataset folder (a parent directory containing three sub-folders '0', '1' and '2' like below):
```python
.../path/to/data1
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
    dataset_path = 'path_to_dataset1'
```
- Use 5-fold cross validation to pre-train the lesion activation map generation network and save the best weights
```python
python pretrain.py
```
## Training and testing MFNWSL
- Create environment & install required packages
- Prepare dataset folder (a parent directory containing three sub-folders '0', '1' and '2' like below):
```python
.../path/to/data2
            | 0 (containing GIM grade 0 images)
            | 1 (containing GIM grade 1 images)
            | 2 (containing GIM grade 2 images)
```
- Configure training parameters in train_test.py, the default settings are as below:
```python
    # Train Parameters:
    epochs = 40
    batch_size = 64
    learning_rate = 0.0001
    num_classes = 3
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset_path = 'path_to_dataset2'
```
- Import the lesion activation map generation network in models/mfnwsl.py and load the pre-trained weights
```python
from models.pretrained_net import resnet50 as mase_cam
model = mase_cam()
model.load_state_dict(torch.load('weights/pretrained_net_resnet50_K4'))
```
- Use 5-fold cross validation to train and evaluate MFNWSL
```python
python train_test.py
```
## Evaluate
- Configure training parameters in evaluate.py, the default settings are as below:
```python
    # Train Parameters:
    epochs = 40
    batch_size = 64
    learning_rate = 0.0001
    num_classes = 3
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset_path = 'path_to_dataset2'
```
- Use the “timm” library in evaluate.py to build models for comparisons
```python
model=timm.create_model('xception',pretrained=True,num_classes=3)
#[resnet50,vit_base_patch16_224,swin_small_patch4_window7_224,vgg16,inception_v4,densenet121,
# efficientnet_b0,xception,mobilenetv2_050,mobilevitv2_050,convnext_base]
```
- Use 5-fold cross validation to train and evaluate models for comparisons
```python
python evaluate.py
```
## Results
Quantitative results  
![results](https://github.com/zhaohui-sdnu/MFNWSL/blob/main/docs/results.png)
![Confusion Matrix](https://github.com/zhaohui-sdnu/MFNWSL/blob/main/docs/Confusion_Matrix.png)
![ROC](https://github.com/zhaohui-sdnu/MFNWSL/blob/main/docs/ROC.png)  
Visualization results  
![Visualization results](https://github.com/zhaohui-sdnu/MFNWSL/blob/main/docs/Visualize.png)
