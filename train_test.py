import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torchvision import models, transforms
from skimage.feature import local_binary_pattern
import timm
import torch
from torch.utils.data import DataLoader
from torchvision import models,transforms, datasets
from transformers import ViTModel
from timm.models.vision_transformer import VisionTransformer
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function
from PIL import Image
import cv2
from models.pretrained_net import resnet50 as mase
from models.mfnwsl import resnet50 as cam
#from models.resnet50_mase_classification import resnet50 as mase_cam
from torchvision.models.resnet import resnet50
from torchvision.models.resnet import resnet34
from torchvision.models.resnet import resnet101
import numpy as np
from skimage import color
from skimage.feature import local_binary_pattern
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

# 定义超参数（需要根据实际情况调整）
epochs = 40
batch_size = 64
learning_rate = 0.0001
num_classes = 3
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 创建数据集和数据加载器
def filter_dark_and_reflection(image, brightness_threshold=60, saturation_threshold=50):
    # 读取图像
    #image = cv2.imread(image_path)
    #print("11")
    image = np.array(image)
    # 将图像从BGR颜色空间转换为HSV颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 分离HSV通道
    h, s, v = cv2.split(hsv_image)

    # 创建两个掩码，分别用于暗区和反射区
    dark_mask = np.logical_not((v < brightness_threshold) & (s > saturation_threshold)).astype(np.uint8)
    reflection_mask = np.logical_not((v >= brightness_threshold) & (s <= saturation_threshold)).astype(np.uint8)

    # 将暗区和反射区的像素置零
    image = cv2.bitwise_and(image, image, mask=cv2.UMat(dark_mask))
    image = cv2.bitwise_and(image, image, mask=cv2.UMat(reflection_mask))
    #print("wancheng:",type(image))
    image = cv2.UMat.get(image)
    return image


enhance_transform = transforms.Compose([
    #transforms.ToPILImage(),
    transforms.Lambda(lambda img: filter_dark_and_reflection(img)),
    #transforms.ToTensor(),
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    #transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.2),
    transforms.RandomRotation(degrees=30),
    transforms.RandomPerspective(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4664, 0.3260, 0.2799], std=[0.2243, 0.1694, 0.1500])
])
data_transform = transforms.Compose([

    transforms.Lambda(lambda img: filter_dark_and_reflection(img)),
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4664, 0.3260, 0.2799], std=[0.2243, 0.1694, 0.1500])
])

dataset = datasets.ImageFolder('data', transform=enhance_transform)


#model=mase_cam()


#model.load_state_dict(torch.load('save_vit/newmase_pretrain0720.pth'))
#miss,unexpected=model.load_state_dict(torch.load("save_vit/resnet50-0676ba61.pth"),strict=False)

#model=torchvision.models.resnet50(pretrained=True)
#model.fc=nn.Linear(model.fc.in_features,3)

#model=timm.create_model('resnet50',pretrained=False,num_classes=3)
#model = ModifiedResNet50(num_classes=3, num_heads=4)



# 定义训练和验证函数
def train(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        # 将数据移动到GPU上
        images, labels = images.to(device), labels.to(device)
        #torch.cuda.init()
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计预测信息
        _, predicted = torch.max(outputs.data, 1)
        #print("train_pre:",predicted)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        running_loss += loss.item() * images.size(0)

    # 计算平均loss和准确率
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion):
    model.eval()
    #model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    confusion_matrix = torch.zeros(3, 3)

    #with torch.no_grad():
    for images, labels in val_loader:
            # 将数据移动到GPU上
        images, labels = images.to(device), labels.to(device)

            # 前向传播
        #torch.cuda.empty_cache()
        #print("--------------------")
        outputs = model(images)
            #print("val_out:", outputs)
        loss = criterion(outputs, labels)
        #torch.cuda.empty_cache()

            # 统计预测信息
        _, predicted = torch.max(outputs.data, 1)
            #print("val_pre:", predicted)
        for i in range(labels.shape[0]):
            confusion_matrix[labels[i], predicted[i]] += 1
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        running_loss += loss.item() * images.size(0)

    # 计算平均loss和准确率
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = correct / total
    epoch_p = torch.diag(confusion_matrix) / confusion_matrix.sum(dim=0)
    epoch_r = torch.diag(confusion_matrix) / confusion_matrix.sum(dim=1)
    epoch_f1 = 2 * epoch_p * epoch_r / (epoch_p + epoch_r)
    p = epoch_p.mean().item()
    r = epoch_r.mean().item()
    f1 = epoch_f1.mean().item()

    return epoch_loss, epoch_acc, p,r,f1


# 定义KFold
kfold = KFold(n_splits=5, shuffle=True, random_state=24)

# 初始化指标列表
acc = []
p = []
r = []
f1 = []

# K折交叉验证循环
for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
    print(f"Fold {fold + 1}")
    save_path = f"weights/mfnwsl_resnet50_K{fold+1}"

    # 划分数据集
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    #model
    model = cam()

    resnet50_model = resnet50(pretrained=True)
    resnet50_state_dict = resnet50_model.state_dict()
    del_key = []
    for key, _ in resnet50_state_dict.items():
        if "fc" in key:
            del_key.append(key)
    for key in del_key:
        del resnet50_state_dict[key]
    model_state_dict = model.state_dict()
    for key in resnet50_state_dict:
        if key in model_state_dict:
            model_state_dict[key] = resnet50_state_dict[key]

    for k in model_state_dict:
        if '_hsv_c' in k:
            model_state_dict[k] = resnet50_state_dict[k.replace('_hsv_c', '')]
        if '_lbp_c' in k:
            model_state_dict[k] = resnet50_state_dict[k.replace('_lbp_c', '')]
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    # 定义损失函数和优化器
    class_weights = torch.tensor([1.0, 2.0, 1.0]).cuda(0)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.1, verbose=True)
    # 训练和测试
    best_f1 = 0
    best_acc = 0
    best_p = 0
    best_r = 0
    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_p, val_r, val_f1 = validate(model, test_loader, criterion)
        scheduler.step(val_loss)

        print('Epoch [{}/{}], TLoss:{:.4f}, TAcc:{:.4f},vLoss:{:.4f}, VAcc:{:.4f}, Vp:{:.4f}, Vr:{:.4f}, Vf1:{:.4f}'
              .format(epoch + 1, epochs, train_loss, train_acc, val_loss, val_acc, val_p, val_r, val_f1))

        # 保存最好的模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), save_path)
        if val_acc > best_acc:
            best_acc = val_acc
        if val_p > best_p:
            best_p = val_p
        if val_r > best_r:
            best_r = val_r
    acc.append(best_acc)
    p.append(best_p)
    r.append(best_r)
    f1.append(best_f1)


average_acc = sum(acc) / len(acc)
average_p = sum(p) / len(p)
average_r = sum(r) / len(r)
average_f1 = sum(f1) / len(f1)
print("acc:",acc)
print("p:",p)
print("r:",r)
print("f1:",f1)
print(f"Average acc Score: {average_acc}")
print(f"Average p Score: {average_p}")
print(f"Average r Score: {average_r}")
print(f"Average f1 Score: {average_f1}")




