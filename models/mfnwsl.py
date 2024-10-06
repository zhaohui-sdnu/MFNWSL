import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models,transforms, datasets
from skimage.feature import local_binary_pattern
from torch.hub import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
import numpy as np
from skimage import color
from pytorch_grad_cam import GradCAMPlusPlus
from models.pretrained_net import resnet50 as mase_cam
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
model = mase_cam()
#model.fc=nn.Linear(model.fc.in_features,3)
model.load_state_dict(torch.load('weights/pretrained_net_resnet50_K4'))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
 #图像变换模块
class ImageTransform(nn.Module):
    def __init__(self):
        super(ImageTransform, self).__init__()
        self.hsv_transform = HSVTransform()
        self.lbp_transform = LBPTransform()

    def forward(self, x):
        x_hsv = self.hsv_transform(x)
        x_lbp = self.lbp_transform(x)
        return x, x_hsv, x_lbp

#HSV
class HSVTransform(nn.Module):
    def __init__(self):
        super(HSVTransform, self).__init__()
    def forward(self, x):
        # Convert torch tensor to numpy array
        #x_np = x.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        hsv_tensor = torch.zeros_like(x)
        for i in range(x.shape[0]):
            rgb_img=x[i].permute(1,2,0).detach().cpu().numpy()
            hsv_image=color.rgb2hsv(rgb_img)
            hsv_tensor[i]=torch.from_numpy(hsv_image).permute(2,0,1)

        hsv_tensor = hsv_tensor.cuda(0)
        #print(hsv_tensor.shape)
        return hsv_tensor

# LBP纹理变换
class LBPTransform(nn.Module):
    def __init__(self, radius=1, n_points=8):
        super(LBPTransform, self).__init__()
        self.radius = radius
        self.n_points = n_points

    def forward(self, x):
        # Convert torch tensor to numpy array
        x_np = x.permute(0, 2, 3, 1).detach().contiguous().cpu().numpy()

        lbp_images = []
        for i in range(x_np.shape[0]):
            # Convert each channel to grayscale and apply LBP
            lbp_channel = []
            for c in range(x_np.shape[3]):
                lbp_channel.append(local_binary_pattern(x_np[i, :, :, c], P=self.n_points, R=self.radius))
            lbp_images.append(torch.tensor(lbp_channel, dtype=torch.float32).unsqueeze(0))

        lbp_tensor = torch.cat(lbp_images, dim=0)
        lbp_tensor = lbp_tensor.cuda(0)
        #print("lbp_tensor:",lbp_tensor.size())
        #lbp_tensor = lbp_tensor.permute(0, 2, 3, 1).contiguous()
        #print("lbp_tensor:", lbp_tensor.size())
        return lbp_tensor


#多头自注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_size = input_size // num_heads

        self.linear_q = nn.Linear(input_size, input_size, bias=False)
        self.linear_k = nn.Linear(input_size, input_size, bias=False)
        self.linear_v = nn.Linear(input_size, input_size, bias=False)

    def forward(self, x):
        #print("input shape:",x.shape)
        batch_size,feature_dim,seq_len,_ = x.size()
        #print("l_q(x): ",self.linear_q(x).shape)
        #q = self.linear_q(x).view(batch_size, self.num_heads, seq_len,  self.head_size)
        q = self.linear_q(x).view(batch_size * seq_len, feature_dim, self.num_heads, self.head_size)
        k = self.linear_k(x).view(batch_size * seq_len, feature_dim, self.num_heads, self.head_size)
        v = self.linear_v(x).view(batch_size * seq_len, feature_dim, self.num_heads, self.head_size)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_size)
        attention_weights = F.softmax(attention_scores, dim=-1)

        output = torch.matmul(attention_weights, v)
        output = output.view(batch_size, feature_dim,56,56)
        #print("MH:",output.shape)

        return output


class SE(nn.Module):

    def __init__(self, in_chnls, ratio):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_chnls, in_chnls//ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls//ratio, in_chnls, 1, 1, 0)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = F.relu(out)
        out = self.excitation(out)
        return F.sigmoid(out)

# SE Block
class SEBlock(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_channels, input_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(input_channels // reduction_ratio, input_channels, bias=False),
            nn.Sigmoid()
        )
        #降维操作
        self.reduce_channels=nn.Conv2d(input_channels,input_channels//3,kernel_size=1,bias=False)
    def forward(self, x):
        batch_size, feature_dim ,_,_= x.size()
        #print("x.size:",x.shape)
        # Global average pooling
        #y = self.avg_pool(x).view(batch_size, 1,feature_dim)
        #x=x.view(x.size(0),x.size(2),x.size(1))
        #print("se_avgpool:",self.avg_pool(x).shape)
        y = self.avg_pool(x).view(batch_size, feature_dim)
        #print("y:", y.shape)
        # Squeeze and Excitation
        y = self.fc(y)
        #print("fc_y:",y.shape)
        y = y.view(batch_size, feature_dim,1, 1)

        #return x * y.expand_as(x)
        #y=y.squeeze(3)
        w = x * y.expand_as(x)
        #w= self.reduce_channels(w)
        #print("se:",w.shape)
        return w

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        is_se=False
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.is_se = is_se
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        if self.is_se:
            self.se = SE(planes,16)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.is_se:
            coefficient = self.se(out)
            out *=coefficient

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        is_se=False
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.is_se = is_se
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        if self.is_se:
            self.se = SE(planes * self.expansion,16)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.is_se:
            coefficient = self.se(out)
            out *= coefficient

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        is_se=False
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.cam_model = model
        self.cam_model.load_state_dict(
            torch.load('weights/pretrained_net_resnet50_K4'))
        self.target_layers = [self.cam_model.layer4[-1]]
        # self.cam_generator = CAMGenerator(model=self.cam_model,target_layers=self.target_layers)
        self.cam = GradCAMPlusPlus(model=self.cam_model, target_layers=self.target_layers, use_cuda=True)

        self._norm_layer = norm_layer
        self.inplanes = 64
        self.inplanes_hsv = 64
        self.inplanes_lbp = 64

        self.dilation = 1
        self.dilation_hsv = 1
        self.dilation_lbp = 1

        self.is_se=is_se
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.groups_hsv = groups
        self.groups_lbp = groups

        self.base_width = width_per_group
        self.base_width_hsv = width_per_group
        self.base_width_lbp = width_per_group
        #rgb_branch
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        input_channels = 56
        #hsv_branch
        self.conv1_hsv = nn.Conv2d(3, self.inplanes_hsv, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1_hsv = norm_layer(self.inplanes_hsv)
        self.relu_hsv = nn.ReLU(inplace=True)
        self.maxpool_hsv = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_hsv = self._make_layer_hsv(block, 64, layers[0])
        self.multi_head_attention_hsv = MultiHeadAttention(input_channels, 4)
        self.gap_hsv = nn.AdaptiveAvgPool2d((1,1))

        #lbp_branch
        self.conv1_lbp = nn.Conv2d(3, self.inplanes_lbp, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        self.bn1_lbp = norm_layer(self.inplanes_lbp)
        self.relu_lbp = nn.ReLU(inplace=True)
        self.maxpool_lbp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_lbp = self._make_layer_lbp(block, 64, layers[0])
        self.multi_head_attention_lbp = MultiHeadAttention(input_channels, 4)
        self.gap_lbp = nn.AdaptiveAvgPool2d((1, 1))

        # 初始化 MultiHeadAttention 和 SEBlock
        self.image_transform = ImageTransform()
         # 根据实际需要调整输入通道数 resnet50 101:2560 resnet34:640
        self.se_block = SEBlock(2560)
        self.dropout = nn.Dropout(p=0.5)
        #self.fc1 = nn.Linear(2560, 1280)
        self.fc2 = nn.Linear(2560, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer,self.is_se))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer,is_se=self.is_se))

        return nn.Sequential(*layers)
    def _make_layer_hsv(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation_hsv
        if dilate:
            self.dilation_hsv *= stride
            stride = 1
        if stride != 1 or self.inplanes_hsv != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes_hsv, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes_hsv, planes, stride, downsample, self.groups,
                            self.base_width_hsv, previous_dilation, norm_layer,self.is_se))
        self.inplanes_hsv = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes_hsv, planes, groups=self.groups,
                                base_width=self.base_width_hsv, dilation=self.dilation_hsv,
                                norm_layer=norm_layer,is_se=self.is_se))

        return nn.Sequential(*layers)
    def _make_layer_lbp(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation_lbp
        if dilate:
            self.dilation_lbp *= stride
            stride = 1
        if stride != 1 or self.inplanes_lbp != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes_lbp, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes_lbp, planes, stride, downsample, self.groups,
                            self.base_width_lbp, previous_dilation, norm_layer,self.is_se))
        self.inplanes_lbp = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes_lbp, planes, groups=self.groups,
                                base_width=self.base_width_lbp, dilation=self.dilation_lbp,
                                norm_layer=norm_layer,is_se=self.is_se))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]

        self.cam_model.load_state_dict(
            torch.load('weights/pretrained_net_resnet50_K4'))
        self.cam_model.eval()
        with torch.no_grad():
            output = self.cam_model(x)
        _, predicted_classes = torch.max(output.data, 1)
        # print(predicted_classes)
        cam_features = []
        y = x
        y.requires_grad = True
        for i in range(x.shape[0]):
            pre = predicted_classes[i].item()
            targets = [ClassifierOutputTarget(pre)]
            p = F.softmax(output[i],dim=0)
            cam_feature = self.cam(input_tensor=y[i:i + 1], targets=targets)
            #print(cam_feature)
            cam_feature_tensor = torch.from_numpy(cam_feature)
            #cam_weight = F.sigmoid(cam_feature_tensor)
            cam_weight = cam_feature_tensor
            cam_weight = cam_weight.cuda(0)
            if pre == 0:
                cam_weight=cam_weight * (1-p[0])
            else:
                cam_weight=cam_weight * p[pre]
            # print(cam_feature.shape)
            cam_features.append(cam_weight)
        #cam_features = [torch.from_numpy(cam_feature) for cam_feature in cam_features]
        x_cam = torch.stack(cam_features, dim=0)
        x_cam = x_cam.cuda(0)
        #print(x_cam)

        # 图像变换
        x, x_hsv, x_lbp = self.image_transform(x)
        #x
        #print("x_inplan:",self.inplanes)
        x = x * x_cam
        #x_hsv = x_hsv * x_cam
        #x_lbp = x_lbp * x_cam
        x = self.conv1(x)
        #print("x_conv1_inplan:", self.inplanes)
        x = self.bn1(x)
        #print("x_bn1_inplan:", self.inplanes)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        #print("x:",x.shape)

        #x_hsv
        #print("hsv_inplan:", self.inplanes)
        x_hsv = self.conv1_hsv(x_hsv)
        #print("hsv_conv1_inplan:", self.inplanes)
        x_hsv = self.bn1_hsv(x_hsv)
        #print("hsv_bn1_inplan:", self.inplanes)
        x_hsv = self.relu_hsv(x_hsv)
        x_hsv = self.maxpool_hsv(x_hsv)
        x_hsv = self.layer1_hsv(x_hsv)
        #print("x_hsv:", x_hsv.shape)
        x_hsv = self.multi_head_attention_hsv(x_hsv)
        x_hsv = self.gap_hsv(x_hsv)
        #print("x_hsv:",x_hsv.shape)
        #x_lbp
        x_lbp = self.conv1_lbp(x_lbp)
        x_lbp = self.bn1_lbp(x_lbp)
        x_lbp = self.relu_lbp(x_lbp)
        x_lbp = self.maxpool_lbp(x_lbp)
        x_lbp = self.layer1_lbp(x_lbp)
        x_lbp = self.multi_head_attention_lbp(x_lbp)
        x_lbp = self.gap_lbp(x_lbp)
        #print("x_lbp:", x_lbp.shape)

        #fused
        fused_feature = torch.cat([x, x_hsv, x_lbp], dim=1)
        #fused_feature = x+x_hsv+x_lbp

        #print("x:",x.shape)
        #fused_feature = x
        #print("fused_feature:",fused_feature.size())
        # 使用多头自注意力机制进行融合
        #fused_feature = self.multi_head_attention(fused_feature)
        # print("MH_A:",fused_feature.shape)
        # SEBlock 进行特征选择
        fused_feature = self.se_block(fused_feature)

        #x = self.layer2(fused_feature)
        #x = self.layer3(x)
        #x = self.layer4(x)
        #x = self.avgpool(fused_feature)
        #print("avgpool:",x.shape)
        x = torch.flatten(fused_feature, 1)
        #print("flatten:", x.shape)
        x = self.dropout(x)
        #x = self.fc1(x)
        #x = self.dropout(x)
        x = self.fc2(x)
        #print("fc:", x.shape)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,is_se=False,
                   **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
def test():
    print('--- run resnet test ---')
    x = torch.randn(2, 3, 224, 224)
    for net in [resnet50()]:
        # , ResNet34(10), ResNet50(10), ResNet101(10), ResNet152(10)]:
        print(net)
        y = net(x)
        print(y.size())
        print(y)
#test()
