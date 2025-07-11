import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        self.sigmoid_channel = nn.Sigmoid()

        self.conv_after_concat = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        avg = self.avg_pool(x)
        max = self.max_pool(x)
        out = torch.cat([avg, max], dim=1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid_channel(out)
        out = x * out

        # Spatial attention
        out = torch.cat([out, x], dim=1)
        out = self.conv_after_concat(out)
        out = self.sigmoid_spatial(out)
        out = x * out
        return out


class ResNet34_CBAM(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet34_CBAM, self).__init__()
        self.resnet34 = resnet34(pretrained=True)
        self.conv1 = self.resnet34.conv1
        self.bn1 = self.resnet34.bn1
        self.relu = self.resnet34.relu
        self.maxpool = self.resnet34.maxpool
        self.layer1 = nn.Sequential(
            CBAM(64),
            self.resnet34.layer1[0],
            self.resnet34.layer1[1],
            self.resnet34.layer1[2],
            self.resnet34.layer1[3]
        )
        self.layer2 = nn.Sequential(
            CBAM(128),
            self.resnet34.layer2[0],
            self.resnet34.layer2[1],
            self.resnet34.layer2[2],
            self.resnet34.layer2[3]
        )
        self.layer3 = nn.Sequential(
            CBAM(256),
            self.resnet34
        )