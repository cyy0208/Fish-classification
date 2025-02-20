import torch
import torch.nn as nn
import torch.nn.functional as F

class ShuffleNetV2Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ShuffleNetV2Block, self).__init__()
        
        self.stride = stride
        
        # 第一部分：1x1卷积，改变通道数
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        
        # 第二部分：3x3卷积，深度可分离卷积
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, stride=stride, padding=1, groups=out_channels // 2, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels // 2)
        
        # 第三部分：1x1卷积，恢复通道数
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def channel_shuffle(self, x, groups):
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
        
        # 重排通道
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(batch_size, num_channels, height, width)
        return x

    def forward(self, x):
        # 通道重排
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)

        out = self.conv3(out)
        out = self.bn3(out)

        # 如果步幅为2，则跳过连接需要调整
        if self.stride == 2:
            x = F.avg_pool2d(x, 2)
            x = F.pad(x, (0, 0, 0, 0, 0, out.size(1) - x.size(1)))  # 使得尺寸匹配

        # 通道重排
        out = self.channel_shuffle(out, 2)

        return out + x


class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes=1000):
        super(ShuffleNetV2, self).__init__()

        self.initial_conv = nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(24)
        
        # 构建 ShuffleNetV2 的层
        self.stage1 = self._make_stage(24, 116, 4, 2)  # 输出为 116
        self.stage2 = self._make_stage(116, 232, 8, 2)
        self.stage3 = self._make_stage(232, 464, 4, 2)
        self.stage4 = self._make_stage(464, 1024, 2, 1)
        
        self.fc = nn.Linear(1024, num_classes)

    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ShuffleNetV2Block(in_channels, out_channels, stride))
            in_channels = out_channels
            stride = 1  # 之后的 block stride 都为1
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = F.relu(x, inplace=True)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

# Example usage:
model = ShuffleNetV2(num_classes=1000)
input_tensor = torch.randn(1, 3, 224, 224)
output = model(input_tensor)
print(output.shape)  # Should be [1, 1000] if num_classes=1000
