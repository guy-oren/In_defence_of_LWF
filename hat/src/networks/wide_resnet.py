import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1, conv3x3


class WideResNet(nn.Module):
    def __init__(self, c_in, block, layers, taskcla, width_factor=1, zero_init_residual=False):
        super().__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(c_in, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16 * width_factor, layers[0])
        self.layer2 = self._make_layer(block, 32 * width_factor, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64 * width_factor, layers[2], stride=2)
        #self.layer4 = self._make_layer(block, inplanes * 8, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.taskcla = taskcla

        self.last = nn.ModuleList()
        for t, n in self.taskcla:
            self.last.append(nn.Linear(64 * width_factor * block.expansion, n))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        y = []
        for t, i in self.taskcla:
            y.append(self.last[t](x))
        return y

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def wide_resnet20(c_in, taskcla, width_factor=1, **kwargs):
    """Constructs a WideResNet-20 model with with Filter Response Normalization.
    """
    model = WideResNet(c_in, BasicBlock, [3, 3, 3], taskcla, width_factor, **kwargs)
    return model


def wide_resnet32(c_in, taskcla, width_factor=1, **kwargs):
    """Constructs a WideResNet-20 model with with Filter Response Normalization.
    """
    model = WideResNet(c_in, BasicBlock, [5, 5, 5], taskcla, width_factor, **kwargs)
    return model


def wide_resnet62(c_in, taskcla, width_factor=1, **kwargs):
    """Constructs a WideResNet-20 model with with Filter Response Normalization.
    """
    model = WideResNet(c_in, BasicBlock, [10, 10, 10], taskcla, width_factor, **kwargs)
    return model
