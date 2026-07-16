import torch
import torch.nn as nn
import math
import torch.nn.functional as F



class SEBlock(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x

class Weight(nn.Module):

    def __init__(self):
        super(Weight, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256, 128)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 1)
        # self.relu2 = nn.ReLU()
        # self.bn2 = nn.BatchNorm1d()

    def forward(self, inputs):
        inputs = self.avg(inputs).flatten(1)
        x = self.fc1(inputs)
        x = self.relu1(x)
        x = self.bn1(x)
        x =self.fc2(x)
        # x = self.relu2(x)
        # x = self.bn2(x)
        return x


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=8631, include_top=True):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.rep_dim = 512 * block.expansion

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        # self.se_1 = SEBlock(64 * block.expansion,64 * block.expansion//16)
        self.cv1 = nn.Conv2d(64 * block.expansion, 256, 1)
        self.fc_1 = nn.Linear(64 * block.expansion, num_classes)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # self.se_2 = SEBlock(128 * block.expansion, 128 * block.expansion // 16)
        self.cv2 = nn.Conv2d(128 * block.expansion, 256, 1)
        self.fc_2 = nn.Linear(128 * block.expansion, num_classes)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.se_3 = SEBlock(256 * block.expansion, 256 * block.expansion // 16)
        self.cv3 = nn.Conv2d(256 * block.expansion, 256, 1)
        self.fc_3 = nn.Linear(256 * block.expansion, num_classes)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.weight = Weight()
        # self.fc_4 = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        w1 = self.weight(self.cv1(x))
        a = self.avgpool(x).flatten(1)
        a = self.fc_1(a)*w1

        x = self.layer2(x)
        w2 = self.weight(self.cv2(x))
        b = self.avgpool(x).flatten(1)
        b = self.fc_2(b)*w2

        x = self.layer3(x)
        w3 = self.weight(self.cv3(x))
        c = self.avgpool(x).flatten(1)
        c = self.fc_3(c)*w3

        x = self.layer4(x)
        # d = self.avgpool(x).flatten(1)
        # d = self.fc_4(c)
        return x, a+b+c

if __name__ == '__main__':
    res18 = ResNet(block=BasicBlock, n_blocks=[2, 2, 2, 2], channels=[64, 128, 256, 512], output_dim=1000)
    input = torch.randn(1, 3, 224, 224)
    output = res18(input)
    print(output.size())