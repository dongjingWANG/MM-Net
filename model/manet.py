import tabnanny

import torch.cuda

from .attention import *


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        # self.cbam = CBAM(planes, 8)
        # self.cbam = ResidualDilatedBlock(planes, planes, 1, [1, 2, 3])
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out




class AttentionBlock(nn.Module):
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(AttentionBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.cbam = CBAM(planes, 16)
        # self.cbam = ResidualDilatedBlock(planes, planes, 1, [1, 2, 3])
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResidualDilatedBlock(nn.Module):
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, dilation_size=[1, 2, 3], downsample=None):
        super(ResidualDilatedBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        # self.conv1 = conv3x3(inplanes, planes, dilation=dilation_size[0], stride=stride,)
        self.conv1 = nn.Conv2d(planes, planes, 3, stride=1, dilation=dilation_size[0], padding=dilation_size[0])
        self.bn1 = norm_layer(planes)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, dilation=dilation_size[1], padding=dilation_size[1])
        self.bn2 = norm_layer(planes)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(planes, planes, 3, stride=1, dilation=dilation_size[2], padding=dilation_size[2])
        self.bn3 = norm_layer(planes)
        self.relu3 = nn.ReLU()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)


        return out


class MANet(nn.Module):

    def __init__(self, block_b, block_a, block_d, layers, num_classes=7):

        super(MANet, self).__init__()
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.layer1 = self._make_layer(block_b, 3, 64, layers[0], stride=2)
        self.r1 = ResidualDilatedBlock(64, 64, 1, [1, 2, 3])
        self.layer2 = self._make_layer(block_b, 64, 128, layers[1], stride=2)
        self.r2 = ResidualDilatedBlock(128, 128, 1, [1, 2, 3])
        # In this branch, each BasicBlock replaced by AttentiveBlock.
        self.layer3_1_p1 = self._make_layer(block_a, 128, 256, layers[2], stride=2)
        self.layer4_1_p1 = self._make_layer(block_a, 256, 512, layers[3], stride=2)

        self.layer3_1_p2 = self._make_layer(block_a, 128, 256, layers[2], stride=2)
        self.layer4_1_p2 = self._make_layer(block_a, 256, 512, layers[3], stride=2)

        self.layer3_1_p3 = self._make_layer(block_a, 128, 256, layers[2], stride=2)
        self.layer4_1_p3 = self._make_layer(block_a, 256, 512, layers[3], stride=2)

        self.layer3_1_p4 = self._make_layer(block_a, 128, 256, layers[2], stride=2)
        self.layer4_1_p4 = self._make_layer(block_a, 256, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_1 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(conv1x1(inplanes, planes, stride), norm_layer(planes))
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        branch_1_p1_out = x[:, :, 0:112, 0:112]
        branch_1_p2_out = x[:, :, 0:112, 112:224]
        branch_1_p3_out = x[:, :, 112:224, 0:112]
        branch_1_p4_out = x[:, :, 112:224, 112:224]


        branch_1_p1_out = self.layer1(branch_1_p1_out)
        branch_1_p1_out = self.r1(branch_1_p1_out)
        branch_1_p1_out = self.layer2(branch_1_p1_out)
        branch_1_p1_out = self.r2(branch_1_p1_out)
        branch_1_p1_out = self.layer3_1_p1(branch_1_p1_out)
        branch_1_p1_out = self.layer4_1_p1(branch_1_p1_out)


        branch_1_p2_out = self.layer1(branch_1_p2_out)
        branch_1_p2_out = self.r1(branch_1_p2_out)
        branch_1_p2_out = self.layer2(branch_1_p2_out)
        branch_1_p2_out = self.r2(branch_1_p2_out)
        branch_1_p2_out = self.layer3_1_p2(branch_1_p2_out)
        branch_1_p2_out = self.layer4_1_p2(branch_1_p2_out)


        branch_1_p3_out = self.layer1(branch_1_p3_out)
        branch_1_p3_out = self.r1(branch_1_p3_out)
        branch_1_p3_out = self.layer2(branch_1_p3_out)
        branch_1_p3_out = self.r2(branch_1_p3_out)
        branch_1_p3_out = self.layer3_1_p3(branch_1_p3_out)
        branch_1_p3_out = self.layer4_1_p3(branch_1_p3_out)


        branch_1_p4_out = self.layer1(branch_1_p4_out)
        branch_1_p4_out = self.r1(branch_1_p4_out)
        branch_1_p4_out = self.layer2(branch_1_p4_out)
        branch_1_p4_out = self.r2(branch_1_p4_out)
        branch_1_p4_out = self.layer3_1_p4(branch_1_p4_out)
        branch_1_p4_out = self.layer4_1_p4(branch_1_p4_out)

        branch_1_out_1 = torch.cat([branch_1_p1_out, branch_1_p2_out], dim=3)
        branch_1_out_2 = torch.cat([branch_1_p3_out, branch_1_p4_out], dim=3)
        branch_1_out = torch.cat([branch_1_out_1, branch_1_out_2], dim=2)

        branch_1_out = self.avgpool(branch_1_out)
        branch_1_out = torch.flatten(branch_1_out, 1)
        branch_1_out = self.fc_1(branch_1_out)

        return branch_1_out

    def forward(self, x):
        return self._forward_impl(x)


def manet(num_classes=7):
    return MANet(block_b=BasicBlock, block_a=AttentionBlock, block_d=ResidualDilatedBlock, layers=[2, 2, 2, 2], num_classes=num_classes)
