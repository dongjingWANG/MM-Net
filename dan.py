import pickle

import torch.nn.init as init
from model.resnet1 import *
from model.se_block import SEBlock


class DAN(nn.Module):
    def __init__(self, num_class=7):
        super(DAN, self).__init__()

        # resnet = resnet18(pretrained='msceleb')
        #
        # if pretrained:
        #     checkpoint = torch.load('/data/zhujinlin/pre-train-weight/resnet18_msceleb.pth')
        #     resnet.load_state_dict(checkpoint['state_dict'], strict=True)
        self.resnet = ResNet(Bottleneck, [3, 4, 6, 3],num_classes=num_class)
        with open('/data/wdj_zhujl/pre-train-weight/vgg_msceleb_resnet50_ft_weight.pkl', 'rb') as f:
        # with open('/data/zjl/nfs/vgg_msceleb_resnet50_ft_weight.pkl', 'rb') as f:
            obj = f.read()
        state_dict = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
        self.resnet.load_state_dict(state_dict, strict=False)

        self.se = SEBlock(self.resnet.rep_dim, self.resnet.rep_dim//16)
        self.rdb = ResidualDilatedBlock(self.resnet.rep_dim)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.resnet.rep_dim, num_class)
        self.fc1 = nn.Linear(self.resnet.rep_dim, 128)
        self.bn = nn.BatchNorm1d(num_class)

    def forward(self, x):
        x, t = self.resnet(x)
        out = self.avg(x).flatten(1)
        feat = self.fc1(out)
        x = self.se(x)
        x = self.rdb(x)
        out = self.avg(x).flatten(1)
        out = self.fc(out)
        out = self.bn(out+t)
        return out, feat





class ResidualDilatedBlock(nn.Module):
    __constants__ = ['downsample']

    def __init__(self, planes=512, stride=1, dilation_size=[1, 2, 3], downsample=None):
        super(ResidualDilatedBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        # self.conv1 = conv3x3(inplanes, planes, dilation=dilation_size[0], stride=stride,)
        self.conv1 = nn.Conv2d(planes, planes, 3, stride=1, dilation=dilation_size[0], padding=dilation_size[0])
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, dilation=dilation_size[1], padding=dilation_size[1])
        self.conv3 = nn.Conv2d(planes, planes, 3, stride=1, dilation=dilation_size[2], padding=dilation_size[2])
        self.bn1 = norm_layer(planes)
        self.bn2 = norm_layer(planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out2 = self.conv2(x)
        out2 = self.bn2(out2)
        out3 = self.conv3(x)
        out3 = self.bn3(out3)
        y = self.relu(out1 + out2 + out3)
        return y


