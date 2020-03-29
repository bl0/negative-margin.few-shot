# This code is modified from https://github.com/nupurkmr9/S2M2_fewshot

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.config import cfg


class WRNBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super(WRNBlock, self).__init__()
        self.BN1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU()
        self.C1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.BN2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU()
        self.C2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.BN1(x))
        else:
            out = self.relu1(self.BN1(x))
        out = self.relu2(self.BN2(self.C1(out if self.equalInOut else x)))
        if self.drop_rate > 0.0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.C2(out)
        short_out = x if self.equalInOut else self.convShortcut(x)
        out = out + short_out
        return out


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor=1, stride=1, drop_rate=0.5):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = WRNBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        # 1st layer
        self.layer1 = self._make_layer(n, nChannels[0], nChannels[1], block, stride, drop_rate)
        # 2nd layer
        self.layer2 = self._make_layer(n, nChannels[1], nChannels[2], block, 2, drop_rate)
        # 3rd layer
        self.layer3 = self._make_layer(n, nChannels[2], nChannels[3], block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.nChannels = nChannels[3]
        self.final_feat_dim = 640
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_layer(self, nb_layers, in_planes, out_planes, block, stride, drop_rate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.relu(self.bn1(x))
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        return x


def WideResNet28_10():
    return WideResNet(28, 10, drop_rate=cfg.train.drop_rate)
