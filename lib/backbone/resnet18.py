import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.dropblock import DropBlock
from lib.config import cfg


def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


# Simple ResNet Block
class SimpleBlock(nn.Module):
    def __init__(self, indim, outdim, half_res, drop_rate, block_size, drop_block=False):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.C1 = nn.Conv2d(indim, outdim, kernel_size=3,
                            stride=2 if half_res else 1, padding=1, bias=False)
        self.BN1 = nn.BatchNorm2d(outdim)
        self.C2 = nn.Conv2d(
            outdim, outdim, kernel_size=3, padding=1, bias=False)
        self.BN2 = nn.BatchNorm2d(outdim)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim != outdim:
            self.shortcut = nn.Conv2d(
                indim, outdim, 1, 2 if half_res else 1, bias=False)
            self.BNshortcut = nn.BatchNorm2d(outdim)

            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)

        # drop block
        self.drop_rate = drop_rate
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)

    def forward(self, x):
        # get the num of batches
        num_batches_tracked = int(self.BN1.num_batches_tracked.cpu().data)

        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(
            self.shortcut(x))
        out = out + short_out
        out = self.relu2(out)

        if self.drop_rate > 0:
            if self.drop_block:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20 * 2000) * num_batches_tracked, 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size ** 2 * feat_size ** 2 / (feat_size - self.block_size + 1) ** 2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training)

        return out


class ResNet18(nn.Module):
    def __init__(self, block, list_of_out_dims, flatten=True, drop_rate=0.1, dropblock_size=5):
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet18, self).__init__()
        conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        bn1 = nn.BatchNorm2d(64)
        relu = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        init_layer(conv1)
        init_layer(bn1)
        trunk = [conv1, bn1, relu, pool1]
        self.trunk = nn.Sequential(*trunk)
        self.layer1 = self._make_layer(block, 64, list_of_out_dims[0], half_res=False,
                                       drop_rate=drop_rate, dropblock_size=dropblock_size)
        self.layer2 = self._make_layer(block, list_of_out_dims[0], list_of_out_dims[1], half_res=True,
                                       drop_rate=drop_rate, dropblock_size=dropblock_size)
        self.layer3 = self._make_layer(block, list_of_out_dims[1], list_of_out_dims[2], half_res=True,
                                       drop_rate=drop_rate, dropblock_size=dropblock_size, drop_block=True)
        self.layer4 = self._make_layer(block, list_of_out_dims[2], list_of_out_dims[3], half_res=True,
                                       drop_rate=drop_rate, dropblock_size=dropblock_size, drop_block=True)

        if flatten:
            self.avgpool = nn.AvgPool2d(7)
            self.Flatten = Flatten()
            self.final_feat_dim = list_of_out_dims[3]
        else:
            self.final_feat_dim = [list_of_out_dims[3], 7, 7]
        self.flatten = flatten
        self.num_batches_tracked = 0

    def _make_layer(self, block, indim, outdim, half_res=False, drop_rate=0.1, dropblock_size=5, drop_block=False):
        layers = [block(indim, outdim, half_res, drop_rate, dropblock_size, drop_block=drop_block),
                  block(outdim, outdim, False, drop_rate, dropblock_size, drop_block=drop_block)]

        return nn.Sequential(*layers)

    def forward(self, x):
        self.num_batches_tracked += 1
        x = self.trunk(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.flatten:
            x = self.avgpool(x)
            x = self.Flatten(x)
        return x


def resnet18(flatten=True):
    return ResNet18(SimpleBlock, [64, 128, 256, 512], flatten,
                    drop_rate=cfg.train.drop_rate, dropblock_size=cfg.train.dropblock_size)
