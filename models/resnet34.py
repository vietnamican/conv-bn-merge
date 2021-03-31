import torch
from torch import nn
from typing import Type, Any, Callable, Union, List, Optional

from .base import Base, ConvBatchNormRelu, BasicBlock


class Resnet34(Base):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBatchNormRelu(
            3, 64, kernel_size=7, padding=3, stride=1, bias=False)
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64),
            BasicBlock(64, 64)
        )
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, downsample=True,
                       stride=2),
            BasicBlock(128, 128),
            BasicBlock(128, 128),
            BasicBlock(128, 128)
        )
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, downsample=True,
                       stride=2),
            BasicBlock(256, 256),
            BasicBlock(256, 256),
            BasicBlock(256, 256),
            BasicBlock(256, 256),
            BasicBlock(256, 256)
        )
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, downsample=True,
                       stride=2),
            BasicBlock(512, 512),
            BasicBlock(512, 512)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def release(self):
        is_self = True
        for module in self.modules():
            if is_self:
                is_self = False
                continue
            if hasattr(module, 'release'):
                module.release()
