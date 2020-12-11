from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.autograd import Variable
import math
from .meta_efficientnet.meta import *
import pdb
import pretrainedmodels
import torchvision

class MetaTranserModel(MetaModule):
    def __init_with_imagenet(self, baseModel):
        from .efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained(self.model_name)
        baseModel.copyWeight(model.state_dict())

    def getBase(self,pretained):
        from .meta_efficientnet import MetaEfficientNet
        baseModel = MetaEfficientNet.from_name(self.model_name)
        if pretained:
            self.__init_with_imagenet(baseModel)
        return baseModel

    def __init__(self, model_name, num_classes=1,pretained = True):
        super(MetaTranserModel, self).__init__()
        self.model_name = model_name
        if model_name.startswith('efficientnet'):
            self.base = self.getBase(pretained)
            self.num_ftrs = self.base._fc.in_features
            self.base._fc = MetaLinear(self.num_ftrs, num_classes)

    def forward(self, x):
        output,feature = self.base(x)
        return output,feature

class FNet(MetaModule):
    def __init__(self,in_channel):
        super(FNet, self).__init__()
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                MetaConv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                MetaBatchNorm2d(inp),
                nn.ReLU(inplace=True),
                MetaConv2d(inp, oup, 1, 1, 0, bias=False),
                MetaBatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
        self.feature = nn.Sequential(
            conv_dw(in_channel,in_channel//2,1),
            conv_dw(in_channel//2,in_channel//4,1),
        )
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(0.5)
        self.bn = MetaBatchNorm1d(1)
        self.fc = MetaLinear(in_channel//4, 1)

    def forward(self, x):
        feature = self.feature(x)
        x = self._avg_pooling(feature)
        x = x.view(x.size(0), -1)
        x = self._dropout(x)
        output = self.fc(x)
        output = self.bn(output)
        return output

if __name__ == '__main__':
    model = MetaTranserModel("efficientnet-b0",0)
