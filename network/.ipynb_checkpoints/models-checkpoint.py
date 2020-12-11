"""

Author: Andreas Rössler
"""
import os
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pretrainedmodels


class TransferModel(nn.Module):
    """
    Simple transfer learning model that takes an imagenet pretrained model with
    a fc layer as base model and retrains a new fc layer for num_classes
    """
    def __init__(self, model_name, num_classes=2, dropout=0.0, pretrained=True):
        super(TransferModel, self).__init__()
        self.model_name = model_name
        if model_name == 'xception':
            self.image_size = 299
            if pretrained: pretrained = 'imagenet' 
            self.model = pretrainedmodels.xception(pretrained=pretrained)            
            # Replace fc
            num_ftrs = self.model.last_linear.in_features
            if not dropout:
                self.model.last_linear = nn.Linear(num_ftrs, num_classes)
            else:
                print('Using dropout', dropout)
                self.model.last_linear = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_classes)
                )
        elif model_name == 'resnet50' or model_name == 'resnet18':
            self.image_size = 224
            if model_name == 'resnet50':
                self.model = torchvision.models.resnet50(pretrained)
            if model_name == 'resnet18':
                self.model = torchvision.models.resnet18(pretrained)
            # Replace fc
            num_ftrs = self.model.fc.in_features
            if not dropout:
                self.model.fc = nn.Linear(num_ftrs, num_classes)
            else:
                self.model.fc = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_classes)
                )
        elif model_name == 'InceptionResnetV1':
            from facenet_pytorch import InceptionResnetV1
            self.image_size = 160
            if pretrained: pretrained = 'vggface2'
            self.model = InceptionResnetV1(pretrained=pretrained, classify=True, num_classes=2)
        elif model_name == 'SPPNet':
            from network.SPPNet import SPPNet
            self.image_size = 224
            self.model = SPPNet(backbone=50, num_class=num_classes, pretrained=pretrained)
        elif model_name == 'ResNetXt':
            self.image_size = 224
            self.model =  torchvision.models.resnext50_32x4d(pretrained=pretrained)
            self.model.fc = nn.Linear(2048, num_classes)
        else:
            raise Exception('Choose valid model, e.g. resnet50')

    def set_trainable_up_to(self, boolean, layername="Conv2d_4a_3x3"):
        """
        Freezes all layers below a specific layer and sets the following layers
        to true if boolean else only the fully connected final layer
        :param boolean:
        :param layername: depends on network, for inception e.g. Conv2d_4a_3x3
        :return:
        """
        # Stage-1: freeze all the layers
        if layername is None:
            for i, param in self.model.named_parameters():
                param.requires_grad = True
                return
        else:
            for i, param in self.model.named_parameters():
                param.requires_grad = False
        if boolean:
            # Make all layers following the layername layer trainable
            ct = []
            found = False
            for name, child in self.model.named_children():
                if layername in ct:
                    found = True
                    for params in child.parameters():
                        params.requires_grad = True
                ct.append(name)
            if not found:
                raise Exception('Layer not found, cant finetune!'.format(
                    layername))
        else:
            if self.model_name == 'xception':
                # Make fc trainable
                for param in self.model.last_linear.parameters():
                    param.requires_grad = True

            else:
                # Make fc trainable
                for param in self.model.fc.parameters():
                    param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    model = TransferModel('SPPNet', num_classes=2)
    print(model)
    model = model.cuda()
    from torchsummary import summary
    input_s = (3, model.image_size, model.image_size)
    print(summary(model, input_s))
