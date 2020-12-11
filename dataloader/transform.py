"""

Author: Andreas Rössler
"""
from torchvision import transforms
import sys
sys.path.append('../')
from utils.config import *



def get_transform(input_size):
    SPPNet_default_data_transforms = {
    'train': transforms.Compose([
        # transforms.CenterCrop((260, 260)),
        transforms.Resize((input_size, input_size)),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomCrop(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #transforms.RandomErasing()
    ]),
    'val': transforms.Compose([
        # transforms.CenterCrop((260, 260)),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        # transforms.CenterCrop((260, 260)),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }
    return SPPNet_default_data_transforms

