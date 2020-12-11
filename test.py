import time
import os
import sys
import argparse
import shutil
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from easydict import EasyDict
from network import model_selection_meta as model_selection
from torchvision.utils import make_grid
from torch.nn import functional as F

from dataloader.dataloader_ffpp import FFpp
from dataloader.dataloader_celebdf import CeleDF
from dataloader.dataloader_dfdc import DFDCDetection
from dataloader.transform import get_transform
from utils import *
from utils.config_test import *
import pdb
from network import FNet
from thop import clever_format
from thop import profile




def model_forward(image, model, post_function=nn.Sigmoid(),feat = False):
    """
    Predicts the label of an input image. Preprocesses the input image and
    casts it to cuda if required

    :param image: numpy image
    :param model: torch model with linear layer at the end
    :param post_function: e.g., softmax
    :return: prediction (1 = fake, 0 = real)
    """

    # Model prediction
    output,feature = model(image)
    output = post_function(output)
    output = output.squeeze()
    
    prediction = (output >= 0.5).float()
    if feat==False:
        return prediction, output
    else:
        return prediction, output,feature




def test(data_loader, model, device):
    model.eval()
    acces = []
    losses = 0.0
    label_list = []
    output_list = []
    criterion = torch.nn.BCELoss().to(device)
    wrongimg = []
    for i, datas in enumerate(tqdm(data_loader)):
        images = datas[0].to(device)#3,3,224,224
        targets = datas[1].float().to(device)
        
        with torch.no_grad():
            prediction, output = model_forward(images, model)
        label_list.extend(targets.cpu().numpy().tolist())
        output_list.extend(output.cpu().numpy().tolist())

        acces.append((targets == prediction).cpu().numpy())
        loss = criterion(output, targets).item()
        losses += loss
    metrics = EasyDict()
    metrics.acc = np.mean(acces)
    eer,TPRs, auc,scaler = cal_metric(label_list,output_list,False)

    metrics.loss = losses / len(data_loader)
    metrics.auc = auc
    metrics.tpr = TPRs
    metrics.eer = eer
    model.train()
    
    return metrics


def main():
    
    sys.stdout = Logger(os.path.join('./test_output', f'{type}_{compress}_{model_name}_{input_size}.log'))
    device = 'cuda:0' if torch.cuda.is_available == True else 'cpu'
    
    model = model_selection(model_name=model_name, num_classes=1).to(device)
    if model_path is not None:    
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        print(f'resume model from {model_path}')
    else:
        print('No model found, initializing random model.')

    _preproc = get_transform(input_size)['test']

    cele_test_dataset = CeleDF(train = False, frame_nums=frame_nums, transform=_preproc,data_root = celebdf_path)#98855)
    df_test_dataset = FFpp(split='test', frame_nums=frame_nums, transform=_preproc,detect_name = detect_name,compress = compress,type = "Deepfakes")
    f2f_test_dataset = FFpp(split='test', frame_nums=frame_nums, transform=_preproc,detect_name = detect_name,compress = compress,type = 'Face2Face')
    fs_test_dataset = FFpp(split='test', frame_nums=frame_nums, transform=_preproc,detect_name = detect_name,compress = compress,type = 'FaceSwap')
    nt_test_dataset = FFpp(split='test', frame_nums=frame_nums, transform=_preproc,detect_name = detect_name,compress = compress,type = 'NeuralTextures')
    dfdc_test_dataset = DFDCDetection(root = dfdc_path, train=False, frame_nums=frame_nums, transform=_preproc)
    df_test_dataloader = data.DataLoader(df_test_dataset, batch_size=2, shuffle=False, num_workers=8)
    f2f_test_dataloader = data.DataLoader(f2f_test_dataset, batch_size=2, shuffle=False, num_workers=8)
    fs_test_dataloader = data.DataLoader(fs_test_dataset, batch_size=2, shuffle=False, num_workers=8)
    nt_test_dataloader = data.DataLoader(nt_test_dataset, batch_size=2, shuffle=False, num_workers=8)
    cele_test_dataloader = data.DataLoader(cele_test_dataset, batch_size=2, shuffle=True, num_workers=8)
    dfdc_test_dataloader = data.DataLoader(dfdc_test_dataset, batch_size=3, shuffle=True, num_workers=8)
   
    
    df_metrics = test(df_test_dataloader, model, device)
    f2f_metrics = test(f2f_test_dataloader, model, device)
    fs_metrics = test(fs_test_dataloader, model, device)
    nt_metrics = test(nt_test_dataloader, model, device)
    celedf_metrics = test(cele_test_dataloader,model,device)
    dfdc_metrics = test(dfdc_test_dataloader,model,device)

    metrics_list = [df_metrics,f2f_metrics,fs_metrics,nt_metrics]
    avg_metrics = EasyDict()
    all_avg_metrics = EasyDict()
    avg_metrics.acc = (df_metrics.acc+f2f_metrics.acc+fs_metrics.acc+nt_metrics.acc)/4
    avg_metrics.loss = (df_metrics.loss+f2f_metrics.loss+fs_metrics.loss+nt_metrics.loss)/4
    avg_metrics.auc = (df_metrics.auc+f2f_metrics.auc+fs_metrics.auc+nt_metrics.auc)/4
    avg_metrics.eer = (df_metrics.eer+f2f_metrics.eer+fs_metrics.eer+nt_metrics.eer)/4

    all_avg_metrics.acc = (df_metrics.acc+f2f_metrics.acc+fs_metrics.acc+nt_metrics.acc+celedf_metrics.acc+dfdc_metrics.acc)/6
    all_avg_metrics.loss = (df_metrics.loss+f2f_metrics.loss+fs_metrics.loss+nt_metrics.loss+celedf_metrics.loss+dfdc_metrics.loss)/6
    all_avg_metrics.auc = (df_metrics.auc+f2f_metrics.auc+fs_metrics.auc+nt_metrics.auc+celedf_metrics.auc+dfdc_metrics.auc)/6
    all_avg_metrics.eer = (df_metrics.eer+f2f_metrics.eer+fs_metrics.eer+nt_metrics.eer+celedf_metrics.eer+dfdc_metrics.eer)/6

    print(f"df acc:{df_metrics.acc:.5f},loss:{df_metrics.loss:.3f},auc:{df_metrics.auc:.3f},eer:{df_metrics.eer:.3f}")
    print(f"f2f acc:{f2f_metrics.acc:.3f},loss:{f2f_metrics.loss:.3f},auc:{f2f_metrics.auc:.3f},eer:{f2f_metrics.eer:.3f}")
    print(f"fs acc:{fs_metrics.acc:.3f},loss:{fs_metrics.loss:.3f},auc:{fs_metrics.auc:.3f},eer:{fs_metrics.eer:.3f}")
    print(f"nt acc:{nt_metrics.acc:.3f},loss:{nt_metrics.loss:.3f},auc:{nt_metrics.auc:.3f},eer:{nt_metrics.eer:.3f}")
    print(f"avg acc:{avg_metrics.acc:.3f},loss:{avg_metrics.loss:.3f},auc:{avg_metrics.auc:.3f},eer:{avg_metrics.eer:.3f}")
    print(f"celedf acc:{celedf_metrics.acc:.3f},loss:{celedf_metrics.loss:.3f},auc:{celedf_metrics.auc:.3f},eer:{celedf_metrics.eer:.3f}")
    print(f"dfdc acc:{dfdc_metrics.acc:.3f},loss:{dfdc_metrics.loss:.3f},auc:{dfdc_metrics.auc:.3f},eer:{dfdc_metrics.eer:.3f}")
    print(f"all_avg acc:{all_avg_metrics.acc:.3f},loss:{all_avg_metrics.loss:.3f},auc:{all_avg_metrics.auc:.3f},eer:{all_avg_metrics.eer:.3f}")



main()