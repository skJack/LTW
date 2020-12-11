import os
import sys
import errno
import os.path as osp
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import log_loss
from collections import OrderedDict


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def evaluate(submission, metadata=None):
    if metadata is None:
        metadata = pd.read_csv('../train_faces/metadata.csv', low_memory=False)
        metadata = metadata.set_index('filename',drop=False)
    submission = pd.read_csv(submission)
    y_true = metadata.loc[submission.filename].label.apply(lambda x: 1 if x=='FAKE' else 0).to_list()
    y_pred = submission.label.to_list()
    score = log_loss(y_true, y_pred)
    return score


def save_checkpoint(state, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)


def load_weights(model_path):
    model_weight = torch.load(model_path)
    new_state_dict = OrderedDict()
    for key in model_weight.keys():
        if 'pointwise' in key:
            new_state_dict['model.'+key] = model_weight[key].unsqueeze(-1).unsqueeze(-1)
        elif 'fc' in key:
            continue
        else:
            new_state_dict['model.'+key] = model_weight[key]
    return new_state_dict
