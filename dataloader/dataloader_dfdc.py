import os
import cv2
import glob
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.utils.data as data
import pdb

class DFDCDetection(data.Dataset):
    def __init__(self, root, train=True, frame_nums=5, 
                transform=None, target_transform=None, dataset_name='DFDC'):
        self.root = root
        self.train = train
        self.frame_nums = frame_nums
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self.datas = self._split_data()


    def __getitem__(self, index):
        img_path, target, video_fn = self.datas[index]

        img = Image.open(img_path[0])
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, video_fn
       
    def __len__(self):
        return len(self.datas)

    def _sampler(self, sample_datas):
        datas = []
        for fn, label, folder in sample_datas:
            if type(fn) != str:  # 会有nan
                continue
            face_paths = glob.glob(os.path.join(self.root, folder, fn.split('.')[0], '*.jpg'))
            if len(face_paths) > self.frame_nums:
                face_paths = np.array(sorted(face_paths, key=lambda x: int(x.split('/')[-1].split('.')[0])))#排序
                ind = np.linspace(0, len(face_paths) - 1, self.frame_nums, endpoint=True, dtype=np.int)  # 生成均匀分布的样本
                face_paths = face_paths[ind]
            if self.train:
                datas.extend([[face_path, label, fn] for face_path in face_paths])
            else:
                datas.append([face_paths, label, fn])
        return datas

    def _split_data(self, test_pos_size=2000, seed=0):
        ## load dataset metadata
        metadata = pd.read_csv(os.path.join(self.root, 'metadata.csv'), low_memory=False)
        metadata = metadata.set_index('filename', drop=False)
        ## filter out videos without images
        tmp = self.root + "/*/*"
        # video_paths = glob.glob(str(self.root/'*'/'*'))
        video_paths = glob.glob(tmp)
        vn = [os.path.basename(x) + '.mp4' for x in video_paths if len(os.listdir(x)) > 0]
        metadata = metadata.loc[vn]  
        ## random permutation
        metadata['label'] = metadata['label'].map({'FAKE': 1, 'REAL': 0})
        metadata = metadata[['filename', 'label', 'original', 'folder']]
        metadata = metadata.sample(frac=1, random_state=seed) 

        reals = metadata[metadata['original'].eq('NAN')].drop('original', axis=1)

        fakes = metadata.drop(reals.filename).set_index('original')
        if self.train:
            train_pos = reals[test_pos_size:] 
            train_pos = train_pos[train_pos.filename.isin(fakes.index)] 

            train_neg = fakes.loc[train_pos.filename]
            datas = self._sampler_new(train_pos,train_neg)
        else:
            test_pos = reals[:test_pos_size]
            test_neg = fakes.loc[test_pos.filename].groupby(level=0, group_keys=False).apply(
                lambda x: x.sample(1, random_state=seed))
            test_datas = np.concatenate([test_pos.values, test_neg.values])

            datas = self._sampler(test_datas)
        return datas

