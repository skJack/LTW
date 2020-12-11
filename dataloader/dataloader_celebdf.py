import os
import cv2
import glob
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.utils.data as data
import pdb
from torchvision import transforms as torch_transforms


class CeleDF(data.Dataset):
    """DFDC Detection Dataset Object
    input is image, target is annotation
    Arguments:
        root (string): filepath to VOCdevkit folder.
        train (bool): imageset to use (eg. 'train', test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'DFDC')
    """

    def __init__(self, train=True, frame_nums=5,
                 transform=None, target_transform=None,data_root = '/media/sdc/datasets/celedf/', dataset_name='CELEDF'):
        self.train = train
        self.frame_nums = frame_nums
        self.transform = transform
        self.target_transform = target_transform
        self.type = dataset_name
        self.data_root = data_root
        self.datas = self.init()


    def __getitem__(self, index):
        img_path, target,folder = self.datas[index]


        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target,folder,img_path

    def __len__(self):
        return len(self.datas)

    def init(self):
        datas = []
        path_list = []

        Deepfakes_path = f'{self.data_root}/Celeb-synthesis-mtcnn/*'
        original_path = f'{self.data_root}/Celeb-real-mtcnn/*'
        youtube_path = f'{self.data_root}/YouTube-real-mtcnn/*'
        split_path = f'{self.data_root}/List_of_testing_videos.txt'
        data_root = self.data_root
        file = open(split_path)
        with open(split_path,'r') as f:
            self.raw_list = f.read().splitlines()
        test_list = [x.split(" ")[1] for x in self.raw_list]
        test_list = [data_root+x.split("/")[0]+'-mtcnn/'+x.split("/")[1][:-4] for x in test_list]

        path_list.append(Deepfakes_path)
        path_list.append(original_path)
        path_list.append(youtube_path)
        self.fake_num = 0
        self.real_num = 0
        if self.train == True:
            for path in path_list:
                folder_paths_all = glob.glob(path)
                folder_paths = np.setdiff1d(folder_paths_all,test_list)
                label_str = path.split('/')[5]
                

                label = 1 if label_str == 'Celeb-synthesis-mtcnn' else 0
                for folder in folder_paths:
                    if label == 1:
                        self.fake_num = self.fake_num+1
                    else:
                        self.real_num = self.real_num+1
                    face_paths = glob.glob(os.path.join(folder, '*.png'))

                    if len(face_paths) < 5:
                        continue
                    if len(face_paths) > self.frame_nums:
                        face_paths = np.array(sorted(face_paths, key=lambda x: int(x.split('/')[-1].split('.')[0][-4:])))
                        ind = np.linspace(0, len(face_paths) - 1, self.frame_nums, endpoint=True, dtype=np.int)
                        face_paths = face_paths[ind]

                    datas.extend([[face_path, label,folder] for face_path in face_paths])
        else:
            folder_paths = test_list
            
            for folder in folder_paths:
                label_str = folder.split('/')[5]
                label = 1 if label_str == 'Celeb-synthesis-mtcnn' else 0
                
                if label == 1:
                        self.fake_num = self.fake_num+1
                else:
                        self.real_num = self.real_num+1
                face_paths = glob.glob(os.path.join(folder, '*.png'))

                if len(face_paths) < 5:
                    continue
                if len(face_paths) > self.frame_nums:
                    face_paths = np.array(sorted(face_paths, key=lambda x: int(x.split('/')[-1].split('.')[0][-4:])))
                    ind = np.linspace(0, len(face_paths) - 1, self.frame_nums, endpoint=True, dtype=np.int)
                    face_paths = face_paths[ind]

                datas.extend([[face_path, label,folder] for face_path in face_paths])
            
        return datas


def load_FaceForens(train_size=128, test_size=1, transform=None):
    if transform == None:
        from transforms import ResNet_default_data_transforms
        transform = ResNet_default_data_transforms

    trans = transform['train']
    train_dataset = DFDCDetection(train=True, frame_nums=5, transform=trans)
    train_dataloader = data.DataLoader(train_dataset, batch_size=train_size, shuffle=True, num_workers=4)

    trans = transform['test']
    test_dataset = DFDCDetection(train=False, frame_nums=12, transform=trans)
    test_dataloader = data.DataLoader(test_dataset, batch_size=test_size, shuffle=True, num_workers=4)

    return train_dataloader, test_dataloader


if __name__ == '__main__':
    from transform import SPPNet_default_data_transforms as transform_data

    trans = transform_data['train']
    train_dataset = CeleDF(train=True, frame_nums=3, transform=trans)
    print(f"train dataset real :{train_dataset.real_num},fake :{train_dataset.fake_num}")
    train_dataloader = data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    #pdb.set_trace()
    print(len(train_dataset))
    for i, datas in enumerate(train_dataloader):
        print(i, datas[0].shape, datas[1].shape)
        break

    trans = transform_data['test']
    test_dataset = CeleDF(train=False, frame_nums=3, transform=trans)
    print(f"test dataset real :{test_dataset.real_num},fake :{test_dataset.fake_num}")

    test_dataloader = data.DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=4)
    print(len(train_dataset))
    print(len(test_dataset))
    for i, datas in enumerate(test_dataloader):
        print(i, datas[0].shape, datas[1].shape)
        print(datas[1])
        break
