import os
import cv2
import glob
import numpy as np
import pandas as pd
from PIL import Image
import random
import torch
import torch.utils.data as data
import pdb
from torchvision import transforms as torch_transforms
import json
import sys
sys.path.append("./")
class FFpp(data.Dataset):
    def __init__(self, split='train', frame_nums=5,
                 transform=None, target_transform=None,detect_name='mtcnn',compress = 'c40',type = 'Face2Face',pair = False,original_path="/media/sdc/datasets/faceforensics++/original_sequences/youtube",fake_path="/media/sdc/datasets/faceforensics++/manipulated_sequences"
                 ,split_path = "./dataloader/splits"):
        self.split = split
        self.frame_nums = frame_nums
        self.transform = transform
        self.target_transform = target_transform
        self.detect_name = detect_name
        self.compress = compress
        self.type = type
        self.real_id = []
        self.fake_id = []
        self.fake_id_dict = {}
        self.meta_type = type
        self.original_path = original_path
        self.fake_path = fake_path
        self.pair = pair

        self.image_id = get_video_ids(split,split_path)
        self._get_id()
        self._convert_dict()
        
        self.datas = self.init()

    def _convert_dict(self):
        for path in self.fake_id:
            self.fake_id_dict[path.split("_")[0]] = path
    def set_meta_type(self,meta_type = None):
        self.meta_type = meta_type
    
    def _get_id(self):
        for id in self.image_id:
            if len(id) == 3:
                self.real_id.append(id)
            else:
                self.fake_id.append(id)
        self.real_id = sorted(self.real_id)
        self.fake_id = sorted(self.fake_id)

    def delete_real(self):
        del self.datas[self.num_fake:]
        
    def __getitem__(self, index):
        img_path, target,folder = self.datas[index]
        if self.pair == True:
            '''
            meta-split stragey
            '''
            if target == 1:
                origin_index = img_path.split('/')[-2].split('_')[0]
                frame_index = img_path.split('/')[-1].split('_')[-1]
                opposite_path = f'{self.original_path}/{self.compress}/{self.detect_name}/'+origin_index+"/"+origin_index+"_"+frame_index
                meta_path = img_path.replace(img_path.split('/')[-5],self.meta_type)
                if not os.path.exists(opposite_path):
                    original_folder = f'{self.original_path}/{self.compress}/{self.detect_name}/'+origin_index+"/"
                    face_paths = glob.glob(os.path.join(original_folder, '*.png'))
                    opposite_path = random.choice(face_paths)
                if not os.path.exists(meta_path):
                    original_folder = f'{self.fake_path}/{self.meta_type}/{self.compress}/{self.detect_name}/'+img_path.split('/')[-2]+"/"
                    face_paths = glob.glob(os.path.join(original_folder, '*.png'))
                    meta_path = random.choice(face_paths)
            if target == 0:
                origin_index = img_path.split('/')[-2]
                frame_index = img_path.split('/')[-1].split('_')[-1]
                fake_index = self.fake_id_dict[origin_index]
                target_index = fake_index.split('_')[-1]
                meta_path = img_path.replace(origin_index,target_index)
                opposite_path = f'{self.fake_path}/{self.meta_type}/{self.compress}/{self.detect_name}/'+fake_index+"/"+fake_index+"_"+frame_index
                if not os.path.exists(opposite_path):
                    original_folder = f'{self.fake_path}/{self.meta_type}/{self.compress}/{self.detect_name}/'+fake_index+"/"
                    face_paths = glob.glob(os.path.join(original_folder, '*.png'))
                    opposite_path = random.choice(face_paths)
                if not os.path.exists(meta_path):
                    original_folder = f'{self.original_path}/{self.compress}/{self.detect_name}/'+target_index+"/"
                    face_paths = glob.glob(os.path.join(original_folder, '*.png'))
                    meta_path = random.choice(face_paths)
                
            opposite_img = Image.open(opposite_path)
            meta_img = Image.open(meta_path)
            img = Image.open(img_path)
            if self.transform is not None:
                img = self.transform(img)
                opposite_img = self.transform(opposite_img)
                meta_img = self.transform(meta_img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, target,folder,img_path,opposite_img,opposite_path,meta_img,meta_path
        else:
            img = Image.open(img_path)
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, target,folder,img_path

    def cat(self,dataset,randomseed = None):
        '''
        concat two dataset
        '''
        self.datas.extend(dataset.datas)
        if randomseed!=None:
            random.seed(randomseed)
            random.shuffle(self.datas)
        return self

    def __len__(self):
        return len(self.datas)

    def init(self):
        datas = []
        path_list = []
        if self.type != 'all' and self.type!='real':
            Deepfakes_path = f'{self.fake_path}/{self.type}/{self.compress}/{self.detect_name}/'
            path_list.append(Deepfakes_path)
        elif self.type == 'all':
            for t in ['Deepfakes','Face2Face','FaceSwap','NeuralTextures']:
                Deepfakes_path = f'{self.fake_path}/{t}/{self.compress}/{self.detect_name}/'
                path_list.append(Deepfakes_path)
        
        original_path = f'{self.original_path}/{self.compress}/{self.detect_name}/'
        path_list.append(original_path)
        self.num_real = 0
        self.num_fake = 0
        for path in path_list:

            label_str = path.split('/')[5]
            label = 0 if label_str == 'original_sequences' else 1
            if label == 0:
                folder_paths = [path+id for id in self.real_id]
            else:
                folder_paths = [path + id for id in self.fake_id]
            for folder in folder_paths:
                face_paths = sorted(glob.glob(os.path.join(folder, '*.png')))
                if len(face_paths) < 5:
                    continue
                if len(face_paths) > self.frame_nums:
                    face_paths = np.array(sorted(face_paths, key=lambda x: int(x.split('/')[-1].split('.')[0])))
                    ind = np.linspace(0, len(face_paths) - 1, self.frame_nums, endpoint=True, dtype=np.int)
                    face_paths = face_paths[ind]
                if label == 0:
                    self.num_real = self.num_real+len(face_paths)
                else:
                    self.num_fake = self.num_fake+len(face_paths)
                datas.extend([[face_path, label,folder] for face_path in face_paths])
        return datas


def listdir_with_full_paths(dir_path):
    return [os.path.join(dir_path, x) for x in os.listdir(dir_path)]


def random_split(data, split):
    size = int(len(data)*split)
    random.shuffle(data)
    return data[:size], data[size:]


def get_file_name(file_path):
    return file_path.split('/')[-1]


def read_json(file_path):
    with open(file_path) as inp:
        return json.load(inp)


def get_sets(data):
    return {x[0] for x in data} | {x[1] for x in data} | {'_'.join(x) for x in data} | {'_'.join(x[::-1]) for x in data}


def get_video_ids(spl, splits_path):
    return get_sets(read_json(os.path.join(splits_path, f'{spl}.json')))


def read_train_test_val_dataset(
        dataset_dir, name, target, splits_path, **dataset_kwargs
):
    for spl in ['train', 'val', 'test']:
        import pdb; pdb.set_trace()
        video_ids = get_video_ids(spl, splits_path)
        video_paths = listdir_with_full_paths(dataset_dir)
        videos = [x for x in video_paths if get_file_name(x) in video_ids]
        dataset = ImagesDataset(videos, name, target, **dataset_kwargs)
        yield dataset


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
    from transform import xception_default_data_transforms as transform_data
    
    image_id = get_video_ids("test","/home/kesun/deepfake/ff++v2/splits/")
    real_id = []
    fake_id = []
    for id in image_id:
        if len(id)==3:
            real_id.append(id)
        else:
            fake_id.append(id)
    print(len(real_id))
    print(len(fake_id))

    trans = transform_data['train']
    #train_dataset = DFDCDetection(split='train', frame_nums=3, transform=trans)
    f2f_train_dataset = FFpp(split='val', frame_nums=3, transform=trans,detect_name = "mtcnn",compress = "c40",type = 'Face2Face',randomseed=None,pair = True)#98855
    print(f2f_train_dataset.num_real)
    print(f2f_train_dataset.num_fake)

    #f2f_train_dataset.delete_real()
    train_dataloader = data.DataLoader(f2f_train_dataset, batch_size=32, shuffle=True, num_workers=4)
    #pdb.set_trace()
    
    print(len(f2f_train_dataset.datas))
    for i, datas in enumerate(train_dataloader):
        print(i, datas[0].shape, datas[1].shape)
        print(datas[1])

    trans = transform_data['test']
    test_dataset = DFDCDetection(split="test", frame_nums=3, transform=trans)
    test_dataloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    print(len(test_dataset))

    for i, datas in enumerate(test_dataloader):
        print(i, datas[0].shape, datas[1].shape)
        break
