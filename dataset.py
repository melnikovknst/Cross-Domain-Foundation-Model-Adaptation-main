import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

DATA_ROOT = os.environ.get(
    'DATA_ROOT',
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
)

class BasicDataset(Dataset):

    def __init__(self,patch_h,patch_w,datasetName,netType,train_mode = False,input_mode='3slice'):

        self.patch_h = patch_h
        self.patch_w = patch_w
        if input_mode not in ('3slice', 'single'):
            raise ValueError("input_mode must be '3slice' or 'single'")
        self.input_mode = input_mode

        if netType == 'unet' or netType == 'deeplabv3plus':
            self.imgTrans = False
            self.input_mode = 'single'
        else: 
            self.imgTrans = True

        self.resize_size = (patch_h * 14, patch_w * 14)

        self.dataset = datasetName

        if datasetName == 'amplitude':
            self.n1 = 1006
            self.n2 = 782
            self.train_data_dir = os.path.join(DATA_ROOT, 'amplitude/train/input')
            self.train_label_dir = os.path.join(DATA_ROOT, 'amplitude/train/target')
            self.valid_data_dir = os.path.join(DATA_ROOT, 'amplitude/valid/input')
            self.valid_label_dir = os.path.join(DATA_ROOT, 'amplitude/valid/target')
        else:
            print("Dataset error!!")
        print('netType:' + netType)
        print('dataset:' + datasetName)
        print('patch_h:' + str(patch_h))
        print('patch_w:' + str(patch_w))
        print('input_mode:' + self.input_mode)

        if train_mode:
            self.data_dir = self.train_data_dir
            self.label_dir = self.train_label_dir
        else:
            self.data_dir = self.valid_data_dir
            self.label_dir = self.valid_label_dir

        self.ids = sorted(
            (
                file_name for file_name in os.listdir(self.data_dir)
                if file_name.endswith('.dat')
            ),
            key=self.slice_sort_key
        )
        if not self.ids:
            raise RuntimeError(f'No .dat files found in {self.data_dir}')
        missing_labels = [
            file_name for file_name in self.ids
            if not os.path.exists(os.path.join(self.label_dir, file_name))
        ]
        if missing_labels:
            raise RuntimeError(
                f'Missing label files in {self.label_dir}: {missing_labels[:5]}'
            )
    def __len__(self):
        return len(self.ids)

    def __getitem__(self,index):
        
        file_name = self.ids[index]
        tPath = os.path.join(self.label_dir, file_name)
        if self.imgTrans:
            if self.input_mode == '3slice':
                slice_indexes = [
                    max(index - 1, 0),
                    index,
                    min(index + 1, len(self.ids) - 1),
                ]
                channels = [
                    np.fromfile(os.path.join(self.data_dir, self.ids[i]),np.float32).reshape(self.n1,self.n2)
                    for i in slice_indexes
                ]
            else:
                center = np.fromfile(os.path.join(self.data_dir, file_name),np.float32).reshape(self.n1,self.n2)
                channels = [center, center, center]
            data = np.stack(channels,axis=0).reshape(1,3,self.n1,self.n2)
        else:
            dPath = os.path.join(self.data_dir, file_name)
            data = np.fromfile(dPath,np.float32).reshape(self.n1,self.n2)
            data = np.reshape(data,(1,1,self.n1,self.n2))
        label = np.fromfile(tPath,np.int8).reshape(self.n1,self.n2)

        data = np.concatenate([data,self.data_aug(data)],axis=0)
        label = np.reshape(label,(1,1,self.n1,self.n2))
        label = np.concatenate([label,self.data_aug(label)],axis=0)

        if self.imgTrans:
            data = self.z_normalize(data)
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            data = torch.from_numpy(data).float()
            data = F.interpolate(data, size=self.resize_size, mode='bilinear', align_corners=False)
            data = data.numpy().astype(np.float32)
        elif not self.imgTrans:
            data = self.z_normalize(data)

        return data,label

    def data_aug(self,data):
        b,c,h,w = data.shape
        data_fliplr = np.flip(data,axis=-1)
        return data_fliplr.reshape(b,c,h,w)

    @staticmethod
    def z_normalize(data, eps=1e-6):
        mean = data.mean(axis=(-2, -1), keepdims=True)
        std = data.std(axis=(-2, -1), keepdims=True)
        return ((data - mean) / np.maximum(std, eps)).astype(np.float32)

    @staticmethod
    def slice_sort_key(file_name):
        name = os.path.splitext(file_name)[0]
        return (0, int(name)) if name.isdigit() else (1, name)

if __name__ == '__main__':

    train_set = BasicDataset(72,56,'amplitude','setr1',True,True)
    print(train_set.__getitem__(0)[1].shape)
    print(len(train_set))
