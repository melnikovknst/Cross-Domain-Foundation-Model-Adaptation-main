import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

DATA_ROOT = os.environ.get(
    'DATA_ROOT',
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
)

class BasicDataset(Dataset):

    def __init__(self,patch_h,patch_w,datasetName,netType,train_mode = False):

        self.patch_h = patch_h
        self.patch_w = patch_w

        if netType == 'unet' or netType == 'deeplabv3plus':
            self.imgTrans = False
        else: 
            self.imgTrans = True

        self.transform = T.Compose([
            T.Resize((patch_h * 14, patch_w * 14)),
            T.ToTensor(),
        ])    

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

        if train_mode:
            self.data_dir = self.train_data_dir
            self.label_dir = self.train_label_dir
        else:
            self.data_dir = self.valid_data_dir
            self.label_dir = self.valid_label_dir

        self.ids = sorted(
            file_name for file_name in os.listdir(self.data_dir)
            if file_name.endswith('.dat')
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
        dPath = os.path.join(self.data_dir, file_name)
        tPath = os.path.join(self.label_dir, file_name)
        data = np.fromfile(dPath,np.float32).reshape(self.n1,self.n2)
        label = np.fromfile(tPath,np.int8).reshape(self.n1,self.n2)

        data = np.reshape(data,(1,1,self.n1,self.n2))
        data = np.concatenate([data,self.data_aug(data)],axis=0)
        label = np.reshape(label,(1,1,self.n1,self.n2))
        label = np.concatenate([label,self.data_aug(label)],axis=0)

        if self.imgTrans:
            img_tensor = np.zeros([2,1,self.patch_h*14,self.patch_w*14],np.float32)
            for i in range(data.shape[0]):
                img = Image.fromarray(np.uint8(data[i,0]))
                img_tensor[i,0] = self.transform(img)
            data = img_tensor
            data = data.repeat(3,axis=1)
        elif not self.imgTrans:
            data = data/255

        return data,label

    def data_aug(self,data):
        b,c,h,w = data.shape
        data_fliplr = np.fliplr(np.squeeze(data))
        return data_fliplr.reshape(b,c,h,w)

if __name__ == '__main__':

    train_set = BasicDataset(72,56,'amplitude','setr1',True,True)
    print(train_set.__getitem__(0)[1].shape)
    print(len(train_set))
