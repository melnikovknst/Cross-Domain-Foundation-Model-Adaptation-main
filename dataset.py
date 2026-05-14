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
    coherence_sample_window = 9
    coherence_trace_window = 9

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
            img_tensor = np.zeros([2,3,self.patch_h*14,self.patch_w*14],np.float32)
            for i in range(data.shape[0]):
                img = self.build_dinov2_attribute_image(data[i,0])
                img_tensor[i] = self.transform(img)
            data = img_tensor
        elif not self.imgTrans:
            data = data/255

        return data,label

    def data_aug(self,data):
        b,c,h,w = data.shape
        data_fliplr = np.fliplr(np.squeeze(data))
        return data_fliplr.reshape(b,c,h,w)

    def build_dinov2_attribute_image(self,data):
        data = data.astype(np.float32,copy=False)
        centered = data - np.median(data)
        regularized = self.binomial_blur2d(centered)

        amplitude = self.robust_unit_scale(data)
        coherence = self.semblance_coherence(regularized)
        coherence = self.robust_unit_scale(coherence)
        gradient = self.gradient_magnitude(regularized)
        gradient = self.robust_unit_scale(gradient,upper=99.5)

        attributes = np.stack([amplitude,coherence,gradient],axis=-1)
        attributes = np.clip(attributes*255.0,0.0,255.0).astype(np.uint8)
        return Image.fromarray(attributes)

    def semblance_coherence(self,data):
        trace_window = self.coherence_trace_window
        sample_window = self.coherence_sample_window

        trace_sum = self.sum_filter_axis(data,trace_window,axis=1)
        numerator = self.sum_filter_axis(trace_sum*trace_sum,sample_window,axis=0)
        energy = self.box_sum2d(data*data,sample_window,trace_window)
        denominator = trace_window*energy

        numerator = self.binomial_blur2d(numerator)
        denominator = self.binomial_blur2d(denominator)
        coherence = numerator/(denominator + 1e-6)
        return np.clip(coherence,0.0,1.0).astype(np.float32)

    def gradient_magnitude(self,data):
        gy,gx = np.gradient(data)
        return np.sqrt(gx*gx + gy*gy).astype(np.float32)

    def robust_unit_scale(self,data,lower=1.0,upper=99.0):
        low,high = np.percentile(data,[lower,upper])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            return np.zeros_like(data,dtype=np.float32)
        data = (data - low)/(high - low)
        return np.clip(data,0.0,1.0).astype(np.float32)

    def box_sum2d(self,data,height,width):
        return self.sum_filter_axis(
            self.sum_filter_axis(data,width,axis=1),
            height,
            axis=0
        )

    def sum_filter_axis(self,data,window,axis):
        if window <= 1:
            return data.astype(np.float32,copy=True)

        before = window//2
        after = window - 1 - before
        pad_width = [(0,0)]*data.ndim
        pad_width[axis] = (before,after)
        padded = np.pad(data,pad_width,mode='reflect')
        cumsum = np.cumsum(padded,axis=axis,dtype=np.float64)

        zero_shape = list(cumsum.shape)
        zero_shape[axis] = 1
        cumsum = np.concatenate(
            [np.zeros(zero_shape,dtype=cumsum.dtype),cumsum],
            axis=axis
        )
        upper = np.take(cumsum,np.arange(window,window + data.shape[axis]),axis=axis)
        lower = np.take(cumsum,np.arange(data.shape[axis]),axis=axis)
        return (upper - lower).astype(np.float32)

    def binomial_blur2d(self,data):
        kernel = np.array([1.0,4.0,6.0,4.0,1.0],dtype=np.float32)/16.0
        data = self.convolve1d_reflect(data,kernel,axis=1)
        data = self.convolve1d_reflect(data,kernel,axis=0)
        return data.astype(np.float32,copy=False)

    def convolve1d_reflect(self,data,kernel,axis):
        radius = len(kernel)//2
        pad_width = [(0,0)]*data.ndim
        pad_width[axis] = (radius,radius)
        padded = np.pad(data,pad_width,mode='reflect')
        output = np.zeros_like(data,dtype=np.float32)

        for offset,weight in enumerate(kernel):
            slices = [slice(None)]*data.ndim
            slices[axis] = slice(offset,offset + data.shape[axis])
            output += weight*padded[tuple(slices)]
        return output

if __name__ == '__main__':

    train_set = BasicDataset(72,56,'amplitude','setr1',True)
    print(train_set.__getitem__(0)[0].shape)
    print(train_set.__getitem__(0)[1].shape)
    print(len(train_set))
