import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.nn import functional as F
import torchvision.transforms as transforms


config_class2 = [
    ['benign'],
    ['malignant'],
]
config_normal = [
    ['normal'],
]

class BUSI(Dataset):
    def __init__(self, base_path='Dataset/BUSI/', label_config='class2', image_size=255, input_channels=['mask'], **_):
        config = config_class2 if label_config=='class2' else config_normal
        self.image_size = image_size
        self.random_flip = transforms.RandomHorizontalFlip(p=0.5)
        T = transforms.ToTensor()
        self.ccp = transforms.CenterCrop(image_size)
        img_dic = []
        for foder in ['benign', 'malignant', 'normal']:
            files = os.listdir(base_path+foder)
            imgs = [n for n in files if n[-5:]==').png']
            for name in imgs:
                with open(f'{base_path}/{foder}/{name}','rb') as f:
                    img = Image.open(f)
                    img = T(img).mean(dim=0, keepdim=True)
                with open(f'{base_path}/{foder}/{name[:-4]}_mask.png','rb') as f:
                    ori = Image.open(f)
                    ori = T(ori).mean(dim=0, keepdim=True)
                input_image = []
                for channel_name in input_channels:
                    if channel_name == 'image':
                        input_image.append(img)
                    elif channel_name == 'mask':
                        input_image.append(ori)
                    elif channel_name == 'crop':
                        input_image.append(img*ori)
                    else:
                        input_image.append(torch.zeros_like(img))
                input_image = torch.cat(input_image, dim=0)
                img_dic.append((input_image*2-1, foder))
        img_dic = [ (self._img_cut(img), self._get_label_vector(config, label)) for img,label in img_dic]
        self.data = [ (img,label) for img,label in img_dic if label is not None]
    
    def _img_cut(self, img):
        _, H, W = img.shape
        if H > W:
            w = self.image_size
            h = int(w*H/W)+1
        else:
            h = self.image_size
            w = int(h*W/H)+1
        img = F.interpolate(img.unsqueeze(0),[h,w])
        img = img[0]
        img = self.ccp(img)
        return img

    def _get_label_vector(self, config, label_in):
        label_out = None
        for i,ii in enumerate(config):
            if label_in in ii:
                label_out = i
        if label_out is None:
            return None
        return torch.tensor(label_out)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, label = self.data[index]
        img = self.random_flip(img)
        return img, label

