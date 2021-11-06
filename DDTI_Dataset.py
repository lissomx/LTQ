import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# label ids: 
#   0: img id
#   1: folder classification | 'bening', 'maling', 'na'
#   2: composition | 'solid', 'predominantly-solid', 'predominantly-cystic', 'cystic', 'spongiform', ''
#   3: echogenicity | 'marked-hypoechogenicity', 'hypoechogenicity', 'isoechogenicity', 'hyperechogenicity', ''
#   4: margins | 'well-defined', 'microlobulated', 'macrolobulated', 'ill-defined', 'spiculated', ''
#   5: calcifications | 'non', 'microcalcification', 'macrocalcification', '' 
#   6: tirads | '2', '3', '4a', '4b', '4c', '5', ''
#   7: sex | 'F', 'M', ''
#   8: age

config_class2 = [
    {6: '2 3'},
    {6: '4a 4b 4c 5'},
]
config_none = [
    {6: 'None'},
]


class DDTI(Dataset):
    def __init__(self, base_path='Dataset/DDTI/', label_config='class2', input_channels=['image'], centre_crop=False, **_):
        config = config_class2 if label_config=='class2' else config_none
        with open(f'{base_path}/label.txt') as f:
            self.random_flip = transforms.RandomHorizontalFlip(p=0.5)
            labels = [l.strip().split(',') for l in f.readlines()]

            T = transforms.ToTensor()
            img_dic = []
            for img_id in [l[0] for l in labels]:
                with open(f'{base_path}/{img_id}.a.jpg','rb') as f:
                    img = Image.open(f)
                    img = T(img)
                with open(f'{base_path}/{img_id}.b.jpg','rb') as f:
                    ori = Image.open(f)
                    ori = T(ori)
                if centre_crop:
                    img, ori = img[:,:,92:92+315], ori[:,:,92:92+315]
                input_image = []
                for channel_name in input_channels:
                    if channel_name == 'image':
                        input_image.append(img[:1])
                    elif channel_name == 'mask':
                        input_image.append(ori[:1])
                    elif channel_name == 'boundary':
                        input_image.append(ori[1:2])
                    else:
                        input_image.append(torch.zeros_like(img[:1]))
                input_image = torch.cat(input_image, dim=0)
                img_dic.append((img_id, input_image*2-1))
            img_dic = dict(img_dic)

            self.data = [ (img_dic[l[0]], _get_label_vector(config, l)) for l in labels]
            self.data = [ (img,label) for img,label in self.data if label is not None]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, label = self.data[index]
        img = self.random_flip(img)
        return img, label


def _get_label_vector(config, full_label):
    label = None # default
    for i,ii in enumerate(config):
        for col, values in ii.items():
            if full_label[col] in values:
                label = i
    if label is None:
        return None
    return torch.tensor(label)

