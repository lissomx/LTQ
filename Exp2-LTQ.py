import time
import random
from collections import Counter
import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import Subset,ConcatDataset
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import random_split, DataLoader
from einops import rearrange

import Reporter
from DDTI_Dataset import DDTI
from BUSI_Dataset import BUSI
from Vqvae1lv2 import VQVAE
from EmbdClassifier import EmbdClassifier
from IndexImage import numbers2images

class Model(nn.Module):

    def __init__(self, **args):
        super().__init__()
        self.main = VQVAE(**args)
    
    def q_encode(self, imgs):
        imgs = F.interpolate(imgs,[256,256])
        e_imgs, diff, q_imgs = self.main.encode(imgs)
        return diff, q_imgs, e_imgs

    def embed(self):
        return self.main.quantize_b.embed.T
    
    def forward(self, imgs):
        imgs = F.interpolate(imgs,[256,256])
        product, diff, ids = self.main(imgs)
        self.loss = diff.mean() * 0.25
        product = F.interpolate(product,[256,256])
        return product, imgs, ids, diff


n_track_last = 20
n_cross = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_vqvae(ep, loader, model, optimizer, scheduler):
    model.train()
    for j, (imgs, labels) in enumerate(loader):

        imgs = imgs.to(device)

        # Reporter.runNtime(lambda : vutils.save_image(imgs.cpu(), f'Temp/tst.jpg', nrow=4, padding=4), 1, 'img example')
        
        optimizer.zero_grad()

        prod, imgs, *_ = model(imgs)
        loss_rec = (prod-imgs).pow(2).mean()
        loss_vq = model.loss.mean()
        
        loss = loss_rec + loss_vq
        Reporter.record(loss_rec.item(), 'train-rec')
        Reporter.record(loss_vq.item(), 'train-vq')
        loss.backward()
        optimizer.step()

    # scheduler.step()

def test_vqvae(ep, loader, model):
    model.eval()
    IMGS = []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        prod, imgs, ids, diff = model(imgs)
        loss_rec = (prod-imgs).pow(2).mean()
        loss_vq = model.loss.mean()
        Reporter.record(loss_rec.item(), 'test-rec')
        Reporter.record(loss_vq.item(), 'test-vq')
        imgs = imgs.cpu()
        prod = prod.cpu()
        imgs = torch.cat([imgs[:,:1], prod[:,:1], numbers2images(ids.cpu(), 256)], dim=3)
        IMGS.append(imgs)
    return torch.cat(IMGS, dim=0)

def train_classi(ep, loader, model, classifier, optimizer, scheduler, class_weight, do_record):
    classifier.train()
    for j, (imgs, labels) in enumerate(loader):

        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            diff, q_imgs, e_imgs = model.q_encode(imgs)

        optimizer.zero_grad()

        prediction = classifier(q_imgs)
        loss = F.cross_entropy(prediction,labels,class_weight)
        # loss = F.cross_entropy(prediction,labels)

        if do_record: 
            Reporter.srecord(prediction.argmax(dim=1), labels, group='train')
        
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        Reporter.record(loss.item(), 'train-loss')

    # scheduler.step()

def test_classi(ep, loader, model, classifier, class_weight, do_record):
    classifier.eval()
    for j, (imgs, labels) in enumerate(loader):

        imgs, labels = imgs.to(device), labels.to(device)

        diff, q_imgs, e_imgs = model.q_encode(imgs)

        prediction = classifier(q_imgs)
        loss = F.cross_entropy(prediction,labels,class_weight)
        # loss = F.cross_entropy(prediction,labels)

        Reporter.record(loss.item(), 'test-loss')
        if do_record: 
            Reporter.srecord(prediction.argmax(dim=1), labels, group='test')
        

def loop(tag, n_epoch1, n_epoch2, data, Yu, use_pretrained_ae=False, **args):
    print('\n======= start new task =========')
    print(tag, n_epoch1, n_epoch2, data, Yu, use_pretrained_ae, args)
    tag += '-'+data
    tag += '-Yu' if Yu else '-noYu'
    for x in range(n_cross):
        print('\n========== n_cross:',x,'=========')

        dataset = eval(data)(**args)
        indexs_ = list(range(len(dataset)))
        random.seed(x)
        random.shuffle(indexs_)
        _20 = len(indexs_)//5
        train_data = Subset(dataset, indexs_[_20:])
        test_data = Subset(dataset, indexs_[:_20])

        train_loader = DataLoader(train_data, 128, shuffle=True)
        test_loader = DataLoader(test_data, 16)

        train_label_count_ =[l.item() for _,l in train_data]
        train_label_count = Counter(train_label_count_)
        pos_weight = 2 * train_label_count[0] / (train_label_count[0]+train_label_count[1])
        neg_weight = 2 * train_label_count[1] / (train_label_count[0]+train_label_count[1])
        class_weight = torch.tensor([pos_weight, neg_weight]).to(device)

        if not use_pretrained_ae:
            dataset_e = eval(data)(label_config='unlabeled', **args)
            if (Yu):
                train_data_e = ConcatDataset([train_data, dataset_e])
            else:
                train_data_e = train_data
            train_loader_e = DataLoader(train_data_e, 16, shuffle=True)
            
            # ====== Step 1: train the autoencoder for the index grids ======
            model = Model(**args).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=3e-4)

            for ep in range(n_epoch1+1):
                with torch.no_grad():
                    IMGS = test_vqvae(ep, test_loader, model)
                train_vqvae(ep, train_loader_e, model, optimizer, None)
                if ep%10==0:
                    print(f'  ep = {ep},\t train:\t{Reporter.report("train-rec"):.4f}\t{Reporter.report("train-vq"):.4f}\t test:\t{Reporter.report("test-rec"):.4f}\t{Reporter.report("test-vq"):.4f}')
                if ep%50 == 0:
                    vutils.save_image(IMGS/2+0.5, f'Temp/{tag}.jpg', nrow=4, padding=4)
                    torch.save(model.state_dict(), f"Checkpoint/{tag}.pt")
        else:
            tag = use_pretrained_ae
        
        optimizer = None
        model = Model(**args).to(device)
        model.load_state_dict(torch.load(f"Checkpoint/{tag}.pt"))
        model.eval()

        # ====== Step 2: train the classifier ======        
        optimizer = None
        classifier = EmbdClassifier(pre_embed=model.embed(), **args).to(device)
        optimizer = optim.AdamW(classifier.parameters(), lr=1e-3)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, verbose=False)

        print('\n========== Tag:',tag,'=========')
        for ep in range(n_epoch2+1):
            do_record = ep >= n_epoch2-n_track_last
            with torch.no_grad():
                test_classi(ep, test_loader, model, classifier, class_weight, do_record)
            train_classi(ep, train_loader, model, classifier, optimizer, None, class_weight, do_record)
            if ep%10==0:
                print(f"ep = {ep}, train loss = {Reporter.report('train-loss')}, test loss = {Reporter.report('test-loss')}")
        results_train = Reporter.sreport(group='train')
        results_test = Reporter.sreport(group='test')
        print(results_train)
        print(results_test)
        with open(f"Temp/{tag}.result.txt", 'a') as f:
            f.write(results_test+'\n\n')




ticks = int(time.time()) 
print('##################',"Start!", ticks,'##################')

'''
======== The experiment for DDTI ===========
NB: the original images in DDTI are 500x315 but usually with wide black margins. 
    So, the images shown in the paper are based on the centre cropped images. 
    To enable the centre crop, pass "'centre_crop' : True" to the setting dict.
'''
loop(f'Exp2-PAPER-c-{ticks+0}', **{
    'use_pretrained_ae' : None, 
    'input_channels' : ['image'], # image only
    'n_epoch1': 4000, 
    'n_epoch2' : 200, 
    'Yu' : True, 
    'data' : 'DDTI',
    'n_embed' : 64,
    'ae_plan' : [1, 32, 64, 128, 2048],
    'cl_plan' : [64, 64, 64],
    'centre_crop' : False,
})  
#               precision    recall  f1-score   support

#            0     0.9040    0.9286    0.9161       294
#            1     0.9874    0.9827    0.9851      1680

#     accuracy                         0.9747      1974
#    macro avg     0.9457    0.9557    0.9506      1974
# weighted avg     0.9750    0.9747    0.9748      1974

# [[ 273   21]
#  [  29 1651]]

'''
======== The experiment for BUSI ===========
This experiment uses both ultrasound images and nodule masks.
There is 2-channel input -- the crop nodule and the mask.
The crop nodules are the masked ultrasound images i.e. replacing the pixels out of the nodule by 0.
(so, NB, the 'ae_plan' starts with 2 -- two input channels; 'input_channels' contains two items.)
'''
loop(f'Exp2-PAPER-c-{ticks+1}', **{
    'use_pretrained_ae' : None, 
    'input_channels' : ['crop', 'mask'],
    'n_epoch1': 3000, 
    'n_epoch2' : 500, 
    'Yu' : True, 
    'data' : 'BUSI',
    'n_embed' : 64,
    'ae_plan' : [2, 32, 64, 1024],
    'cl_plan' : [64, 64, 64, 64],
}) 
#               precision    recall  f1-score   support

#            0     0.9476    0.9851    0.9660      1743
#            1     0.9710    0.9017    0.9351       966

#     accuracy                         0.9553      2709
#    macro avg     0.9593    0.9434    0.9505      2709
# weighted avg     0.9559    0.9553    0.9549      2709

# [[1717   26]
#  [  95  871]]

# nohup python3 -u Exp2-LTQ.py >> Exp2-LTQ.log 2>&1 &