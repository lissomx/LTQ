import random
from collections import Counter
import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import Subset
import torchvision
import torchvision.utils as vutils
from torch.utils.data import DataLoader

import Reporter
from DDTI_Dataset import DDTI
from BUSI_Dataset import BUSI


class Model(nn.Module):
    '''
        To wrap and load baseline models
    '''
    def __init__(self, model_name):
        super().__init__()
        _model = eval('torchvision.models.'+model_name)
        self.main = _model(num_classes=2)
    
    def forward(self, imgs):
        imgs = F.interpolate(imgs,[224,224])
        imgs1 = imgs[:,:1]
        zeros = torch.zeros_like(imgs1)
        imgs2 = imgs[:,1:2] if imgs.shape[1] > 1 else zeros
        imgs3 = imgs[:,2:3] if imgs.shape[1] > 2 else zeros
        imgs = torch.cat([imgs1, imgs2, imgs3], dim=1)
        product = self.main(imgs)
        return product



n_cross = 10
n_epoch = 200
n_track_last = 50

batch_size = 64
lr=1e-3


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(ep, loader, classifier, optimizer, scheduler):
    classifier.train()
    for j, (imgs, labels) in enumerate(loader):

        imgs, labels = imgs.to(device), labels.to(device)

        # Reporter.runNtime(lambda : vutils.save_image(imgs.cpu()/2+0.5, f'Temp/tst.jpg', nrow=4, padding=4), 1, 'img example')

        optimizer.zero_grad()

        prediction = classifier(imgs)

        loss = F.cross_entropy(prediction,labels)

        if ep > n_epoch - n_track_last: 
            Reporter.srecord(prediction.argmax(dim=1), labels, group='train')
        
        loss = loss.mean()
        loss.backward()
        # nn.utils.clip_grad_norm_(classifier.parameters(), 0.01)
        optimizer.step()
        Reporter.record(loss.item(), 'train-loss')

    # scheduler.step()

def test(ep, loader, classifier):
    classifier.eval()
    for j, (imgs, labels) in enumerate(loader):

        imgs, labels = imgs.to(device), labels.to(device)

        prediction = classifier(imgs)

        loss = F.cross_entropy(prediction,labels)

        Reporter.record(loss.item(), 'test-loss')

        if ep > n_epoch - n_track_last: 
            Reporter.srecord(prediction.argmax(dim=1), labels, group='test')
        

def loop(tag, n_epoch, dataset_name, model_name, **args):
    tag = f"{tag}-{dataset_name}-{model_name}"

    # load dataset
    dataset = eval(dataset_name)(**args)
    indexs_ = list(range(len(dataset)))
    random.seed(0)
    random.shuffle(indexs_)
    _20 = len(indexs_)//5
    train_data = Subset(dataset, indexs_[_20:])
    test_data = Subset(dataset, indexs_[:_20])

    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size)

    # init the model
    classifier = Model(model_name).to(device)
    optimizer = optim.AdamW(classifier.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, verbose=False)

    # training starts
    print('\n========== Tag:',tag,'=========')
    for ep in range(n_epoch+1):
        with torch.no_grad():
            test(ep, test_loader, classifier)
        train(ep, train_loader, classifier, optimizer, scheduler)
        if ep%10==0: 
            # report processing cotcomes
            print(f"ep = {ep}, train loss = {Reporter.report('train-loss')}, test loss = {Reporter.report('test-loss')}")

    # report final outcomes
    print(Reporter.sreport(group='train'))
    print(Reporter.sreport(group='test'))


'''
    'dataset_name' can be the string of 'DDTI' or 'BUSI'
    'model_name' can be any model under the namespace of 'torchvision.models' e.g. 'resnet50', 'densenet121', etc.
'''

loop('Exp1-Baseline', n_epoch, dataset_name='DDTI', model_name='resnet50') #  
# loop('Exp1-Baseline', n_epoch, dataset_name='BUSI', model_name='densenet121') #  
# loop('Exp1-Baseline', n_epoch, dataset_name='BUSI', model_name='vgg16') #  


# nohup python3 -u Exp9-vqvae2.py >> Exp9-vqvae2.log 2>&1 &
