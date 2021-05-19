# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 11:13:42 2021

@author: lff
"""
import os
import csv
#import argparse
import torch.backends.cudnn as cudnn
import time
import torch
import torch.nn as nn
#import numpy as np
from mydataset import mydataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from net import ReconNet
from utils import AverageMeter
from tqdm import tqdm
from torchsummary import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau
#from torchvision.transforms import transforms

"""
parser=argparse.ArgumentParser(description="pytorch 128x128x128 style training")
parser.add_argument("--exp",type=int,default=1)
"""
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

cudnn.benchmark = True
device_ids = [1]
device = torch.device('cuda:1')


#train_stat=np.load("/media/hdd1/lff/first_test/train.npz")
#normalize=transforms.Normalize(mean=list(train_stat['mean']),std=list(train_stat['std']))
#normalize=transforms.Normalize(0.5,0.5)
#train_transform=transforms.Compose([normalize])
trainfile_list="/media/hdd1/lff/lung/4dtrain.csv"
traindataset=mydataset(trainfile_list,num_view=2,input_size=128)
train_loader=DataLoader(traindataset,batch_size=1, num_workers=0, pin_memory=True)

#test_stat=np.load("/media/hdd1/lff/first_test/test.npz")
#normalize=transforms.Normalize(mean=list(test_stat['mean']),std=list(test_stat['std']))
#normalize=transforms.Normalize(0.5,0.5)
#test_transform=transforms.Compose([normalize])
test_file_list="/media/hdd1/lff/lung/4dtest.csv"
test_dataset=mydataset(test_file_list,num_view=2,input_size=128)
test_loader=DataLoader(test_dataset,batch_size=1, num_workers=0, pin_memory=True)



model=ReconNet(2,128)
model.cuda()


criterion = nn.MSELoss(reduction='mean').cuda()
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0)

optimizer = torch.optim.Adam(model.parameters(),lr=0.0001,betas=(0.5, 0.999))
#scheduler =torch.optim.lr_scheduler.StepLR(optimizer, step_size=600, gamma=0.1, last_epoch=-1)
#scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=10, verbose=True)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,  milestones = [300,600,900], gamma = 0.5, last_epoch=-1)
#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)

print('---------- Networks initialized -------------')
summary(model, (2, 128, 128))
print('-----------------------------------------------')


start=time.time()
with open("/media/hdd1/lff/lung/4d/loss.csv",'w',encoding='utf-8',newline="") as logg:
    
    log1=csv.writer(logg)
    header=['Epoch','train loss','test loss']
    log1.writerow(header)
    with open("/media/hdd1/lff/lung/4d/trainlog.csv",'w',encoding='utf-8',newline="") as log:
        log=csv.writer(log)
        header=['Epoch','trainloader','train loss','train average loss']
        log.writerow(header)
        
        
        for epoch in tqdm(range(1000)):
            t0 =time.time()
            model.train()
            
            train_loss = AverageMeter()
            #print(model.optimizer.state_dict()['param_groups'][0]['lr'])
            for i ,(xray,ct) in enumerate(train_loader):
                
                drr , ct = Variable(xray) , Variable(ct)
                drr = drr.cuda()
                ct  = ct.cuda()
                
                target=model(drr)
                #print(target.size(),ct.size())
                
                loss  =criterion(target, ct)
                train_loss.update(loss.data.item(),xray.size(0))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #scheduler.step()
                #scheduler.step(train_loss.avg)
                
                print('Epoch:[{0}]\t'
                          'Iter:[{1}/{2}]\t'
                          'Train loss: {loss.val:.5f} {loss.avg:.5f}\t'.format(
                              epoch,i,len(train_loader),loss=train_loss
                              ))
                info=[epoch,i+1,train_loss.val,train_loss.avg,]
                log.writerow(info)
                
            #scheduler.step(train_loss.avg)        
            t1 = time.time()
            
            print('End! Eopch time: {}'.format(t1-t0))
            print("Finished Epoch:[{0}\t]"
                  "Average Train Loss:{loss.avg:.5f}\t".format(
                      epoch, loss=train_loss))
            
            model.eval()
            test_loss=AverageMeter()
            for i , (xray,ct) in enumerate(test_loader):
                drr , ct =Variable(xray) , Variable(ct)
                drr=drr.cuda()
                ct=ct.cuda()
            
                target=model(drr)
                loss1=criterion(target, ct)
            
            
                test_loss.update(loss1.data.item(),xray.size(0))
            
                print('{}:{}/{}\t'
                  'test loss:{loss.val:.5f},{loss.avg:.5f}'.format(
                      epoch, i,len(test_loader),loss=test_loss))
            
            info=[epoch,train_loss.avg,test_loss.avg]
            #scheduler.step(test_loss.avg)
            
            #print(lr)
            
            log1.writerow(info)
            
            
            if epoch%100==99:
                print("Save epoch %d model ..."%(epoch+1))
                savename='train'+str(epoch)+'.pkl'
                savepath='/media/hdd1/lff/lung/4d'+'/'+savename
                torch.save(model,savepath)
                
            
            
               
        end=time.time()
        print('Finally! Total time: {}'.format(end-start))
        
