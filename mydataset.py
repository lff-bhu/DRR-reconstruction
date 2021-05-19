# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 14:31:39 2021

@author: lff
"""

import os
from PIL import Image
import numpy as np
import SimpleITK as sitk
import  torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import transforms

class mydataset(Dataset):
    
    def __init__(self,file_list,num_view,input_size,output_size=None):
        
        #file_list: .csv 包含样例数据路径
        #格式：patient,ct_path,xray1.pmg,xray2.png,exist
        #exist:判断文件路径是否完整
        self.df=pd.read_csv(file_list)
        self.num_view=num_view
        self.input_size=input_size
        self.output_size=output_size
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        projs=np.zeros((self.num_view,self.input_size,self.input_size))
        
        #load 2D projections
        
        for view_idx in range(self.num_view):
            if view_idx==0:
                project_path=self.df.iloc[idx]['drr_1']
                #mean1=self.df.iloc[idx]['mean1']
                #std1=self.df.iloc[idx]['std1']
            else:
                project_path=self.df.iloc[idx]['drr_2']
                #mean2=self.df.iloc[idx]['mean2']
                #std2=self.df.iloc[idx]['std2']
            
            
        #resize 2D image
            proj=Image.open(project_path).resize((self.input_size,self.input_size))
            
            projs[view_idx,:,:]=np.array(proj)/255
        """
        project_path=self.df.iloc[idx]['drr_1']
        proj=Image.open(project_path).resize((self.input_size,self.input_size))
        proj=np.array(proj)
        proj=torch.FloatTensor(proj).unsqueeze(0)
        """
        """
        if self.num_view==1:
            mean=mean1
            std=std1
        else:
            mean=[mean1,mean2]
            std=[std1,std2]
        normalize=transforms.Normalize(mean=mean, std=std)
        transform=transforms.Compose([normalize])
        projs=torch.FloatTensor(projs)
        projs=transform(projs)
        """
        projs=torch.FloatTensor(projs)
        
        #load CT image
        ct_path=self.df.iloc[idx]['ct']
        image=sitk.ReadImage(ct_path)
        image=sitk.GetArrayFromImage(image)
        #image = np.fromfile(ct_path, dtype=np.float32)
        #print(image.shape)
       
        #normalize CT image
        
        image=image-np.min(image)
        image=image/np.max(image)
        
        
        image = torch.FloatTensor(image)
        
        return (projs,image)
            
        
            
        
        