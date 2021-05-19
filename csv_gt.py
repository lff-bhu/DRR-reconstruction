# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 10:54:23 2021

@author: lff
"""

import csv
import os
#import SimpleITK as sitk
import numpy as np
from PIL import Image
ctpath="/media/hdd1/lff/CT/test"
xraypath="/media/hdd1/lff/xray/test"

csvpath="/media/hdd1/lff/lung/4dtest.csv"

f=open(csvpath,'w',encoding='utf-8',newline="")
writer=csv.writer(f)
header=['patient','ct','drr_1','drr_2','mean1','mean2','std1','std2']
#header=['patient','ct','drr_1','drr_2']
writer.writerow(header)

for patient in os.listdir(ctpath):
    ct_path=ctpath+'/'+patient
    drr1_path=xraypath+'/'+patient+'/'+'xray1.png'
    drr2_path=xraypath+'/'+patient+'/'+'xray2.png'
    
    drr1=Image.open(drr1_path)#.resize((128,128))
    drr2=Image.open(drr2_path)#.resize((128,128))
    #ct=sitk.ReadImage(ct_path)
    
    #ct=sitk.GetArrayFromImage(ct)
    drr1=np.array(drr1)/255
    drr2=np.array(drr2)/255
    #ct=ct-np.min(ct)
    #ct=ct/np.max(ct)
    
    #ct_mean=np.mean(ct)
    #ct_std=np.std(ct)
    mean1=np.mean(drr1)
    std1=np.std(drr1)
    mean2=np.mean(drr2)
    std2=np.std(drr2)    
    info=[patient,ct_path,drr1_path,drr2_path,mean1,mean2,std1,std2]
    """
    info=[patient,ct_path,drr1_path,drr2_path]
    """
    writer.writerow(info)
    

f.close()
    
