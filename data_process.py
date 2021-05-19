# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 17:39:11 2021

@author: lff
"""


import SimpleITK as sitk
import os
import time
import ctpro
import csv




def load_nii_scan(path,patient):
    
    
    
    patient_image = sitk.ReadImage(path+'/'+patient)
    
    patient_data=sitk.GetArrayFromImage(patient_image)
    
    size=patient_image.GetSize()
    origin=patient_image.GetOrigin()
    spacing=patient_image.GetSpacing()
    
    return patient_image,patient_data,origin,size,spacing


def save_nii_scan(volume,origin,spacing,path):
    itkimage=sitk.GetImageFromArray(volume)
    itkimage.SetOrigin(origin)
    itkimage.SetSpacing(spacing)
    sitk.WriteImage(itkimage, path)

    


if __name__=='__main__':
    
    path="/media/hdd1/lff/test/nii"
    patients=os.listdir(path)
    train_save_path="/media/hdd1/lff/test/320/nii"
    test_save_path="/media/hdd1/lff/test_nii"
    
    
    start=time.time()
    train=open("/media/hdd1/lff/train_ct.csv",'w',encoding='utf-8',newline="")
    test=open("/media/hdd1/lff/test_ct.csv",'w',encoding='utf-8',newline="")
    
    train=csv.writer(train)
    test=csv.writer(test)
    
    header=["patient_ID","ct_path","old_size","new_size"]
    train.writerow(header)
    test.writerow(header)
    
    for file_index ,file_name in enumerate(patients):
        t0=time.time()
        file_path=path+"\\"+file_name
        if file_index<800:
            save_file_path=train_save_path+"/"+file_name
            save_path=train_save_path
        else:
            save_file_path=test_save_path+"/"+file_name
            save_path=test_save_path
            
        print("Begin {}/{}:{}".format(file_index+1, path,file_name))
        
            
        
        image,image_data,image_origin,image_size,image_spacing=load_nii_scan(path,patients[file_index])
        print("old:",image_size)
        image_data2,image2_spacing=ctpro.resample(image_data,image_spacing,[1,1,1])
        print("std:",image_data2.shape[: : -1])
        std_scan=ctpro.crop_to_standard(image_data2,scale=320)
        save_nii_scan(std_scan,(0,0,0),image2_spacing,save_file_path)
        if file_index<800:
            inf=load_nii_scan(save_path,os.listdir(save_path)[file_index])
        else:
            inf=load_nii_scan(save_path,os.listdir(save_path)[file_index-800])
        new_size=inf[3]
        print("new:",new_size)
        info=[file_name,save_file_path,image_size,new_size]
        if file_index<800:
            train.writerow(info)
        else:
            test.writerow(info)
            
        t1=time.time()
        print("End! case time{}".format(t1-t0))
        
    end=time.time()
    print("Finally! Total time{}".format(end-start))

        
        

    
    
    
    