# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 15:16:07 2021

@author: lff
"""

import SimpleITK as sitk
import os


def dcm2nii(dcm_path,nii_path):
    
    #1 dcm序列文件整合
    reader= sitk.ImageSeriesReader()
    
    dcm_names= reader.GetGDCMSeriesFileNames(dcm_path)
    
    reader.SetFileNames(dcm_names)
    
    image2=reader.Execute()
    
    #2. 将整合的信息转换成array，并读取dcm文件的基本信息
    image_array= sitk.GetArrayFromImage(image2)#z y z
    
    origin=image2.GetOrigin() #x y z原点
    print(origin)
    
    spacing=image2.GetSpacing()#x y z 间距
    print(spacing)
    
    direction=image2.GetDirection()#x y z 方向
    print(direction)
    
    print(image2.GetSize())
    #3. 将array转换成image，并保存为nii.gz
    image3=sitk.GetImageFromArray(image_array)
    
    image3.SetSpacing(spacing)
    
    image3.SetOrigin(origin)
    
    image3.SetDirection(direction)
    
    sitk.WriteImage(image3 , nii_path)
    


if __name__ == '__main__':
    
    path="E:/manifest-1620960554915/4D-Lung/100_HM10395/07-02-2003-p4-14571"
    save_path="E:/manifest-1620960554915/4D-Lung/nii"
    for file_index,filename in enumerate(os.listdir(path)):
        dcm_path=os.path.join(path,filename)
        nii_path=save_path+"/"+filename+'.nii'
        dcm2nii(dcm_path,nii_path)
        print("finshde {}/{}".format(file_index+1,len(os.listdir(path))))
    