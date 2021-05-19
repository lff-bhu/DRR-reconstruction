# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 18:34:17 2021

@author: lff
"""

import os
import numpy as np
from subprocess import check_output as qx
import matplotlib.pyplot as plt
import cv2
import time
from pfm_read import pfm_read
from data_process import load_nii_scan , save_nii_scan

#compute x-ray source center in world coordinate
def get_center(origin,size,spacing):
    origin = np.array(origin)
    size   = np.array(size)
    sacing = np.array(spacing)
    center= origin + (size-1) / 2 * spacing
    return center

#convert a ndarray to string
def array2string(ndarray):
     ret=""
     for i in ndarray:
          
         ret = ret +str(i) + " "
         
     return ret[:-2]
 
#save a pfm file as png file
def save_png(filename,direction):
    raw_data , scale = pfm_read(filename)
    max_value = raw_data.max()
    im = (raw_data /max_value*255).astype(np.uint8)
    # PA view should do additional left-right flip
    if direction == 1:
        im = np.fliplr(im)
    
    savedir, _ = os.path.split(filename)
    outfile = os.path.join(savedir, "xray{}.png".format(direction))
    # plt.imshow(im, cmap=plt.cm.gray)
    plt.imsave(outfile, im, cmap=plt.cm.gray)
    # plt.imsave saves an image with 32bit per pixel, but we only need one channel
    image = cv2.imread(outfile)
    gray = cv2.split(image)[0]
    cv2.imwrite(outfile, gray)
    

if __name__=="__main__":
    root_path="/media/hdd1/lff/lungnii/test"
    savepath="/media/hdd1/lff/lungxray/test"
    #plasti_path="D:/plastimatch/bin"
    
    patients=os.listdir(root_path)
    start=time.time()
    for file_index, file_name in enumerate(patients):
        
        t0 =time.time()
        
        save_path=savepath+'/'+file_name
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print("Begin! {}/{}:{}".format(file_index+1,len(patients),file_name))
        image_path=os.path.join(root_path,file_name)
        
        ct_itk,ct_scan,ori_origin,ori_size,ori_spacing=load_nii_scan(root_path,patients[file_index])
        
        center=get_center(ori_origin,ori_size,ori_spacing)
        # map the .mha file value to (-1000, 1000)
        cmd_str =  'plastimatch adjust --input {} --output {} --pw-linear "0, -1000"'.format(image_path, save_path+'/out.mha')  
        #print(cmd_str)
        output = os.system(cmd_str)
        # get virtual xray file
        directions = [1, 2]
        for i in directions:
            if i == 1:
                nrm = "0 1 0"
            else:
                nrm = "1 0 0"    
            '''
            plastimatch usage
            -t : save format
            -g : sid sad [DistanceSourceToPatient]:541 
                         [DistanceSourceToDetector]:949.075012
            -r : output image resolution
            -o : isocenter position
            -z : physical size of imager
            -I : input file in .mha format
            -O : output prefix
            '''
            #cmd_str = plasti_path + '/drr -t pfm -nrm "{}" -g "541 949" \
                    #-r "320 320" -o "{}" -z "500 500" -I {} -O {}'.format\
                    #(nrm, array2string(center), save_path+'/out.mha', save_path+'/{}'.format(i))
            cmd_str = 'plastimatch drr -t pfm --nrm "{}" --sad 541 --sid 949 -r "320 320" -o "{}" -z "600 600" -I {} -O {}'.format(nrm, array2string(center), save_path+'/out.mha', save_path+'/{}'.format(i))
            #print(cmd_str)   
            output = os.system(cmd_str)
            # plastimatch would add a 0000 suffix 
            pfmFile = save_path+'/{}'.format(i) + '0000.pfm'
            save_png(pfmFile, i)
        os.remove(save_path+'/out.mha')
        t1 = time.time()
        print('End! Case time: {}'.format(t1-t0))
    end = time.time()
    print('Finally! Total time: {}'.format(end-start))