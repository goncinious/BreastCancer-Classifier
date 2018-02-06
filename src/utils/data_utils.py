#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 21:22:30 2017

@author: goncalofigueira
"""

import numpy as np
import cv2

def getFileList(root,file_type):
    import os
    l =[]
    for path, dirs, files in os.walk(root):
         [l.append(os.path.join(path, f)) for f in files  if f.endswith(file_type) and f[0] != '.']
    return l


def sortTarget(im_folder, im_info):
    import numpy as np
    
    targets = np.empty(shape=im_folder.shape)
    idx = 0
    for filename in im_folder:
        targets[idx] = np.where(im_info.filename == filename.split('/')[-1])[0].astype(int)
        idx+=1
      
    return im_info.loc[targets,:] 
    
        
def ReadImage(filename):
    import cv2
    return cv2.imread(filename)

def rgb_normalized(rgb):

        rgb = rgb.astype('float64')
        norm=np.zeros((rgb.shape[0],rgb.shape[1],3),np.float32)
        norm_rgb=np.zeros((rgb.shape[0],rgb.shape[1],3),np.uint8)

        b=rgb[:,:,0]
        g=rgb[:,:,1]
        r=rgb[:,:,2]

        rgb_sum=b+g+r

        norm[:,:,0]=b/rgb_sum*255.0
        norm[:,:,1]=g/rgb_sum*255.0
        norm[:,:,2]=r/rgb_sum*255.0

        norm_rgb=cv2.convertScaleAbs(norm)
        return norm_rgb

