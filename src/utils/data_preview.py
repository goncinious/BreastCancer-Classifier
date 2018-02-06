#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 23:53:27 2017

@author: goncalofigueira
"""
# important: assumes that target is dataframe of strings


def DisplaySamples(im_folder, target, class_perc = 0.05):
    from utils.data_utils import ReadImage
    from sklearn.model_selection import StratifiedShuffleSplit
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    plt.close('all')
    import numpy as np
    
    split = StratifiedShuffleSplit(n_splits = 1, test_size = class_perc)
    for g1,g2 in split.split(im_folder,target):
        target = target[g2]
    
    target = target.sort_values()
    im_folder = im_folder[target.index]
    
    # plot image
    gs = gridspec.GridSpec(len(np.unique(target)),int(im_folder.shape[0]/len(np.unique(target))))
    gs.update(wspace=0, hspace=0, top=0.999, right=0.98,left=0.02,bottom=0.001) # set the spacing between axes. 
    
    idx = 0
    for filename in im_folder:
        ax = plt.subplot(gs[idx])
        ax.imshow(ReadImage(filename))
        #ax.axis('off')
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        #ax.set_aspect('equal')
        if any(np.arange(0,im_folder.shape[0],int(im_folder.shape[0]/len(np.unique(target)))) == idx):
           # ax.set_title(target.iloc[idx])
            ax.set_ylabel(target.iloc[idx]) 
        idx+=1

