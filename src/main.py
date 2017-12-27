#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 21:43:33 2017

@author: goncalofigueira
"""
#==============================================================================
# Import modules
#==============================================================================
from __future__ import print_function
import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pylab as plt
from data_read.data_utils import getFileList, sortTarget, ReadImage
from data_vis.data_preview import DisplaySamples
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#==============================================================================
# Define path and image type
#==============================================================================
path = '/Volumes/LACIE SHARE/capstone_project/datasets/ICIAR2018_BACH_Challenge/Photos/'
im_type = '.tif'
#==============================================================================
# CNN parameters
#==============================================================================
batch_size = 1
num_classes = 4
epochs = 1
img_x, img_y = 1536, 2048
input_shape = (img_x, img_y,1)
#==============================================================================
# Get image list and info
#==============================================================================
im_folder = np.array(getFileList(path,im_type))
# Load csv with image information
im_info = pd.read_csv(getFileList(path,'.csv')[0], header = None)
im_info.columns = ['filename','target']
# match image and label indexes
im_info = sortTarget(im_folder,im_info)


M = np.empty((400,img_x, img_y))
idx = 0
for file in tqdm(im_folder):
    im = ReadImage(im_folder[0])
    im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    im_eq = cv2.equalizeHist(im_gray)
    M[idx,:,:] = np.array(im_eq)
    idx += 1

M = M.reshape(M.shape[0],img_x,img_y,1)  

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
T = im_info.target
T = np.array(le.fit_transform(T))


from sklearn.model_selection import StratifiedShuffleSplit   
split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 0)
for train_index, test_index in split.split(M,T):
    X_train = M[train_index,:,:,:]
    y_train = T[train_index]
    X_test = M[test_index,:,:,:]
    y_test = T[test_index]

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
#norma = 1
#stand = 0
#
#
#if norma:
#    print('Applying normalisation')
#    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
#    X_train = scaler.fit_transform(X_train)
#    X_test = scaler.transform(X_test)
##    pickle.dump(scaler, open(param_filename, 'wb')) # save scaler settings
#
#if stand:
#    print('Applying standardisation')
#    scaler = preprocessing.StandardScaler()
#    X_train = scaler.fit_transform(X_train)
#    X_test = scaler.transform(X_test)  
#    
#    
model = Sequential()
model.add(Conv2D(32,kernel_size = (5,5), strides=(1,1),activation = 'relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(64, (5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(1000,activation = 'relu'))
model.add(Dense(num_classes,activation = 'softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,optimizer =keras.optimizers.SGD(lr=0.01),metrics=['accuracy'])

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self,logs={}):
        self.acc = []
        
    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))
        
history = AccuracyHistory()

model.fit(X_train, y_train, batch_size=batch_size, epochs = epochs, verbose=1, validation_data = (X_test, y_test), callbacks = [history])


score = model.evaluate(X_test, y_test, verbose =0)
print(score[0])
print(score[1])
plt.plot(range(1,11), history.acc)
plt.xlabel('Echos')
plt.xlabel('Accuracy')
plt.show()
 
#    
#    
#from sklearn.neural_network import MLPClassifier
#clf = MLPClassifier(solver='adam', alpha=1e-3, hidden_layer_sizes=(200, 4), random_state=1)
#
#t2 = time.time() 
#clf.fit(X_train, y_train)
#elapsed2 = time.time() - t2
#                    
#print()
#print('Training time: ', elapsed2)
#
##==============================================================================
#
##     Print mean and std score for all parameter combination
##==============================================================================
##means = clf.cv_results_['mean_test_score']
#   # stds = clf.cv_results_['std_test_score']
#   # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#   #     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
#
##==============================================================================
##    Print best parameters
#
##==============================================================================
## TRAIN SET
##==============================================================================
#y_true, y_pred = y_train, clf.predict(X_train)
#print('Confusion matrix:')
#print(confusion_matrix(y_true,y_pred))
#print()
#
#
#y_true, y_pred = y_test, clf.predict(X_test)
#print("Classification on test set:")
#print(classification_report(y_true, y_pred))
#print()
#print('Confusion matrix:')
#print(confusion_matrix(y_true,y_pred))






#target = target.sort_values()
#im_folder = im_folder[target.index]
#ReadFileList(im_folder,im_info)
#plt.figure(1)
#im = ReadImage(im_folder[0])
#im_eq = cv2.equalizeHist(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY))
#im_eq = cv2.cvtColor(im_eq,cv2.COLOR_GRAY2RGB)
#
#plt.subplot(1,2,1)
#plt.imshow(im)
#plt.subplot(1,2,2)
#plt.imshow(im_eq)
#
#plt.figure(2)
#plt.subplot(1,2,1)
#color = ('r','g','b')
#for channel, col in enumerate(color):    
#    histr = cv2.calcHist([im],[channel], None, [256], [0,256])
#    plt.plot(histr, color = col)
#    
#plt.subplot(1,2,2)
#color = ('r','g','b')
#for channel, col in enumerate(color):    
#    histr = cv2.calcHist([im_eq],[channel], None, [256], [0,256])
#    plt.plot(histr, color = col)
#        
#plt.show()


#==============================================================================
# Visualise samples from dataset
#==============================================================================
# Preview dataset - displays 0.5% random samples from each class
# DisplaySamples(im_folder,im_info.target, class_perc = 0.05)
