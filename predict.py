# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 17:21:35 2021

@author: chulab
"""

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.test.is_gpu_available()

loadpath = input('INSERT THE IMAGE FOLDER PATH :\n')
filelist = os.listdir(loadpath)
#savepath = input('INSERT THE SAVE PATH :\n')
savepath = loadpath+'/mask'
os.makedirs(loadpath+'/mask')

modelpath = ".\\bri1234_05_p369_model.h5"
model = tf.keras.models.load_model(modelpath)
size = 256
space = 256
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 1
for file in tqdm(filelist):
    img = cv2.imread(loadpath+'\\'+file,2)    
    test_list=[]
    for i in range(0,img.shape[0],space):
        for j in range(0,img.shape[1],space):
            if i > img.shape[0]-size:
                i = img.shape[0]-size
            if j > img.shape[1]-size:
                j = img.shape[1]-size
            test_data = np.zeros((size,size),dtype=img.dtype)
            test_data = img[i:i+size,j:j+size]
            test_data = cv2.GaussianBlur(test_data,(3,3),0)
            test_list.append(test_data)
            
    test_list_array = np.array(test_list)
    X = np.zeros((len(test_list), IMG_HEIGHT, IMG_WIDTH,1))
    for i in range(len(test_list)):
        test_array = cv2.resize(test_list_array[i], (128,128))
        test_array = np.expand_dims(test_array, axis=2)
        X[i] = test_array
    
    x_test = X
    
    predict_list=[]
    for idx in range(len(x_test)):
        x=np.array(x_test[idx])
        x=np.expand_dims(x, axis=0)
        predict = model.predict(x, verbose=1)
        prediction = (predict > 0.5).astype(np.uint8)
        mask=np.squeeze(prediction)
        predict_list.append(mask)
        
    for order in range(len(predict_list)):
        predict_list[order] = cv2.resize(predict_list[order],(size,size))
    
    predict_list_array = np.array(predict_list)
    prediction = np.zeros_like(img,dtype='uint8')
    for i in range(0,img.shape[0],space):
        for j in range(0,img.shape[1],space):
            if i < img.shape[0]-size and j < img.shape[1]-size:
                prediction[i:i+size,j:j+size] = prediction[i:i+size,j:j+size] + predict_list[j//space + i//space*(img.shape[1]//space+1)]
            
            if j < img.shape[1]-size and i > img.shape[0]-size:
                a = img.shape[0]-size
                prediction[a:a+size,j:j+size] = prediction[a:a+size,j:j+size] + predict_list[j//space + (a//space+1)*(img.shape[1]//space+1)]
            
            if j > img.shape[1]-size and i < img.shape[0]-size:
                j = img.shape[1]-size
                prediction[i:i+size,j:j+size] = prediction[i:i+size,j:j+size] + predict_list[(j//space+1) + i//space*(img.shape[1]//space+1)]
                
            if i > img.shape[0]-size and j > img.shape[1]-size:
                i = img.shape[0]-size
                j = img.shape[1]-size
                prediction[i:i+size,j:j+size] = prediction[i:i+size,j:j+size] + predict_list[(j//space+1) + (i//space+1)*(img.shape[1]//space+1)]
            
    prediction[prediction != 0]= 65535
    
    mask = np.zeros((prediction.shape),dtype='uint16')
    mask[:,:] = prediction
    
    kernel = np.ones((7,7),np.uint8)
    opening = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    
    result = prediction-opening
    
    kernel = np.ones((2,2),np.uint8)
    result2 = cv2.morphologyEx(result,cv2.MORPH_OPEN,kernel)
    
    img[img<=10] = 0
    img[img>10] = 1
    result3 = np.multiply(result2,img)
    
    kernel = np.ones((2,2),np.uint8)
    result4 = cv2.morphologyEx(result3,cv2.MORPH_OPEN,kernel) 
    
    Image.fromarray(result4).save(savepath+'\\'+file)