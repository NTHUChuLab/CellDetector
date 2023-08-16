import os
import cv2
import numpy as np
from tqdm import tqdm

def brightimg(rawpath, outputpath, multiply_number):
    filelist = os.listdir(rawpath)
    for file in tqdm(filelist):
        imagepath = rawpath + '\\' + file
        img = cv2.imread(imagepath, 2)
        img = cv2.GaussianBlur(img,(3,3),0)
        img = np.array(img)
        brightimg = np.zeros_like(img,dtype='uint16')
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i,j] > 32:
                    brightimg[i,j] = img[i,j] * multiply_number
                else:
                    brightimg[i,j] = img[i,j] #*(multiply_number-1)
        threshold, brightimg = cv2.threshold(brightimg, 15000, 65535, cv2.THRESH_TRUNC)
        cv2.imwrite(outputpath + '/' + file, brightimg)

def brightimg2(rawpath, outputpath, plus_number):
    plus_number = float(plus_number)
    filelist = os.listdir(rawpath)
    for file in tqdm(filelist):
        imagepath = rawpath + '\\' + file
        img = cv2.imread(imagepath, 2)
        img = cv2.GaussianBlur(img,(3,3),0)
        img = np.array(img)
        brightimg = np.zeros_like(img,dtype='uint16')
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i,j] > 32:
                    brightimg[i,j] = img[i,j] + plus_number
                else:
                    brightimg[i,j] = img[i,j] #*((plus_number/300) +1)
        threshold, brightimg = cv2.threshold(brightimg, 15000, 65535, cv2.THRESH_TRUNC)
        cv2.imwrite(outputpath + '\\' + file, brightimg)
