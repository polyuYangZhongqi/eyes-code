# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 14:50:18 2019

@author: yzq
"""

import matplotlib.pyplot as plt
import pylab
import random
import cv2
import numpy as np
import os
from skimage.measure import CircleModel
from skimage.measure import ransac, LineModelND
img = cv2.imread(r"C:\Users\yzq\Desktop\eyes_all\(21,10)EyeScreening_4.3_2018-3-14_179_OS.jpg",0) 

plt.figure(num='or',figsize=(15,15))
plt.subplot(4,2,1)     
plt.title('1')  
plt.imshow()

#%%

#%%
'''imonepoint=img.copy()
img = cv2.GaussianBlur(img,(3,3),0)
edgeL=(0,0)
edgeR=(0,0)
for y in range(np.shape(img)[0]-1):
    for x in range(np.shape(img)[1]-2):
        
        current=img[y][x]
        nexttwo=img[y][x+2]
        if (int(current)-int(nexttwo))>35:
            
                edgeL=(x,y)
                
                break
    for x1 in range(np.shape(img)[1]-1,-1,-1):
        
        current=img[y][x1]
        nexttwo=img[y][x1-2]
        if (int(current)-int(nexttwo))>35:
                edgeR=(x1,y)
                
                break
    if edgeR[0]>edgeL[0]:
        #print('R>L')
        imonepoint=cv2.circle(imonepoint.copy(), edgeR, 2, (255, 255, 255), 0)
        imonepoint=cv2.circle(imonepoint.copy(), edgeL, 2, (255, 255, 255), 0)
plt.imshow(imonepoint)'''
#%%

#%%
source_path=r"C:\Users\yzq\Desktop\LG_phone_eyes\eyes_all"
target_path=r"C:\Users\yzq\Desktop\sift"        #输出目标文件路径
source_path_high=r"C:\Users\yzq\Desktop\eyes_all(Photo)(scale)(x2.000000)"
if os.path.exists(target_path):
    
    image_list_high=os.listdir(source_path)      #获得文件名
    
if os.path.exists(target_path):
    
    image_list=os.listdir(source_path)      #获得文件名

outputenh(image_list,source_path,r"C:\Users\yzq\Desktop\sift")
#%%

#%%
edgepoints=np.asarray(edgepoints)
print(edgepoints)
plt.imshow(imonepoint)
#%%

points = np.array(np.nonzero(edgepoints)).T

model_robust, inliers = ransac(edgepoints, CircleModel, min_samples=3,
                               residual_threshold=2, max_trials=1000)
#%%

#%%
def outputPoints(image_list,source_path,target_path):
    i=0
    
    for file in image_list:
        
        edgepoints=[]
        edgeL=(0,0)
        edgeR=(0,0)
        leftflag=0
        rightflag=1
        edgepoints=[]
        cy=0.
        cx=0.
        r=0.
        
        i=i+1
        image_c=cv2.imread(os.path.join(source_path,file),1)
        image_source=cv2.imread(os.path.join(source_path,file),0)#读取图片
        
        img1 = cv2.GaussianBlur(image_source.copy(),(3,3),0)
        circlefiti=image_source.copy()
        edgept=image_c.copy()
        
        for y in range(np.shape(img1)[0]-2):
            for x in range(np.shape(img1)[1]-2):
        
                current=img1[y][x]
                nexttwo=img1[y][x+2]
                if (int(current)-int(nexttwo))>40:
                    leftflag=x+1
                    edgeL=(x+1,y)
                   
                    break
            for x1 in range(np.shape(img1)[1]-1,-1,-1):
                if x1>2:
                    current=img1[y][x1]
                    nexttwo=img1[y][x1-2]
                    if (int(current)-int(nexttwo))>40:
                        rightflag=x1-1
                        edgeR=(x1-1,y)
                        
                    
                        break
                
            if (leftflag<rightflag): 
                
                edgepoints.append(edgeR)
                edgepoints.append(edgeL)
                edgept=cv2.circle(edgept.copy(), edgeR, 2, (255, 255, 255), 0)
                edgept=cv2.circle(edgept.copy(), edgeL, 2, (255, 255, 255), 0)
        edgepoints=np.asarray(edgepoints)
        try:
            
            model_robust, inliers = ransac(edgepoints, CircleModel, min_samples=3,
                               residual_threshold=2, max_trials=1000)
            cy, cx, r = model_robust.params
        except:
            print(None)
        circlefiti=cv2.circle(image_c.copy(),(int(round(cy)),int(round( cx))),int(round(r)),(0,255,255),1)
        cv2.imwrite(os.path.join(target_path,file),circlefiti)
        cv2.imwrite(os.path.join(target_path,file)+'ddd.jpg',edgept)
    print(i)
    return None
def threshG(img):
    iar=img.flatten()
    iar=np.sort(iar)
    med=iar[int(len(iar)/2)]
    img=cv2.GaussianBlur(img.copy(),(3,3),0)
    ret1,thresh1=cv2.threshold(img,med,255,cv2.THRESH_BINARY)
    return thresh1
def cannyThresh(img):
    img = cv2.GaussianBlur(img,(3,3),0)
    thresh1=threshG(img)
    cannyThresh=cv2.Canny(thresh1,300, 100, L2gradient=True)
    return cannyThresh
#%%
def outputenh(image_list,source_path,target_path):
    i=0
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4,4))
    for file in image_list:
        
        edgepoints=[]
        edgeL=(0,0)
        edgeR=(0,0)
        leftflag=0
        rightflag=1
        edgepoints=[]
        cy=0.
        cx=0.
        r=0.
        
        i=i+1
        print(i)
        image_c=cv2.imread(os.path.join(source_path,file),1)
        image_source=cv2.imread(os.path.join(source_path,file),0)#读取图片
        image_source=clahe.apply(image_source)
        img1 = cv2.GaussianBlur(image_source.copy(),(3,3),0)
        circlefiti=image_source.copy()
        edgept=image_c.copy()
        
        for y in range(np.shape(img1)[0]-2):
            for x in range(np.shape(img1)[1]-2):
        
                current=img1[y][x]
                nexttwo=img1[y][x+2]
                if (int(current)-int(nexttwo))>40:
                    leftflag=x+1
                    edgeL=(x+1,y)
                   
                    break
            for x1 in range(np.shape(img1)[1]-1,-1,-1):
                if x1>2:
                    current=img1[y][x1]
                    nexttwo=img1[y][x1-2]
                    if (int(current)-int(nexttwo))>40:
                        rightflag=x1-1
                        edgeR=(x1-1,y)
                        
                    
                        break
                
            if (leftflag<rightflag): 
                
                edgepoints.append(edgeR)
                edgepoints.append(edgeL)
                edgept=cv2.circle(edgept.copy(), edgeR, 2, (255, 255, 255), 0)
                edgept=cv2.circle(edgept.copy(), edgeL, 2, (255, 255, 255), 0)
        edgepoints=np.asarray(edgepoints)
        try:
            
            model_robust, inliers = ransac(edgepoints, CircleModel, min_samples=3,
                               residual_threshold=2, max_trials=1000)
            cy, cx, r = model_robust.params
        except:
            print(None)
        circlefiti=cv2.circle(image_c.copy(),(int(round(cy)),int(round( cx))),int(round(r)),(0,255,255),1)
        cv2.imwrite(os.path.join(target_path,file),circlefiti)
        cv2.imwrite(os.path.join(target_path,file)+'ddd.jpg',edgept)
    print(i)
    return None