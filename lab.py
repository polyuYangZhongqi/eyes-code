# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 16:17:22 2019

@author: cszqyang
"""
from scipy import ndimage
import  xml.dom.minidom
import matplotlib.pyplot as plt
import pylab
import random
import cv2
import numpy as np
import os
from skimage.measure import CircleModel
from skimage.measure import ransac, LineModelND
from sklearn.cluster import DBSCAN
#%%
def getCenter(img):
    
    
    edgepoints=[]
    edgeL=(0,0)
    edgeR=(0,0)
    leftflag=0
    rightflag=1
    edgepoints=[]
    cy=0.
    cx=0.
    r=0.
        
        
    image_source=img#读取图片
    if image_source is None:
        print('NO IMAGE')
        return (999,999)
        
    
    img1=image_source.copy()
        
    for y in range(np.shape(img1)[0]-2):
        for x in range(np.shape(img1)[1]-12):
            x=x+5
            current=img1[y][x]
            nexttwo=img1[y][x+2]
            if (int(current)-int(nexttwo))>40:
                leftflag=x+1
                edgeL=(x+2,y)
                   
                break
        for x1 in range(np.shape(img1)[1]-7,-1,-1):
            if x1>2:
                x1=x1
                current=img1[y][x1]
                nexttwo=img1[y][x1-2]
                if (int(current)-int(nexttwo))>40:
                    rightflag=x1-1
                    edgeR=(x1-2,y)
                        
                    
                    break
                
        if (leftflag<rightflag): 
                
            edgepoints.append(edgeR)
            edgepoints.append(edgeL)
    edgepoints=np.asarray(edgepoints)
    #edgepoints=deletEyecorner(edgepoints)
    model_robust, inliers = ransac(edgepoints, CircleModel, min_samples=5,
                               residual_threshold=1, max_trials=1000)
    cy, cx, r = model_robust.params
    return [cy,cx],r
#%%
def centeroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    pos=np.asarray([sum_y/length,sum_x/length])
    return pos


#%%
def findPc(im):
    img=cv2.imread(im,0)
    c,r=getCenter(img)

    mask=np.zeros(np.shape(img))
    
    cv2.circle(mask,(int(round(c[0])),int(round(c[1]))),int(r),(255,255,255),-1)
    mask = mask.astype(np.uint8)
    res = cv2.bitwise_and(img,img,mask = mask)
    tempres=np.sort(res[res!=0], axis=0, kind='quicksort', order=None)
    
    ret,thresh1 = cv2.threshold(res,tempres[-10],255,cv2.THRESH_BINARY)
    points=np.where(thresh1==255)
    points=np.asarray(points).T
    pccluster=DBSCAN(2, 5).fit_predict(points)
    unique, counts = np.unique(pccluster, return_counts=True)
    cludis=999
    
    PC=np.zeros(2)
    for i in unique:
        cluc=points[np.where(pccluster==i)]
        cd=np.linalg.norm(centeroidnp(cluc)-c)
        if cd<cludis:
            cludis=cd
            
            PC=centeroidnp(cluc)
    return PC
#%%
def findPc_channel(im):
    img3=cv2.imread(im,1)
    imggray=cv2.imread(im,0)
    img = img3[:, :, 0]
    c,r=getCenter(imggray)

    mask=np.zeros(np.shape(img))
    
    cv2.circle(mask,(int(round(c[0])),int(round(c[1]))),int(r),(255,255,255),-1)
    mask = mask.astype(np.uint8)
    res = cv2.bitwise_and(img,img,mask = mask)
    tempres=np.sort(res[res!=0], axis=0, kind='quicksort', order=None)
    
    ret,thresh1 = cv2.threshold(res,tempres[-10]-1,255,cv2.THRESH_BINARY)
    points=np.where(thresh1==255)
    points=np.asarray(points).T
    print(len(points))
    pccluster=DBSCAN(2, 5).fit_predict(points)
    unique, counts = np.unique(pccluster, return_counts=True)
    cludis=999
    
    PC=np.zeros(2)
    for i in unique:
        cluc=points[np.where(pccluster==i)]
        cd=np.linalg.norm(centeroidnp(cluc)-c)
        if cd<cludis:
            cludis=cd
            
            PC=centeroidnp(cluc)
    return PC
#%%
def findPc_channel_recmask(im):
    img3=cv2.imread(im,1)
    imggray=cv2.imread(im,0)
    img = img3[:, :, 0]
    c,r=getCenter(imggray)

    mask=np.zeros(np.shape(img))
    
    cv2.rectangle(mask,(int(round(c[0])-3),int(round(c[1]))+3),(int(round(c[0])+3),int(round(c[1]))-3),(255,255,255),-1)
    mask = mask.astype(np.uint8)
    res = cv2.bitwise_and(img,img,mask = mask)
    tempres=np.sort(res[res!=0], axis=0, kind='quicksort', order=None)
    
    ret,thresh1 = cv2.threshold(res,tempres[-10]-1,255,cv2.THRESH_BINARY)
    points=np.where(thresh1==255)
    points=np.asarray(points).T
    print(len(points))
    pccluster=DBSCAN(2,5).fit_predict(points)
    unique, counts = np.unique(pccluster, return_counts=True)
    cludis=999
    
    PC=np.zeros(2)
    for i in unique:
        cluc=points[np.where(pccluster==i)]
        cd=np.linalg.norm(centeroidnp(cluc)-c)
        if cd<cludis:
            cludis=cd
            
            PC=centeroidnp(cluc)
    return PC
#%%
def findPc_recmask(im):
    img3=cv2.imread(im,1)
    imggray=cv2.imread(im,0)
    img = imggray
    c,r=getCenter(imggray)

    mask=np.zeros(np.shape(img))
    
    cv2.rectangle(mask,(int(round(c[0])-3),int(round(c[1]))+3),(int(round(c[0])+3),int(round(c[1]))-3),(255,255,255),-1)
    mask = mask.astype(np.uint8)
    res = cv2.bitwise_and(img,img,mask = mask)
    tempres=np.sort(res[res!=0], axis=0, kind='quicksort', order=None)
    
    ret,thresh1 = cv2.threshold(res,tempres[-10]-1,255,cv2.THRESH_BINARY)
    points=np.where(thresh1==255)
    points=np.asarray(points).T
    
    pccluster=DBSCAN(2,5).fit_predict(points)
    unique, counts = np.unique(pccluster, return_counts=True)
    cludis=999
    
    PC=np.zeros(2)
    for i in unique:
        cluc=points[np.where(pccluster==i)]
        cd=np.linalg.norm(centeroidnp(cluc)-c)
        if cd<cludis:
            cludis=cd
            
            PC=centeroidnp(cluc)
    return PC
#%%
def readcenter_Pc(im):
    irispoints=[]
    for poly in im.getElementsByTagName('polyline'):
        if poly.getAttribute("label")=='center':
            bound=poly.getAttribute("points").split(';')
            for i in bound:
                xy=i.split(',')
                irispoints.append((int(float(xy[0])),int(float(xy[1]))))
    irispoints=np.asarray(irispoints)
    if len(irispoints)==0:
        return np.array([999,999]),0
    
    model_robust, inliers = ransac(irispoints, CircleModel, min_samples=5,
                               residual_threshold=1, max_trials=1000)
    cy, cx, r = model_robust.params
    p=np.asarray([cy, cx])
    return p,r
#%%
target_path=r"C:\Users\cszqyang\Desktop\FPC"
cnt=0
cntPc=0
dom = xml.dom.minidom.parse(r'C:\Users\cszqyang\Desktop\Window.xml')
root = dom.documentElement
qualityflag=False
bb = root.getElementsByTagName('image')
for im in bb:
    qualityflag=False
    bb2 = im.getElementsByTagName('polyline')
    for poly in bb2:
        if poly.getAttribute("label")=='iris':
            etb=poly.getElementsByTagName('attribute')
            for et in etb:
                if et.getAttribute("name")=='Quality':
                    if (et.firstChild.data)=='good':
                        qualityflag=True
                        cnt=cnt+1
                        print(cnt)
    folder=r'C:\Users\cszqyang\Desktop\eyes_all\\'[:-1]
    impath=folder+im.getAttribute("name")
    if qualityflag:
        try:
            f=findPc_recmask(impath)
            rPc,diarPc=readcenter_Pc(im)
            #img=cv2.imread(impath,0)
            #k,rrr=getCenter(img)
            d=np.linalg.norm(rPc-f)
            img=cv2.imread(impath,1)
            print(diarPc)
            if d<diarPc:
                cntPc=cntPc+1
            elif d>100:
                cnt=cnt-1
            else:
                circleread=cv2.circle(img.copy(),(int(f[0]),int(f[1])),2,(255,255,255),1)
                cv2.imwrite(os.path.join(target_path,im.getAttribute("name"))+str(d)+'.jpg',circleread)
        except:
            print(impath)
            
            #%%
print(cntPc)