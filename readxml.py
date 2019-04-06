# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 01:18:50 2019

@author: yzq
"""

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

#打开xml文档
dom = xml.dom.minidom.parse(r'C:\Users\cszqyang\Desktop\Window.xml')
target_path=r"C:\Users\cszqyang\Desktop\failimg"
#得到文档元素对象
root = dom.documentElement
print( root.nodeName)
print( root.nodeValue)
print( root.nodeType)
print( root.ELEMENT_NODE)
print( 'fsfsfsfsf')
bb = root.getElementsByTagName('image')
b= bb[2]


qualityflag=False
bb2 = b.getElementsByTagName('polyline')
for poly in bb2:
    if poly.getAttribute("label")=='iris':
        etb=poly.getElementsByTagName('attribute')
        for et in etb:
            if et.getAttribute("name")=='EyeClosed':
                if (et.firstChild.data)=='open':
                    qualityflag=True
        
print (b.getAttribute("name"))
#%%

cntarea=0
cntE1=0
cntE2=0
cntE3=0
cnt=0
cntE5=0
modicnt=0
#得到文档元素对象
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
    drawboard=cv2.imread(impath,1)
    if qualityflag:
        try:
        
            label,rread=readcenter(im)
    
        except:
            print('er when read')
            print(im.getAttribute("name"))
    
        try:
            pre,rpre=getCenter(impath)
        except:
                print('er when pre')
                print(im.getAttribute("name"))
        dis=np.linalg.norm(np.array(label)-np.array(pre))
        cor=areaRat(label[0],label[1],rread,pre[0],pre[1],rpre,impath)
        '''circleread=cv2.circle(drawboard.copy(),(int(pre[0]),int(pre[1])),int(rpre),(0,255,255),2)
        circlepre=cv2.circle(drawboard.copy(),(int(label[0]),int(label[1])),int(rread),(0,255,255),2)
        cv2.imwrite(os.path.join(target_path,im.getAttribute("name")),circleread)
        cv2.imwrite(os.path.join(target_path,im.getAttribute("name"))+str(cor)+'.jpg',circlepre)'''
        if cor>0.8:
            cntarea=cntarea+1
        else:
            '''circleread=cv2.circle(drawboard.copy(),(int(pre[0]),int(pre[1])),int(rpre),(0,255,255),2)
            circlepre=cv2.circle(drawboard.copy(),(int(label[0]),int(label[1])),int(rread),(0,255,255),2)
            circledpre=cv2.circle(drawboard.copy(),(int(dpre[0]),int(dpre[1])),int(dr),(0,255,255),2)
            cv2.imwrite(os.path.join(target_path,im.getAttribute("name")),circleread)
            cv2.imwrite(os.path.join(target_path,im.getAttribute("name"))+str(cor)+'.jpg',circlepre)
            cv2.imwrite(os.path.join(target_path,im.getAttribute("name"))+'de111111.jpg',circledpre)'''
        if dis<=1 :#and dis/rpre<0.5:
            cntE1=cntE1+1
        elif dis<=2: #and dis/rpre<0.5:
            cntE2=cntE2+1
        elif dis<=3 :#and dis/rpre<0.5:
            cntE3=cntE3+1
        elif dis<=5 :#and dis/rpre<0.5:
            cntE5=cntE5+1
        else:
            print(impath,dis)
            dpre,dr=getCenter_DECN(impath)
            if np.linalg.norm(np.array(label)-np.array(dpre))<3: #and dis/rpre<0.5:
                modicnt=modicnt+1
            '''else:
                circleread=cv2.circle(drawboard.copy(),(int(pre[0]),int(pre[1])),int(rpre),(0,255,255),2)
                circlepre=cv2.circle(drawboard.copy(),(int(label[0]),int(label[1])),int(rread),(0,255,255),2)
                circledpre=cv2.circle(drawboard.copy(),(int(dpre[0]),int(dpre[1])),int(dr),(0,255,255),2)
                cv2.imwrite(os.path.join(target_path,im.getAttribute("name")),circleread)
                cv2.imwrite(os.path.join(target_path,im.getAttribute("name"))+'ddd.jpg',circlepre)
                cv2.imwrite(os.path.join(target_path,im.getAttribute("name"))+'de111111.jpg',circledpre)
    irispoints=[]
    print(im.getAttribute("name"))
    for poly in im.getElementsByTagName('polyline'):
        if poly.getAttribute("label")=='iris':
            bound=poly.getAttribute("points").split(';')
            
            for i in bound:
                xy=i.split(',')
                
                irispoints.append((int(float(xy[0])),int(float(xy[1]))))
            irispoints=np.asarray(irispoints)
                
            model_robust, inliers = ransac(irispoints, CircleModel, min_samples=3,
                               residual_threshold=2, max_trials=1000)
            cy, cx, r = model_robust.params
    im.getAttribute("name")
    cnt=cnt+1'''
#%%
print(cntE1)
print(cntE2)
print(cntE3)
print(cntE5)
print(modicnt)
print(cntarea)
#%%
#%%

#%%
def readcenter(im):
    irispoints=[]
    for poly in im.getElementsByTagName('polyline'):
        if poly.getAttribute("label")=='iris':
            bound=poly.getAttribute("points").split(';')
            for i in bound:
                xy=i.split(',')
                irispoints.append((int(float(xy[0])),int(float(xy[1]))))
    irispoints=np.asarray(irispoints)
    model_robust, inliers = ransac(irispoints, CircleModel, min_samples=5,
                               residual_threshold=1, max_trials=1000)
    cy, cx, r = model_robust.params
    return [cy, cx],r
#%%
def getCenter(image_path):
    
    
    edgepoints=[]
    edgeL=(0,0)
    edgeR=(0,0)
    leftflag=0
    rightflag=1
    edgepoints=[]
    cy=0.
    cx=0.
    r=0.
        
        
    image_source=cv2.imread(image_path,0)#读取图片
    if image_source is None:
        print('NO IMAGE')
        return (999,999)
        
    
    img1=image_source.copy()
        
    for y in range(np.shape(img1)[0]-2):
        for x in range(np.shape(img1)[1]-12):
            x=x+10
            current=img1[y][x]
            nexttwo=img1[y][x+2]
            if (int(current)-int(nexttwo))>40:
                leftflag=x+1
                edgeL=(x+1,y)
                   
                break
        for x1 in range(np.shape(img1)[1]-12,-1,-1):
            if x1>2:
                x1=x1
                current=img1[y][x1]
                nexttwo=img1[y][x1-2]
                if (int(current)-int(nexttwo))>40:
                    rightflag=x1-1
                    edgeR=(x1-1,y)
                        
                    
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
def getCenter_DECN(image_path):
    
    
    edgepoints=[]
    edgeL=(0,0)
    edgeR=(0,0)
    leftflag=0
    rightflag=1
    edgepoints=[]
    cy=0.
    cx=0.
    r=0.
        
        
    image_source=cv2.imread(image_path,0)#读取图片
    if image_source is None:
        print('NO IMAGE')
        return (999,999)
        
    img1 = cv2.GaussianBlur(image_source.copy(),(3,3),0)
    img1=image_source.copy()
        
    for y in range(np.shape(img1)[0]-2):
        for x in range(np.shape(img1)[1]-12):
            x=x+10
            current=img1[y][x]
            nexttwo=img1[y][x+2]
            if (int(current)-int(nexttwo))>40:
                leftflag=x+1
                edgeL=(x+1,y)
                   
                break
        for x1 in range(np.shape(img1)[1]-12,-1,-1):
            if x1>2:
                x1=x1
                current=img1[y][x1]
                nexttwo=img1[y][x1-2]
                if (int(current)-int(nexttwo))>40:
                    rightflag=x1-1
                    edgeR=(x1-1,y)
                        
                    
                    break
                
        if (leftflag<rightflag): 
                
            edgepoints.append(edgeR)
            edgepoints.append(edgeL)
    edgepoints=np.asarray(edgepoints)
    edgepoints=deletEyecorner_notp(edgepoints)
    model_robust, inliers = ransac(edgepoints, CircleModel, min_samples=5,
                               residual_threshold=1, max_trials=1000)
    cy, cx, r = model_robust.params
    return (cy,cx),r
#%%
def deletEyecorner(edgepoints):
    projX=[]
    for i in edgepoints:
        projX.append((i[0],0))
    projX=np.asarray(projX)
    
    edcluster=DBSCAN(3, 5).fit_predict(projX)
    
    unique, counts = np.unique(edcluster, return_counts=True)
    
    counttemp=np.array(counts)
    
    largest=counts.max()
    
    eg1type=unique[np.where(counttemp==largest)[0][0]]
    largedis=(np.sum(projX[np.where(edcluster==eg1type)[0]], axis = 0)[0])/len(projX[np.where(edcluster==eg1type)[0]])
    
    way=100
    t=None
    for i in unique:
        if i>=0 and i!=eg1type:
            jjpos=projX[np.where(edcluster==i)[0]]
            pos=(np.sum(jjpos, axis = 0)[0])/len(jjpos)
            if way>abs(pos-largedis):
                way=abs(pos-largedis)
                t=i
            
    
    eg2type=t
    realDonaldtrump=np.append(edgepoints[np.where(edcluster==eg1type)[0]],edgepoints[np.where(edcluster==eg2type)[0]],axis=0)
    return realDonaldtrump
#%%
def deletEyecorner_notp(edgepoints):
    projX=[]
    for i in edgepoints:
        projX.append((i[0],0))
    projX=np.asarray(projX)
    
    edcluster=DBSCAN(5, 5).fit_predict(edgepoints)
    
    unique, counts = np.unique(edcluster, return_counts=True)
    
    counttemp=np.array(counts)
    
    largest=counts.max()
    
    eg1type=unique[np.where(counttemp==largest)[0][0]]
    largedis=(np.sum(edgepoints[np.where(edcluster==eg1type)[0]], axis = 0)[0])/len(edgepoints[np.where(edcluster==eg1type)[0]])
    #largedis2=(np.sum(edgepoints[np.where(edcluster==eg1type)[0]], axis = 1)[0])/len(edgepoints[np.where(edcluster==eg1type)[0]])
    #largepos=(largedis1,largedis2)
    way=100
    t=None
    for i in unique:
        if i>=0 and i!=eg1type:
            jjpos=projX[np.where(edcluster==i)[0]]
            pos=(np.sum(jjpos, axis = 0)[0])/len(jjpos)
            if way>abs(pos-largedis):
                way=abs(pos-largedis)
                t=i
    
    
    eg2type=t
    realDonaldtrump=np.append(edgepoints[np.where(edcluster==eg1type)[0]],edgepoints[np.where(edcluster==eg2type)[0]],axis=0)
    return realDonaldtrump
#%%
def areaRat(cy,cx,r,cy1,cx1,r1,impath):
    img=cv2.imread(impath,0)
    blackboard1=np.zeros(np.shape(img))
    blackboard2=np.zeros(np.shape(img))
    blackboard1=cv2.circle(blackboard1.copy(),(int(round(cy)),int(round( cx))),int(round(r)),(255,255,255),-1)
    blackboard2=cv2.circle(blackboard2.copy(),(int(round(cy1)),int(round( cx1))),int(round(r1)),(255,255,255),-1)
    bb3=cv2.circle(blackboard2.copy(),(int(round(cy)),int(round( cx))),int(round(r)),(255,255,255),-1)
    a1=len(np.where(blackboard1==255)[0])
    a2=len(np.where(blackboard2==255)[0])
    au=len(np.where(bb3==255)[0])
    ai=a1+a2-au
    return ai/au