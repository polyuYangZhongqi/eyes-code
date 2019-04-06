# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 16:38:23 2019

@author: yzq
"""
import matplotlib.pyplot as plt
import pylab
import random
import cv2
import numpy as np
import os
img = cv2.imread(r"C:\Users\yzq\Desktop\eyes_all\(29,6)EyeScreening_4.3_2018-3-15_2212_OS.jpg",0) 
img = cv2.GaussianBlur(img,(3,3),0)
canny = cv2.Canny(img,300, 100, L2gradient=True)


#%%

#%%
import math
cannyT=cannyThresh(img)
pointF=np.where(cannyT==255)
points=np.append([pointF[1]],[pointF[0]],axis=0)
points=points.T
ellipse = cv2.fitEllipse(points)


rawCanny=cv2.Canny(cv2.GaussianBlur(img,(3,3),0),300, 100, L2gradient=True)
pointF_raw=np.where(rawCanny==255)
points_raw=np.append([pointF_raw[1]],[pointF_raw[0]],axis=0)
points_raw=points_raw.T

elR=FitEllipse_RANSAC(points, img, max_itts=50, max_refines=10, max_perc_inliers=95.0)


ellipse_raw = cv2.fitEllipse(points_raw)
poly_raw = cv2.ellipse2Poly((int(ellipse_raw[0][0]), int(ellipse_raw[0][1])), (int(ellipse_raw[1][0] / 2), int(ellipse_raw[1][1] / 2)), int(ellipse_raw[2]), 0, 360, 5)
polyimg_raw=cv2.polylines(img.copy(), [poly_raw], 1, (255, 0, 255))

poly = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])), (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)), int(ellipse[2]), 0, 360, 5)
polyimg=cv2.polylines(img.copy(), [poly], 1, (255, 0, 255))
_,contours,hierarchy = cv2.findContours(cannyT, 1, 2)
print(ellipse)

linecenter=[int(ellipse[0][0]), int(ellipse[0][1])]
length=int(ellipse[1][1] / 2)
angle=int(ellipse[2])
leftpoint=(int(linecenter[0] - length * math.sin(angle * 3.14 / 180.0)),int(linecenter[1] + length * math.cos(angle * 3.14 / 180.0)))
rightpoint=(int(linecenter[0] + length * math.sin(angle * 3.14 / 180.0)),int(linecenter[1] - length * math.cos(angle * 3.14 / 180.0 )))
print(rightpoint)
imgline=cv2.line(img.copy(),leftpoint,rightpoint,(255,0,0),1)
linelit=createLineIterator(leftpoint, rightpoint, img)

ar=np.array([])

print('222222222222222222')
for i in range(len(linelit)):
    current=linelit[i][2]
    
    ar=np.append(ar,current)
for i in range(len(linelit)):
    current=linelit[i][2]
    nexttwo=linelit[i+2][2]
   
    if (current-nexttwo)>45:
        edgeL=(linelit[i][0],linelit[i][1])
        break
for i in range(len(linelit)-1,-1,-1):
    current=linelit[i][2]
    nexttwo=linelit[i-2][2]
    if (current-nexttwo)>45:
        edgeR=(linelit[i][0],linelit[i][1])
        break    
imonepoint=cv2.circle(img.copy(), edgeR, 2, (255, 255, 255), 0)
imtwopoint=cv2.circle(imonepoint.copy(), edgeL, 2, (255, 255, 255), 0)
print('sssssss')

ctimg=cv2.drawContours(img.copy(),contours,0,(0,0,255),1)
plt.figure(num='or',figsize=(15,15))
plt.subplot(4,2,1)
plt.title('3')
plt.imshow(imtwopoint)
plt.subplot(4,2,2)
plt.title('2')
plt.imshow(imgline)
plt.subplot(4,2,3)
plt.title('1')
plt.imshow(polyimg)
plt.subplot(4,2,4)
plt.title('2')
plt.imshow(imgline)
print(len(linelit))
#%%
print(len(ar))
plt.bar(range(len(ar)), ar)
plt.show()
#%%
outputEllipandPoints(image_list,source_path,r"C:\Users\yzq\Desktop\sift")
#%%
kernelup=np.array([[-1,-1,-1],
                   [0 , 3, 0],
                   [0,0,0]])
kerneldo=np.array([[0,0,0],
                   [0 , 3, 0],
                   [-1,-1,-1]])
res1 = cv2.filter2D(cv2.GaussianBlur(img.copy(),(3,3),0),-1,kernelup)
res2 = cv2.filter2D(cv2.GaussianBlur(img.copy(),(3,3),0),-1,kerneldo)
cannyThresh1=cv2.Canny(res1,200, 50, L2gradient=True)
cannyThresh2=cv2.Canny(res2,200, 50, L2gradient=True)
plt.figure(num='or',figsize=(15,15))
plt.subplot(4,2,1)     
plt.title('1')  
plt.imshow(res1)
plt.subplot(4,2,2) 
plt.title('2') 
plt.imshow(res2)
plt.subplot(4,2,3)     
plt.title('3')  
plt.imshow(cannyThresh1)
plt.subplot(4,2,4) 
plt.title('4') 
plt.imshow(cannyThresh2)


#%%
img = cv2.imread(r"C:\Users\yzq\Desktop\eyes_all\(25,11)EyeScreening_4.3_2018-3-12_1755_OS.jpg",0) 
detector = cv2.SimpleBlobDetector_create()
keypoints = detector.detect(threshG(img))
print(keypoints)
im_with_keypoints = cv2.drawKeypoints(img.copy(), keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(im_with_keypoints)

#%%
source_path=r"C:\Users\yzq\Desktop\eyes_all"
target_path=r"C:\Users\yzq\Desktop\thres"        #输出目标文件路径
source_path_high=r"C:\Users\yzq\Desktop\eyes_all(Photo)(scale)(x2.000000)"
if os.path.exists(target_path):
    
    image_list_high=os.listdir(source_path)      #获得文件名
    
if os.path.exists(target_path):
    
    image_list=os.listdir(source_path)      #获得文件名

#outputRgb(image_list,source_path,r"C:\Users\yzq\Desktop\rgb")
#%%
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
def outputThre(image_list,source_path,target_path):
    i=0
    for file in image_list:
        i=i+1
        
        image_source=cv2.imread(os.path.join(source_path,file),0)#读取图片
        
        outi=cannyThresh(image_source)
      
        
        cv2.imwrite(os.path.join(target_path,file),outi)
    print(i)
    return None
#%%
def outputEllipandPoints_raw(image_list,source_path,target_path):
    i=0
    for file in image_list:
        i=i+1
        
        image_source=cv2.imread(os.path.join(source_path,file),0)#读取图片
        cannyT=cv2.Canny(cv2.GaussianBlur(image_source,(3,3),0),300, 100, L2gradient=True)
        pointF=np.where(cannyT==255)
        points=np.append([pointF[1]],[pointF[0]],axis=0)
        points=points.T
        ellipse = cv2.fitEllipse(points)
        
        linecenter=[int(ellipse[0][0]), int(ellipse[0][1])]
        length=int(ellipse[1][1] / 2)
        angle=int(ellipse[2])
        leftpoint=(int(linecenter[0] - length * math.sin(angle * 3.14 / 180.0)),int(linecenter[1] - length * math.cos(angle * 3.14 / 180.0)))
        rightpoint=(int(linecenter[0] + length * math.sin(angle * 3.14 / 180.0)),int(linecenter[1] + length * math.cos(angle * 3.14 / 180.0)))
        linelit=createLineIterator(leftpoint, rightpoint, img)
        for i in range(len(linelit)):
            current=linelit[i][2]
            nexttwo=linelit[i+2][2]
            if (current-nexttwo)>35:
                edgeL=(linelit[i][0],linelit[i][1])
                break
        for i in range(len(linelit)-1,-1,-1):
            current=linelit[i][2]
            nexttwo=linelit[i-2][2]
            if (current-nexttwo)>35:
                edgeR=(linelit[i][0],linelit[i][1])
                break    
        
        imonepoint=cv2.circle(img.copy(), edgeR, 2, (255, 255, 255), 0)
        imtwopoint=cv2.circle(imonepoint.copy(), edgeL, 2, (255, 255, 255), 0)
        
        poly = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])), (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)), int(ellipse[2]), 0, 360, 5)
        polyimg=cv2.polylines(imtwopoint.copy(), [poly], 1, (255, 0, 255))
        cv2.imwrite(os.path.join(target_path,file),polyimg)
    print(i)
    return None
    #%%
def outputEllip(image_list,source_path,target_path):
    i=0
    for file in image_list:
        i=i+1
        
        image_source=cv2.imread(os.path.join(source_path,file),0)#读取图片
        cv2.Canny(cv2.GaussianBlur(image_source,(3,3),0),300, 100, L2gradient=True)
        pointF=np.where(cannyT==255)
        points=np.append([pointF[1]],[pointF[0]],axis=0)
        points=points.T
        ellipse = cv2.fitEllipse(points)
        
        poly = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])), (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)), int(ellipse[2]), 0, 360, 5)
        polyimg=cv2.polylines(image_source.copy(), [poly], 1, (255, 0, 255))
        
        cv2.imwrite(os.path.join(target_path,file),polyimg)
    print(i)
    return None
#%%
def outputEllipandPoints(image_list,source_path,target_path):
    i=0
    for file in image_list:
        i=i+1
        
        image_source=cv2.imread(os.path.join(source_path,file),0)#读取图片
        cannyT=cannyThresh(image_source)
        pointF=np.where(cannyT==255)
        points=np.append([pointF[1]],[pointF[0]],axis=0)
        points=points.T
        ellipse = cv2.fitEllipse(points)
        
        linecenter=[int(ellipse[0][0]), int(ellipse[0][1])]
        length=int(ellipse[1][1] / 2)
        angle=int(ellipse[2])
        leftpoint=(int(linecenter[0] - length * math.sin(angle * 3.14 / 180.0)),int(linecenter[1] - length * math.cos(angle * 3.14 / 180.0)))
        rightpoint=(int(linecenter[0] + length * math.sin(angle * 3.14 / 180.0)),int(linecenter[1] + length * math.cos(angle * 3.14 / 180.0)))
        linelit=createLineIterator(leftpoint, rightpoint, img)
        for i in range(len(linelit)):
            current=linelit[i][2]
            nexttwo=linelit[i+2][2]
            if (int(current)-int(nexttwo))>45:
                edgeL=(linelit[i][0],linelit[i][1])
                break
        for i in range(len(linelit)-1,-1,-1):
            current=linelit[i][2]
            nexttwo=linelit[i-2][2]
            if (int(current)-int(nexttwo))>45:
                edgeR=(linelit[i][0],linelit[i][1])
                break    
        
        imonepoint=cv2.circle(image_source.copy(), edgeR, 2, (255, 255, 255), 0)
        imtwopoint=cv2.circle(imonepoint.copy(), edgeL, 2, (255, 255, 255), 0)
        
        poly = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])), (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)), int(ellipse[2]), 0, 360, 5)
        polyimg=cv2.polylines(imtwopoint, [poly], 1, (255, 0, 255))
        cv2.imwrite(os.path.join(target_path,file),polyimg)
    print(i)
    return None
#%%
def outputPoints(image_list,source_path,target_path):
    i=0
    edgeL=(0,0)
    edgeR=(0,0)
    for file in image_list:
        i=i+1
        image_c=cv2.imread(os.path.join(source_path,file),1)
        image_source=cv2.imread(os.path.join(source_path,file),0)#读取图片
        imonepoint=image_c.copy()
        img = cv2.GaussianBlur(image_source.copy(),(3,3),0)
        for y in range(np.shape(img)[0]-2):
            for x in range(np.shape(img)[1]-2):
        
                current=img[y][x]
                nexttwo=img[y][x+2]
                if (int(current)-int(nexttwo))>40:
            
                    edgeL=(x,y)
                
                    break
            for x1 in range(np.shape(img)[1]-1,-1,-1):
                if x1>2:
                    current=img[y][x1]
                    nexttwo=img[y][x1-2]
                if (int(current)-int(nexttwo))>40:
                    edgeR=(x1,y)
                
                    break
                
       
            imonepoint=cv2.circle(imonepoint.copy(), edgeR, 2, (0, 255, 255), 0)
            imonepoint=cv2.circle(imonepoint.copy(), edgeL, 2, (255, 255, 0), 0)
        cv2.imwrite(os.path.join(target_path,file),imonepoint)
    print(i)
    return None
    #%%
def outputRgb(image_list,source_path,target_path):
    i=0
    for file in image_list:
        i=i+1
    
        image_source=cv2.imread(os.path.join(source_path,file))#读取图片
        filer=file+'R.jpg'
        imr=image_source[:,:,0]
        fileg=file+'g.jpg'
        fileb=file+'b.jpg'
        img=image_source[:,:,1]
        imb=image_source[:,:,2]
      
        
        cv2.imwrite(os.path.join(target_path,filer),imr)
        cv2.imwrite(os.path.join(target_path,fileg),img)
        cv2.imwrite(os.path.join(target_path,fileb),imb)
    print(i)
    return None
#%%
def FitEllipse_RANSAC_Support(pnts, roi, cfg, max_itts=5, max_refines=3, max_perc_inliers=95.0):
    '''
    Robust ellipse fitting to segmented boundary with image support
    Parameters
    ----
    pnts : n x 2 array of integers
        Candidate pupil-iris boundary points from edge detection
    roi : 2D scalar array
        Grayscale image of pupil-iris region for support calculation.
    max_itts : integer
        Maximum RANSAC ellipse candidate iterations
    max_refines : integer
        Maximum RANSAC ellipse inlier refinements
    max_perc_inliers : float
        Maximum inlier percentage of total points for convergence
    Returns
    ----
    best_ellipse : tuple of tuples
        Best fitted ellipse parameters ((x0, y0), (a,b), theta)
    '''

    # Debug flagTure
    DEBUG = True

    # Output flags
    graphics=True

    # Suppress invalid values
    np.seterr(invalid='ignore')

    # Maximum normalized error squared for inliers
    max_norm_err_sq = 4.0

    # Tiny circle init
    best_ellipse = ((0,0),(1e-6,1e-6),0)

    # High support is better, so init with -Infinity
    best_support = -np.inf

    # Create display window and init overlay image
    if graphics:
        cv2.namedWindow('RANSAC', cv2.WINDOW_AUTOSIZE)

    # Count pnts (n x 2)
    n_pnts = pnts.shape[0]

    # Break if too few points to fit ellipse (RARE)
    if n_pnts < 5:
        return best_ellipse

    # Precalculate roi intensity gradients
    dIdx = cv2.Sobel(roi, cv2.CV_32F, 1, 0)
    dIdy = cv2.Sobel(roi, cv2.CV_32F, 0, 1)

    # Ransac iterations
    for itt in range(0,max_itts):

        # Select 5 points at random
        sample_pnts = np.asarray(random.sample(pnts, 5))

        # Fit ellipse to points
        ellipse = cv2.fitEllipse(sample_pnts)

        # Dot product of ellipse and image gradients for support calculation
        grad_dot = EllipseImageGradDot(sample_pnts, ellipse, dIdx, dIdy)

        # Skip this iteration if one or more dot products are <= 0
        # implying that the ellipse is unlikely to bound the pupil
        if all(grad_dot > 0):

            # Refine inliers iteratively
            for refine in range(0,max_refines):

                # Calculate normalized errors for all points
                norm_err = EllipseNormError(pnts, ellipse)

                # Identify inliers
                inliers = np.nonzero(norm_err**2 < max_norm_err_sq)[0]

                # Update inliers set
                inlier_pnts = pnts[inliers]

                # Protect ellipse fitting from too few points
                if inliers.size < 5:
                    if DEBUG: print('Break < 5 Inliers (During Refine)')
                    break

                # Fit ellipse to refined inlier set
                ellipse = cv2.fitEllipse(inlier_pnts)

            # End refinement

            # Count inliers (n x 2)
            n_inliers    = inliers.size
            perc_inliers = (n_inliers * 100.0) / n_pnts

            # Calculate support for the refined inliers
            support = EllipseSupport(inlier_pnts, ellipse, dIdx, dIdy)

            # Report on RANSAC progress
            if DEBUG:
                print('RANSAC %d,%d : %0.3f (%0.1f)' % (itt, refine, support, best_support))

            # Update overlay image and display
            if graphics:
                overlay = cv2.cvtColor(roi/2,cv2.COLOR_GRAY2RGB)
                OverlayRANSACFit(overlay, pnts, inlier_pnts, ellipse)
                cv2.imshow('RANSAC', overlay)
                cv2.waitKey(5)

            # Update best ellipse
            if support > best_support:
                best_support = support
                best_ellipse = ellipse

        else:

            # Ellipse gradients did not match image gradients
            support = 0.0
            perc_inliers = 0.0

        if perc_inliers > max_perc_inliers:
            if DEBUG: print('Break Max Perc Inliers')
            break

    return best_ellipse
#%%
def EllipseSupport(pnts, ellipse, dIdx, dIdy):
    """
    Ellipse support function
    """

    if pnts.size < 5:
        return -np.inf

    # Return sum of (grad Q . grad image) over point set
    return EllipseImageGradDot(pnts, ellipse, dIdx, dIdy).sum()


def EllipseImageGradDot(pnts, ellipse, dIdx, dIdy):

    # Calculate normalized grad Q at inlier pnts
    distance, grad, absgrad, normgrad = ConicFunctions(pnts, ellipse)

    # Extract vectors of x and y values
    x, y = pnts[:,0], pnts[:,1]

    # Extract image gradient at inlier points
    dIdx_pnts = dIdx[y,x]
    dIdy_pnts = dIdy[y,x]

    # Construct intensity gradient array (2 x N)
    gradI = np.array( (dIdx_pnts, dIdy_pnts) )

    # Calculate the sum of the column-wise dot product of normgrad and gradI
    # http://stackoverflow.com/questions/6229519/numpy-column-wise-dot-product
    return np.einsum('ij,ij->j', normgrad, gradI)
def EllipseError(pnts, ellipse):
    """
    Ellipse fit error function
    """

    # Suppress divide-by-zero warnings
    np.seterr(divide='ignore')

    # Calculate algebraic distances and gradients of all points from fitted ellipse
    distance, grad, absgrad, normgrad = ConicFunctions(pnts, ellipse)

    # Calculate error from distance and gradient
    # See Swirski et al 2012
    # TODO : May have to use distance / |grad|^0.45 - see Swirski source

    # Gradient array has x and y components in rows (see ConicFunctions)
    err = distance / absgrad

    return err


def EllipseNormError(pnts, ellipse):
    """
    Error normalization factor, alpha
    Normalizes cost to 1.0 at point 1 pixel out from minor vertex along minor axis
    """

    # Ellipse tuple has form ( ( x0, y0), (bb, aa), phi_b_deg) )
    # Where aa and bb are the major and minor axes, and phi_b_deg
    # is the CW x to minor axis rotation in degrees
    (x0,y0), (bb,aa), phi_b_deg = ellipse

    # Semiminor axis
    b = bb/2

    # Convert phi_b from deg to rad
    phi_b_rad = phi_b_deg * np.pi / 180.0

    # Minor axis vector
    bx, by = np.cos(phi_b_rad), np.sin(phi_b_rad)

    # Point one pixel out from ellipse on minor axis
    p1 = np.array( (x0 + (b + 1) * bx, y0 + (b + 1) * by) ).reshape(1,2)

    # Error at this point
    err_p1 = EllipseError(p1, ellipse)

    # Errors at provided points
    err_pnts = EllipseError(pnts, ellipse)

    return err_pnts / err_p1
def Conic2Geometric(conic):
    """
    Merge geometric parameter functions from van Foreest code
    References
    ----
    http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
    """

    # Extract modified conic parameters
    A,B,C,D,E,F = conic[0], conic[1]/2, conic[2], conic[3]/2, conic[4]/2, conic[5]

    # Usefult intermediates
    dAC = A-C
    Z = np.sqrt( 1 + 4*B*B/(dAC*dAC) )

    # Center
    num = B * B - A * C
    x0 = (C * D - B * E) / num
    y0 = (A * E - B * D) / num

    # Axis lengths
    up    = 2 * (A*E*E + C*D*D + F*B*B - 2*B*D*E - A*C*F)
    down1 = (B*B-A*C) * ( -dAC*Z - (C+A) )
    down2 = (B*B-A*C) * (  dAC*Z - (C+A) )
    b, a  = np.sqrt(up/down1), np.sqrt(up/down2)

    # Minor axis rotation angle in degrees (CW from x axis, origin upper left)
    phi_b_deg =  0.5 * np.arctan(2 * B / dAC) * 180.0 / np.pi

    # Note OpenCV ellipse parameter format (full axes)
    return (x0,y0), (2*b, 2*a), phi_b_deg


def ConicFunctions(pnts, ellipse):
    """
    Calculate various conic quadratic curve support functions
    General 2D quadratic curve (biquadratic)
    Q = Ax^2 + Bxy + Cy^2 + Dx + Ey + F
    For point on ellipse, Q = 0, with appropriate coefficients
    Parameters
    ----
    pnts : n x 2 array of floats
    ellipse : tuple of tuples
    Returns
    ----
    distance : array of floats
    grad : array of floats
    absgrad : array of floats
    normgrad : array of floats
    References
    ----
    Adapted from Swirski's ConicSection.h
    https://bitbucket.org/Leszek/pupil-tracker/
    """

    # Suppress invalid values
    np.seterr(invalid='ignore')

    # Convert from geometric to conic ellipse parameters
    conic = Geometric2Conic(ellipse)

    # Row vector of conic parameters (Axx, Axy, Ayy, Ax, Ay, A1) (1 x 6)
    C = np.array(conic)

    # Extract vectors of x and y values
    x, y = pnts[:,0], pnts[:,1]

    # Construct polynomial array (6 x n)
    X = np.array( ( x*x, x*y, y*y, x, y, np.ones_like(x) ) )

    # Calculate Q/distance for all points (1 x n)
    distance = C.dot(X)

    # Quadratic curve gradient at (x,y)
    # Analytical grad of Q = Ax^2 + Bxy + Cy^2 + Dx + Ey + F
    # (dQ/dx, dQ/dy) = (2Ax + By + D, Bx + 2Cy + E)

    # Construct conic gradient coefficients vector (2 x 3)
    Cg = np.array( ( (2*C[0], C[1], C[3]), (C[1], 2*C[2], C[4]) ) )

    # Construct polynomial array (3 x n)
    Xg = np.array( (x, y, np.ones_like(x) ) )

    # Gradient array (2 x n)
    grad = Cg.dot(Xg)

    # Normalize gradient -> unit gradient vector
    # absgrad = np.apply_along_axis(np.linalg.norm, 0, grad)
    absgrad = np.sqrt(np.sqrt(grad[0,:]**2 + grad[1,:]**2))
    normgrad = grad / absgrad

    return distance, grad, absgrad, normgrad


def Eccentricity(ellipse):
    '''
    Calculate eccentricity of an ellipse
    '''

    # Ellipse tuple has form ( ( x0, y0), (bb, aa), phi_b_deg) )
    # Where aa and bb are the major and minor axes, and phi_b_deg
    # is the CW x to minor axis rotation in degrees
    (x0,y0), (bb, aa), phi_b_deg = ellipse

    return np.sqrt(1 - (bb/aa)**2)


def OverlayRANSACFit(img, all_pnts, inlier_pnts, ellipse):
    """
    NOTE
    ----
    All points are (x,y) pairs, but arrays are (row, col) so swap
    coordinate ordering for correct positioning in array
    """

    # Overlay all pnts in red
    for col,row in all_pnts:
        img[row,col] = [0,0,255]

    # Overlay inliers in green
    for col,row in inlier_pnts:
        img[row,col] = [0,255,0]

    # Overlay inlier fitted ellipse in yellow
    cv2.ellipse(img, ellipse, (0,255,255), 1)
def Geometric2Conic(ellipse):
    """
    Geometric to conic parameter conversion
    References
    ----
    Adapted from Swirski's ConicSection.h
    https://bitbucket.org/Leszek/pupil-tracker/
    """

    # Ellipse tuple has form ( ( x0, y0), (bb, aa), phi_b_deg) )
    # Where aa and bb are the major and minor axes, and phi_b_deg
    # is the CW x to minor axis rotation in degrees
    (x0,y0), (bb, aa), phi_b_deg = ellipse

    # Semimajor and semiminor axes
    a, b = aa/2, bb/2

    # Convert phi_b from deg to rad
    phi_b_rad = phi_b_deg * np.pi / 180.0

    # Major axis unit vector
    ax, ay = -np.sin(phi_b_rad), np.cos(phi_b_rad)

    # Useful intermediates
    a2 = a*a
    b2 = b*b

    #
    # Conic parameters
    #
    if a2 > 0 and b2 > 0:

        A = ax*ax / a2 + ay*ay / b2;
        B = 2*ax*ay / a2 - 2*ax*ay / b2;
        C = ay*ay / a2 + ax*ax / b2;
        D = (-2*ax*ay*y0 - 2*ax*ax*x0) / a2 + (2*ax*ay*y0 - 2*ay*ay*x0) / b2;
        E = (-2*ax*ay*x0 - 2*ay*ay*y0) / a2 + (2*ax*ay*x0 - 2*ax*ax*y0) / b2;
        F = (2*ax*ay*x0*y0 + ax*ax*x0*x0 + ay*ay*y0*y0) / a2 + (-2*ax*ay*x0*y0 + ay*ay*x0*x0 + ax*ax*y0*y0) / b2 - 1;

    else:

        # Tiny dummy circle - response to a2 or b2 == 0 overflow warnings
        A,B,C,D,E,F = (1,0,1,0,0,-1e-6)

    # Compose conic parameter array
    conic = np.array((A,B,C,D,E,F))

    return conic
#%%
def FitEllipse_RANSAC(pnts, roi, max_itts, max_refines, max_perc_inliers):
    '''
    Robust ellipse fitting to segmented boundary points
    Parameters
    ----
    pnts : n x 2 array of integers
        Candidate pupil-iris boundary points from edge detection
    roi : 2D scalar array
        Grayscale image of pupil-iris region for display only
    max_itts : integer
        Maximum RANSAC ellipse candidate iterations
    max_refines : integer
        Maximum RANSAC ellipse inlier refinements
    max_perc_inliers : float
        Maximum inlier percentage of total points for convergence
    Returns
    ----
    best_ellipse : tuple of tuples
        Best fitted ellipse parameters ((x0, y0), (a,b), theta)
    '''

    # Debug flag
    DEBUG = True

    # Output flags
    graphics = False

    # Suppress invalid values
    np.seterr(invalid='ignore')

    # Maximum normalized error squared for inliers
    max_norm_err_sq = 4.0

    # Tiny circle init
    best_ellipse = ((0,0),(1e-6,1e-6),0)

    # Create display window and init overlay image
    if graphics:
        cv2.namedWindow('RANSAC', cv2.WINDOW_AUTOSIZE)

    # Count pnts (n x 2)
    n_pnts = pnts.shape[0]

    # Break if too few points to fit ellipse (RARE)
    if n_pnts < 5:
        print('FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF')
        return best_ellipse

    # Ransac iterations
    for itt in range(0,max_itts):

        # Select 5 points at random
        sample_pnts = np.asarray(random.sample(list(pnts), 5))

        # Fit ellipse to points
        ellipse = cv2.fitEllipse(sample_pnts)

        # Refine inliers iteratively
        for refine in range(0,max_refines):

            # Calculate normalized errors for all points
            norm_err = EllipseNormError(pnts, ellipse)

            # Identify inliers
            inliers = np.nonzero(norm_err**2 < max_norm_err_sq)[0]

            # Update inliers set
            inlier_pnts = pnts[inliers]

            # Protect ellipse fitting from too few points
            if inliers.size < 5:
                if DEBUG: print('Break < 5 Inliers (During Refine)')
                break

            # Fit ellipse to refined inlier set
            ellipse = cv2.fitEllipse(inlier_pnts)

        # End refinement

        # Count inliers (n x 2)
        n_inliers    = inliers.size
        perc_inliers = (n_inliers * 100.0) / n_pnts

        # Update overlay image and display
        if graphics:
            overlay = cv2.cvtColor(roi/2,cv2.COLOR_GRAY2RGB)
            OverlayRANSACFit(overlay, pnts, inlier_pnts, ellipse)
            cv2.imshow('RANSAC', overlay)
            cv2.waitKey(5)

        # Update best ellipse
        best_ellipse = ellipse

        if perc_inliers > max_perc_inliers:
            if DEBUG: print('Break Max Perc Inliers')
            break

    return best_ellipse
#%%
def createLineIterator(P1, P2, img):
    """
    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

    Parameters:
        -P1: a numpy array that consists of the coordinate of the first point (x,y)
        -P2: a numpy array that consists of the coordinate of the second point (x,y)
        -img: the image being processed

    Returns:
        -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])     
    """
   #define local variables for readability
    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

   #difference and absolute difference between points
   #used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

   #predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa,dXa),3),dtype=np.float32)
    itbuffer.fill(np.nan)

   #Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X: #vertical line segment
        itbuffer[:,0] = P1X
        if negY:
                itbuffer[:,1] = np.arange(P1Y - 1,P1Y - dYa - 1,-1)
        else:
                itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)              
    elif P1Y == P2Y: #horizontal line segment
        itbuffer[:,1] = P1Y
        if negX:
           itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
        else:
           itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
    else: #diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
           slope = dX/dY
           if negY:
               itbuffer[:,1] = np.arange(P1Y-1,P1Y-dYa-1,-1)
           else:
               itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
           itbuffer[:,0] = (slope*(itbuffer[:,1]-P1Y)).astype(np.int) + P1X
        else:
           slope = dY/dX
           if negX:
               itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
           else:
               itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
           itbuffer[:,1] = (slope*(itbuffer[:,0]-P1X)).astype(np.int) + P1Y

   #Remove points outside of image
    colX = itbuffer[:,0]
    colY = itbuffer[:,1]
    itbuffer = itbuffer[(colX >= 0) & (colY >=0) & (colX<imageW) & (colY<imageH)]

   #Get intensities from img ndarray
    itbuffer[:,2] = img[itbuffer[:,1].astype(np.uint),itbuffer[:,0].astype(np.uint)]

    return itbuffer