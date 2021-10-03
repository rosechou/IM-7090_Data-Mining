# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 20:59:27 2020

@author: user
"""
import glob
import cv2    
from sklearn import decomposition    
import matplotlib.pyplot as plt 
import numpy as np   

for path in glob.glob("fruit-recognition/Pitaya/test/*.png"):
    
    img = cv2.imread(path,0)  
    vmax = max(img.max(), -img.min())
    
    #fig, (ax, ax2)  =plt.subplots(ncols=2)    
    #ax.imshow(img, cmap=plt.cm.gray, interpolation = 'nearest',vmin=-vmax,vmax=vmax)
    
    n_components = 20
    
    estimator = decomposition.NMF(n_components = n_components, init = 'random', tol=5e-3)    
    W = estimator.fit_transform(img)
    H = estimator.components_
    
    new_img = np.dot(W,H)
    plt.imshow(new_img, cmap=plt.cm.gray,
                       interpolation='nearest',
                       vmin=-vmax, vmax=vmax)
    plt.axis('off') 
    
    #plt.show()
    plt.savefig(path)
    plt.clf()