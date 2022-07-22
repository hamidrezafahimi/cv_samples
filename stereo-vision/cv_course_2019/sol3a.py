# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 11:46:16 2020

@author: Hamidreza
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

imgL = cv.imread('im0.png',0)
imgR = cv.imread('im1.png',0)

scale_percent = 20 
width = int(imgL.shape[1] * scale_percent / 100)
height = int(imgL.shape[0] * scale_percent / 100)
dim = (width, height)
imgL = cv.resize(imgL, dim, interpolation = cv.INTER_AREA) 
imgR = cv.resize(imgR, dim, interpolation = cv.INTER_AREA) 

stereo = cv.StereoBM_create(numDisparities=64, blockSize=15)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()



cv.imwrite('a.jpg', disparity)

cv.imshow('a',disparity)

cv.waitKey() 
cv.destroyAllWindows()
