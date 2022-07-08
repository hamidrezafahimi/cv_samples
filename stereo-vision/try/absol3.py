# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 13:45:10 2020

@author: Hamidreza
"""

import argparse 
import cv2 
import numpy as np 
from matplotlib import pyplot as plt 
i1 = cv2.imread('im0.png', 0) 
im1 = cv2.pyrDown(i1) 
 
#queryimage # left image 
img1 = cv2.pyrDown(im1) 
 
i2 = cv2.imread('im1.png', 0) 
im2 = cv2.pyrDown(i2) 
 
#trainimage # right image 
img2 = cv2.pyrDown(im2) 
 
# disparity range is tuned for 'aloe' image pair 
win_size = 1 
min_disp = 16 
max_disp = min_disp * 9 
num_disp = max_disp - min_disp # Needs to be divisible by 16 
stereo = cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities=num_disp, uniquenessRatio=10, speckleWindowSize=100, speckleRange=32, disp12MaxDiff=1, P1=8*3* win_size**2, P2=32 * 3 * win_size**2) 
 
disparity_map = stereo.compute(img1, img2).astype(np.float32) / 16.0 
print(disparity_map) 
h, w = img1.shape[:2] 
focal_length = 0.8 * w 
# Perspective transformation matrix 
Q = np.float32([[1, 0, 0, -w / 2.0], 
                 [0, -1, 0, h / 2.0], 
                 [0, 0, 0, -focal_length], 
                 [0, 0, 1, 0]]) 
points_3D = cv2.reprojectImageTo3D(disparity_map, Q) 
colors = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) 
mask_map = disparity_map > disparity_map.min() 
output_points = points_3D[mask_map] 
output_colors = colors[mask_map] 
out_fn = 'out.ply' 
#cv2.imshow('disparity_map', disparity_map) 
cv2.imshow('left', img1) 
cv2.imshow('disparity', (disparity_map - min_disp) / num_disp) 
#create_output(output_points, output_colors, output_file) 
cv2.waitKey() 
cv2.destroyAllWindows()
