import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb

METHOD = 'uniform'

# settings for LBP
radius = 1
n_points = 8 * radius

image1 = cv2.imread('fig1.jpg')
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
lbp1 = local_binary_pattern(image1, n_points, radius, METHOD)

image2 = cv2.imread('fig2.jpg')
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
lbp2 = local_binary_pattern(image2, n_points, radius, METHOD)

image3 = cv2.imread('fig3.jpg')
image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)
lbp3 = local_binary_pattern(image3, n_points, radius, METHOD)

cv2.imshow('lbp1', lbp1)
cv2.imshow('lbp2', lbp2)
cv2.imshow('lbp3', lbp3)


cv2.waitKey(0)
