import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def nothing(x):
    pass


img = cv2.imread('fig3.jpg', 0)
cv2.imshow('an', img)
# img = np.float32(inn)
# img = cv2.convertTo(inn, cv2.CV_32F)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# pi = 3.141592

window_name = 'dome asb'

ksize = 5
sigma = [2.97, 2.98, 2.99, 3]
theta = [25/180.*np.pi, 65/180.*np.pi, 115/180.*np.pi, 155/180.*np.pi]
lamda = 4
gamma = 0
phi = 0

cv2.namedWindow(window_name,cv2.WINDOW_AUTOSIZE)

i = 0
fimg = np.zeros(img.shape)

# cv2.imshow('w', fimg)



while(1):
    kernel = cv2.getGaborKernel((ksize, ksize), sigma[i], theta[i], lamda, gamma, phi)
    fimg = fimg + cv2.filter2D(img, cv2.CV_8UC3, kernel)
    if i == 3:
        break
    i = i + 1

cv2.imshow('w', fimg)
cv2.imwrite('out.jpg', fimg)

cv2.waitKey(0);
