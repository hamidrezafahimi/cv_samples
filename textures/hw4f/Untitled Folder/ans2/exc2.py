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

ksize = 1.
sigma = 1.
theta = 1.
lamda = 1.
gamma = 1.
phi = 1.

cv2.namedWindow(window_name,cv2.WINDOW_AUTOSIZE)

cv2.createTrackbar('kernel size', window_name, 5, 40, nothing)
cv2.createTrackbar('sigma', window_name, 300, 300, nothing)
cv2.createTrackbar('theta', window_name, 90, 180, nothing)
cv2.createTrackbar('lambda', window_name, 40, 100, nothing)
cv2.createTrackbar('gamma', window_name, 0, 100, nothing)
cv2.createTrackbar('phi', window_name, 0, 180, nothing)

while(1):
    ksize = cv2.getTrackbarPos('kernel size', window_name)
    sigma = cv2.getTrackbarPos('sigma', window_name)/100.
    theta = cv2.getTrackbarPos('theta', window_name)
    theta = theta / 180. * np.pi
    lamda = cv2.getTrackbarPos('lambda', window_name)/10.
    gamma = cv2.getTrackbarPos('gamma', window_name)/10.
    phi = cv2.getTrackbarPos('phi', window_name)
    phi = phi * np.pi/180.
    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi)
    fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
    cv2.imshow(window_name, fimg)
    cv2.imshow('window', kernel)
    k = cv2.waitKey(1)
    if k == ord('a'):
        break


cv2.destroyAllWindows()
