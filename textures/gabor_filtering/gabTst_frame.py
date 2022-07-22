import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def nothing(x):
    pass

# img = cv2.imread(str(sys.argv[1]))

erosion_type = cv2.MORPH_RECT
erosion_size1 = 1
erosion_size2 = 1
element1 = cv2.getStructuringElement(erosion_type, (2*erosion_size1 + 1, 2*erosion_size1+1), (erosion_size1, erosion_size1))
element2 = cv2.getStructuringElement(erosion_type, (2*erosion_size2 + 1, 2*erosion_size2+1), (erosion_size2, erosion_size2))

vid = "/home/hamidreza/project_82/offline_work/vids/15.mp4"
cap = cv2.VideoCapture(vid)
cap.set(cv2.CAP_PROP_POS_FRAMES, 20)
# img = cv2.imread('out.jpg', 0)

res, img = cap.read()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


cv2.imshow('ooooo', img)

window_name = 'gabor'

ksize = 1.
sigma = 1.
theta = 1.
lamda = 1.
gamma = 0
phi = 0

g_ker = 7
g_sigma = 12

cv2.namedWindow(window_name,cv2.WINDOW_AUTOSIZE)

cv2.createTrackbar('kernel size', window_name, 13, 40, nothing)
cv2.createTrackbar('100*sigma', window_name, 300, 300, nothing)
cv2.createTrackbar('theta', window_name, 80, 180, nothing)
cv2.createTrackbar('lambda', window_name, 62, 100, nothing)
# cv2.createTrackbar('gamma', window_name, 0, 100, nothing)
# cv2.createTrackbar('phi', window_name, 0, 180, nothing)
# cv2.createTrackbar('(g_ker+1)/2', window_name, g_ker, 50, nothing)
# cv2.createTrackbar('10*g_sig', window_name, g_sigma, 90, nothing)
cv2.createTrackbar('er size', window_name, erosion_size1, 10, nothing)
cv2.createTrackbar('di size', window_name, erosion_size2, 10, nothing)
cv2.createTrackbar('th', window_name, 30, 255, nothing)

while(1):
    # g_sigma = cv2.getTrackbarPos('10*g_sig', window_name)/10.
    # g_ker = 2*(cv2.getTrackbarPos('(g_ker+1)/2', window_name))-1
    # img0 = cv2.GaussianBlur(img,(g_ker,g_ker),g_sigma)

    ksize = cv2.getTrackbarPos('kernel size', window_name)
    sigma = cv2.getTrackbarPos('100*sigma', window_name)/100.
    theta = cv2.getTrackbarPos('theta', window_name)
    theta = theta / 180. * np.pi
    lamda = cv2.getTrackbarPos('lambda', window_name)/10.
    # gamma = cv2.getTrackbarPos('gamma', window_name)/10.
    # phi = cv2.getTrackbarPos('phi', window_name)
    phi = phi * np.pi/180.
    erosion_size1 = cv2.getTrackbarPos('er size', window_name)
    erosion_size2 = cv2.getTrackbarPos('di size', window_name)
    thresh = cv2.getTrackbarPos('th', window_name)
    element1 = cv2.getStructuringElement(erosion_type, (2*erosion_size1 + 1, 2*erosion_size1+1), (erosion_size1, erosion_size1))
    element2 = cv2.getStructuringElement(erosion_type, (2*erosion_size2 + 1, 2*erosion_size2+1), (erosion_size2, erosion_size2))



    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi)
    fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)

    fimg = cv2.erode(fimg, element1)
    fimg = cv2.dilate(fimg, element2)
    if thresh!=0:
        _,fimg = cv2.threshold(fimg,thresh,255,cv2.THRESH_BINARY)

    cv2.imshow(window_name, fimg)
    # cv2.imshow('window_name', img)
    cv2.imshow('window', kernel)
    k = cv2.waitKey(1)
    if k == ord('a'):
        break


cv2.destroyAllWindows()
