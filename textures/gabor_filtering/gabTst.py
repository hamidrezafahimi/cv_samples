import sys
import numpy as np
import time
import cv2


vid = "/home/hamidreza/project_82/offline_work/vids/16.mp4"
cap = cv2.VideoCapture(vid)
initBB = None


erosion_type = cv2.MORPH_RECT
erosion_size1 = 1
erosion_size2 = 1
element1 = cv2.getStructuringElement(erosion_type, (2*erosion_size1 + 1, 2*erosion_size1+1), (erosion_size1, erosion_size1))
element2 = cv2.getStructuringElement(erosion_type, (2*erosion_size2 + 1, 2*erosion_size2+1), (erosion_size2, erosion_size2))
ksize = 11
sigma = 3
theta = 80/180.*np.pi
lamda = 6.2
gamma1 = 0
phi = 0
kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma1, phi)
kernels = []
thresh = 30

while True:
    res, frame = cap.read()
    # print(frame.size)
    uimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # uimg = uimg - uimg
    # for ker in kernels:
    #     uimg = uimg + cv2.filter2D(uimg, cv2.CV_8U, ker)
    uimg = cv2.filter2D(uimg, cv2.CV_8U, kernel)
    # uimg = cv2.erode(uimg, element1)
    # uimg = cv2.dilate(uimg, element2)
    # _,uimg = cv2.threshold(uimg,thresh,255,cv2.THRESH_BINARY)

    cv2.imshow('wl', uimg)
    cv2.imshow('w', frame)
    key = cv2.waitKey(50) & 0xFF
    if key == ord("q"):
        break
