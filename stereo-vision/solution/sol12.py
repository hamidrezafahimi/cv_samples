import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

### read the images
img1 = cv.imread('/home/hamidreza/cv/HW/hw6/im0s.png',0)    # left image  :queryimage
img2 = cv.imread('/home/hamidreza/cv/HW/hw6/im1s.png',0)    # right image :trainimage

### resize the images to desired scale percent
scale_percent = 20
width = int(img1.shape[1] * scale_percent / 100)
height = int(img1.shape[0] * scale_percent / 100)
dim = (width, height)
img1 = cv.resize(img1, dim, interpolation = cv.INTER_AREA)
img2 = cv.resize(img2, dim, interpolation = cv.INTER_AREA)


### create detector & find keypoints and descriptors of two images
sift = cv.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

### create matcher & match the points between two images
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

### choose the points with the matches inside a threshold
good = []                                               # good point indexes
p1 = []                                               # good points of query image (left)
p2 = []                                               # good points of train image (right)
for m,n in matches:                                     # ratio test::  m, n : counters in matches elements
    if m.distance < 0.8*n.distance:
        good.append(m)
        p1.append(kp1[m.queryIdx].pt)
        p2.append(kp2[m.trainIdx].pt)

### find fundamental matrix according to the two matched point sets
p1 = np.int32(p1)
p2 = np.int32(p2)
F, mask = cv.findFundamentalMat(p1,p2,cv.FM_LMEDS)
p1 = p1[mask.ravel()==1]                            # select the inliers
p2 = p2[mask.ravel()==1]                            # select the inliers

### knowing the fundamental matrix, finding the epilines
lines1 = cv.computeCorrespondEpilines(p2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
lines2 = cv.computeCorrespondEpilines(p1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)

### this function takes the epilines of im_l which correspond to matched points in im_p, then draws them
def drawlines(im_line,im_point,lines,p_line,p_point):
    x,y = img1.shape
    im_line = cv.cvtColor(im_line,cv.COLOR_GRAY2BGR)
    im_point = cv.cvtColor(im_point,cv.COLOR_GRAY2BGR)
    for l,pl,pp in zip(lines,p_line,p_point):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -l[2]/l[1] ])
        x1,y1 = map(int, [y, -(l[2]+l[0]*y)/l[1] ])
        im_line = cv.line(im_line, (x0,y0), (x1,y1), color,1)
        im_line = cv.circle(im_line,tuple(pl),2,(0,0,0),-1)
        im_point = cv.circle(im_point,tuple(pp),2,(255,255,255),-1)
    return im_line,im_point

### now call the defined function and draw keypoints and epilines
img1_lines,img2_points = drawlines(img1,img2,lines1,p1,p2)
img2_lines,img1_points = drawlines(img2,img1,lines2,p2,p1)

### the output
print(F)
cv.imshow('points in left',img1_points)
cv.imshow('points in right',img2_points)
cv.imshow('lines in left',img1_lines)
cv.imshow('lines in right',img2_lines)

cv.imwrite('img1_points.jpg',img1_points)
cv.imwrite('img2_points.jpg',img2_points)
cv.imwrite('img1_lines.jpg',img1_lines)
cv.imwrite('img2_lines.jpg',img2_lines)

cv.waitKey()
cv.destroyAllWindows()
