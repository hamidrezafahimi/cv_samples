import numpy as np
import cv2 as cv

### read the images and scale them down
imgL = cv.imread('/home/hamidreza/cv/HW/hw6/im0a.png',0)
imgR = cv.imread('/home/hamidreza/cv/HW/hw6/im1a.png',0)

scale_percent = 20
width = int(imgL.shape[1] * scale_percent / 100)
height = int(imgL.shape[0] * scale_percent / 100)
dim = (width, height)
imgL = cv.resize(imgL, dim, interpolation = cv.INTER_AREA)
imgR = cv.resize(imgR, dim, interpolation = cv.INTER_AREA)


#imgL = cv.pyrDown(imgL)
#imgL = cv.pyrDown(imgL)
#imgR = cv.pyrDown(imgR)
#imgR = cv.pyrDown(imgR)

### create the stereo matchers, create a postfiltering object & compute disparity
left_mchr = cv.StereoBM_create(numDisparities=160, blockSize=15)
right_mchr = cv.ximgproc.createRightMatcher(left_mchr)
wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_mchr)
displ = left_mchr.compute(imgL, imgR)
dispr = right_mchr.compute(imgR, imgL)
displ = np.int16(displ)
dispr = np.int16(dispr)

### adjust postfiltering
lmbda = 80000
sigma = 1.2
visual_multiplier = 1.0
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

### filter the image with postfilter made
filtered = wls_filter.filter(displ, imgL, None, dispr)
filtered = cv.normalize(src=filtered, dst=filtered, beta=0, alpha=255, norm_type=cv.NORM_MINMAX);
filtered = np.uint8(filtered)

### output for disparity map
cv.imshow('Disparity Map', filtered)
cv.imwrite('bb.jpg', filtered)
cv.waitKey()
cv.destroyAllWindows()


### segmentation
filteredc = cv.Canny(filtered, 10, 500);

laplacian = cv.Laplacian(filtered,cv.CV_8U)
ret,thresh = cv.threshold(laplacian,5,255,cv.THRESH_BINARY)


sobelx = cv.Sobel(filtered,cv.CV_8U,1,0,ksize=3)
sobely = cv.Sobel(filtered,cv.CV_8U,0,1,ksize=3)
sobelim = sobelx + sobely

### output for segmentation
cv.imshow('canny segmentation', filteredc)
cv.imshow('laplacian segmentation', laplacian)
cv.imshow('laplacian segmentation thesholded', thresh)
cv.imshow('sobel segmentation', sobelim)

cv.imwrite('canny.jpg', filteredc)
cv.imwrite('laplacian.jpg', laplacian)
cv.imwrite('sobel.jpg', sobelim)
cv.imwrite('thesholded.jpg', thresh)


cv.waitKey()
cv.destroyAllWindows()
