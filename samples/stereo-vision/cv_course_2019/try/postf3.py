
import numpy as np
# from sklearn.preprocessing import normalize
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

left_matcher = cv.StereoBM_create(numDisparities=128, blockSize=15)

right_matcher = cv.ximgproc.createRightMatcher(left_matcher)

# FILTER Parameters
lmbda = 80000
sigma = 1.2
visual_multiplier = 1.0

wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

    displ = left_matcher.compute(imgL, imgR)#.astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

filteredImg = cv.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv.NORM_MINMAX);
filteredImg = np.uint8(filteredImg)
cv.imshow('Disparity Map', filteredImg)

cv.imwrite('bb.jpg', filteredImg)


cv.waitKey()
cv.destroyAllWindows()
