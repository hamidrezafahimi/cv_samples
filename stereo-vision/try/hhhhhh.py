
import numpy as np
from sklearn.preprocessing import normalize
import cv2

imgL = cv2.imread('im0a.png',0)  
imgR = cv2.imread('im1a.png',0)

scale_percent = 20 
width = int(imgL.shape[1] * scale_percent / 100)
height = int(imgL.shape[0] * scale_percent / 100)
dim = (width, height)
imgL = cv2.resize(imgL, dim, interpolation = cv2.INTER_AREA) 
imgR = cv2.resize(imgR, dim, interpolation = cv2.INTER_AREA) 



left_matcher = cv2.StereoBM_create(numDisparities=128, blockSize=15)

right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

# FILTER Parameters
lmbda = 80000
sigma = 1.2
visual_multiplier = 1.0
 
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

print('computing disparity...')
displ = left_matcher.compute(imgL, imgR)#.astype(np.float32)/16
dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
displ = np.int16(displ)
dispr = np.int16(dispr)
filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
filteredImg = np.uint8(filteredImg)
cv2.imshow('Disparity Map', filteredImg)

cv2.imwrite('bb.jpg', filteredImg)


cv2.waitKey()
cv2.destroyAllWindows()