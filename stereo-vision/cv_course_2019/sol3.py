import numpy as np
import cv2 as cv

### read the images and scale them down
imgL = cv.imread('im0a.png',0)  
imgR = cv.imread('im1a.png',0)    

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

### create the stereo matchers & compute disparity
left_mchr = cv.StereoBM_create(numDisparities=128, blockSize=15)
right_mchr = cv.ximgproc.createRightMatcher(left_mchr)
displ = left_mchr.compute(imgL, imgR)
dispr = right_mchr.compute(imgR, imgL) 
displ = np.int16(displ)
dispr = np.int16(dispr)

### create a postfiltering object & adjust it
lmbda = 80000
sigma = 1.2
visual_multiplier = 1.0
wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_mchr)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

### filter the image with postfilter made
filtered = wls_filter.filter(displ, imgL, None, dispr)  
filtered = cv.normalize(src=filtered, dst=filtered, beta=0, alpha=255, norm_type=cv.NORM_MINMAX);
filtered = np.uint8(filtered)

### output
cv.imshow('Disparity Map', filtered)
cv.imwrite('bb.jpg', filtered)


cv.waitKey()
cv.destroyAllWindows()


