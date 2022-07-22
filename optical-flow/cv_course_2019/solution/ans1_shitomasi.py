import numpy as np
import cv2
import glob
from skimage.io import imread_collection


##### read the collection
# col_dir = '/home/hamidreza/cv/HW/hw7/JPEGS/birds/*.jpg'
# col_dir = '/home/hamidreza/cv/HW/hw7/JPEGS/boats/*.jpg'
col_dir = '/home/hamidreza/cv/HW/hw7/JPEGS/bottle/*.jpg'
# col_dir = '/home/hamidreza/cv/HW/hw7/JPEGS/cyclists/*.jpg'
# col_dir = '/home/hamidreza/cv/HW/hw7/JPEGS/surf/*.jpg'
images = imread_collection(col_dir)


##### Parameters for Shi-Tomasi algorithm
maxCorners = 23
qualityLevel = 0.01
minDistance = 10
blockSize = 3
gradientSize = 3
useHarrisDetector = False
k = 0.04
p0 = cv2.goodFeaturesToTrack(images[1], maxCorners, qualityLevel, minDistance, None, \
        blockSize=blockSize, gradientSize=gradientSize, useHarrisDetector=useHarrisDetector, k=k)
p1 = cv2.goodFeaturesToTrack(images[2], maxCorners, qualityLevel, minDistance, None, \
        blockSize=blockSize, gradientSize=gradientSize, useHarrisDetector=useHarrisDetector, k=k)


##### calculate optical flow
mask = np.zeros_like(images[1])
j = 1
while(j<=len(images)):
    print(j)
    j = j+1
    frame = images[j]
    p1, st, err = cv2.calcOpticalFlowPyrLK(images[j-1], images[j], p0, None, winSize  = (15,15), maxLevel = 0, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    ### /// PART 1-2 ###
    dif = cv2.absdiff(frame, images[j-1])                           # find moving objects
    ret, dif = cv2.threshold(dif, 10, 255, cv2.THRESH_BINARY)
    backgraound = cv2.absdiff(frame, dif)                           # difference of moving and the image
    ### /// PART 1-2 ###


    ### /// PART 1-3 ###
    # p1, st, err = cv2.calcOpticalFlowPyrLK(images[j-1], images[j], p0, p1, None, winSize  = (75,75), maxLevel = 0, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 2, 0.01), flags = cv2.OPTFLOW_USE_INITIAL_FLOW)
    # p1, st, err = cv2.calcOpticalFlowPyrLK(images[j-1], images[j], p0, None, winSize  = (15,15), maxLevel = 0, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03), flags = cv2.OPTFLOW_LK_GET_MIN_EIGENVALS)
    ### /// PART 1-3 ###


##### draw the flow
    good_new = p1[st==1]
    good_old = p0[st==1]
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), (255,255,255), 2)
        frame = cv2.circle(frame,(a,b),5,(0,0,0),-1)
    img = cv2.add(frame,mask)


##### output
    ### optical flow
    img = cv2.resize(img, (int(img.shape[1] * 3),int(img.shape[0] * 3)), interpolation = cv2.INTER_AREA)
    cv2.imshow('frame',img)

    ### background
    backgraound = cv2.resize(backgraound, (int(img.shape[1]),int(img.shape[0])), interpolation = cv2.INTER_AREA)
    cv2.imshow('diff', backgraound)

    k = cv2.waitKey(500) & 0xff
    if k == 27:
        break

    old_gray = frame.copy()
    p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()
