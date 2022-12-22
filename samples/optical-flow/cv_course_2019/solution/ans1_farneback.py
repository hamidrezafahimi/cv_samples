import cv2
import argparse
import sys
import numpy as np
from skimage.io import imread_collection

##### read the collection
col_dir = '/home/hamidreza/cv/HW/hw7/JPEGS/birds/*.jpg'
# col_dir = '/home/hamidreza/cv/HW/hw7/JPEGS/boats/*.jpg'
# col_dir = '/home/hamidreza/cv/HW/hw7/JPEGS/bottle/*.jpg'
# col_dir = '/home/hamidreza/cv/HW/hw7/JPEGS/cyclists/*.jpg'
# col_dir = '/home/hamidreza/cv/HW/hw7/JPEGS/surf/*.jpg'
images = imread_collection(col_dir)


##### function to draw the dence flow
def draw_flow(img, flow, step=20):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


##### calculate optical flow
prevgray = images[1]
prevgray = cv2.resize(prevgray, (int(prevgray.shape[1] * 3),int(prevgray.shape[0] * 3)), interpolation = cv2.INTER_AREA)
j = 1
while(1):
    j = j+1
    gray = images[j]
    gray = cv2.resize(gray, (int(gray.shape[1] * 3),int(gray.shape[0] * 3)), interpolation = cv2.INTER_AREA)
    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    prevgray = gray

    ### /// PART 1-2 ###
    dif = cv2.absdiff(images[j], images[j-1])                           # find moving objects
    ret, dif = cv2.threshold(dif, 10, 255, cv2.THRESH_BINARY)
    backgraound = cv2.absdiff(images[j], dif)                           # difference of moving and the image
    ### /// PART 1-2 ###


##### output
    ### optical flow
    cv2.imshow("Dense Optic Flow", draw_flow(gray, flow))
    ### background
    backgraound = cv2.resize(backgraound, (int(gray.shape[1]),int(gray.shape[0])), interpolation = cv2.INTER_AREA)
    cv2.imshow("backgraound", draw_flow(backgraound, flow))
    ### moving parts
    dif = cv2.resize(dif, (int(gray.shape[1]),int(gray.shape[0])), interpolation = cv2.INTER_AREA)
    cv2.imshow("moving parts", draw_flow(dif, flow))

    key = cv2.waitKey(100)
    if (key == ord('x')):
        break

cv2.destroyAllWindows()
