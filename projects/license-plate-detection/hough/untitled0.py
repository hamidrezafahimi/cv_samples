
import cv2
import numpy as np

i = 0


#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("_01_20180203_174219.h264")
cap.set(cv2.CAP_PROP_POS_FRAMES, 800000)

while True:
    i = i+1
    res, frame = cap.read()
    
    cv2.imshow("Homography", frame)

    
    if cv2.waitKey(1) == ord('a'):
        break
    
cap.release()
cv2.destroyAllWindows()