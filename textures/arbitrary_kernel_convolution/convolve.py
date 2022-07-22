import cv2
import numpy as np
kernel = cv2.imread('temp.jpg',cv2.CV_32F)
kernel = cv2.normalize(kernel, kernel, 0, 1, norm_type=cv2.NORM_MINMAX)

# kernel = np.array([[0, 0, 1, 1],
#                     [0, 0, 1, 1],
#                     [0, 0, 1, 1],
#                     [0, 0, 1, 1]], dtype=np.float32)
frame = cv2.imread('temp2.jpg',cv2.CV_32F)

vid = "/home/hamidreza/project_82/videos/openWindow/15_vid2.mp4"
# img = cv2.imread("temp.jpg", cv2.IMREAD_GRAYSCALE)
cap = cv2.VideoCapture(vid)
# cap.set(cv2.CAP_PROP_POS_FRAMES, 270)

while True:
    # res, frame = cap.read()
    # gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fimg = cv2.filter2D(frame, -1, kernel)
    result = cv2.normalize(fimg, fimg, 0, 255, norm_type=cv2.NORM_MINMAX)

    cv2.imshow('kernel',kernel)
    cv2.imshow("Homography", result)

    if cv2.waitKey(1) == ord('a'):
        break

cap.release()
cv2.destroyAllWindows()
