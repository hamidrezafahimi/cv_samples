import cv2
import numpy as np


img = cv2.imread("fig1.jpg", cv2.IMREAD_GRAYSCALE)
sift = cv2.xfeatures2d.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(img, None)
img = cv2.drawKeypoints(img, keypoints, None)

cv2.imshow("Image", sift)
cv2.waitKey()
cv2.destroyAllWindows()