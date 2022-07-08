import cv2
from skimage.feature import local_binary_pattern

image = cv2.imread('fig1.jpg')
lbp_image=local_binary_pattern(image,8,2,method='uniform')
histogram=scipy.stats.itemfreq(lbp_image)
print histogram
