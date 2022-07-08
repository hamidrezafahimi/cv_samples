import cv2
from skimage.feature import greycomatrix, greycoprops
from skimage import data

from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb

radius = 2
no_points = 8
im = cv2.imread('fig1.jpg')
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
lbp = local_binary_pattern(im_gray, no_points, radius, method='default')
#(hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0,(no_points*(no_points-1)+4)))
#hist = hist.astype("float")
#hist /= (hist.sum() + eps)
#print hist.sum()
print lbp

