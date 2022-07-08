# GLCM feature extraction
import cv2
import numpy as np
from skimage import feature, io
from sklearn import preprocessing
from skimage.color import rgb2gray

img = cv2.imread("fig1.jpg", 0)
S = preprocessing.MinMaxScaler((0,11)).fit_transform(img).astype(int)
#S = cv2.cvtColor(S, cv2.COLOR_BGR2GRAY)
S = rgb2gray(S)
io.imshow(S)


io.imsave("test.jpg", S)