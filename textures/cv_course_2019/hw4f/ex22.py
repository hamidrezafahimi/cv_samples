# GLCM feature extraction
import cv2
import numpy as np
from skimage import feature, io
from sklearn import preprocessing
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte

img = io.imread("fig1.jpg", as_grey=True)

S = preprocessing.MinMaxScaler((0,11)).fit_transform(img).astype(int)
Grauwertmatrix = feature.greycomatrix(S, [1,2], [0,np.pi/4], levels=12, symmetric=False, normed=True)

ContrastStats = feature.greycoprops(Grauwertmatrix, 'contrast')
CorrelationtStats = feature.greycoprops(Grauwertmatrix, 'correlation')
HomogeneityStats = feature.greycoprops(Grauwertmatrix, 'homogeneity')
ASMStats = feature.greycoprops(Grauwertmatrix, 'ASM')

io.imsave("test.jpg", S)
io.imshow(S)