# developed by- SUJOY KUMAR GOSWAMI
# source paper- https://people.ece.cornell.edu/acharya/papers/mlt_thr_img.pdf

import cv2
import numpy as np
import math

img = cv2.imread('/home/hamidreza/cv/HW/hw6/out3/aa.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
a = 0
b = 255
n = 6 # number of thresholds (better choose even value)
k = 0.7 # free variable to take any positive value
T = [] # list which will contain 'n' thresholds

def sujoy(img, a, b):
    if a>b:
        s=-1
        m=-1
        return m,s

    img = np.array(img)
    t1 = (img>=a)
    t2 = (img<=b)
    X = np.multiply(t1,t2)
    Y = np.multiply(img,X)
    s = np.sum(X)
    m = np.sum(Y)/s
    return m,s

for i in range(int(n/2-1)):
    img = np.array(img)
    t1 = (img>=a)
    t2 = (img<=b)
    X = np.multiply(t1,t2)
    Y = np.multiply(img,X)
    mu = np.sum(Y)/np.sum(X)

    Z = Y - mu
    Z = np.multiply(Z,X)
    W = np.multiply(Z,Z)
    sigma = math.sqrt(np.sum(W)/np.sum(X))

    T1 = mu - k*sigma
    T2 = mu + k*sigma

    x, y = sujoy(img, a, T1)
    w, z = sujoy(img, T2, b)

    T.append(x)
    T.append(w)

    a = T1+1
    b = T2-1
    k = k*(i+1)

T1 = mu
T2 = mu+1
x, y = sujoy(img, a, T1)
w, z = sujoy(img, T2, b)
T.append(x)
T.append(w)
T.sort()
print(T)
