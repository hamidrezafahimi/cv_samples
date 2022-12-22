
import cv2 
import numpy as np 
i1 = cv2.imread('im0.png',0) 
im1 = cv2.pyrDown(i1) 
 
#queryimage # left image 
img1 = cv2.pyrDown(im1) 
 
i2 = cv2.imread('im1.png',0) 
im2 = cv2.pyrDown(i2) 
 
#trainimage # right image 
img2 = cv2.pyrDown(im2) 
# SIFT alghorithm 
 
sift = cv2.xfeatures2d.SIFT_create() 
 
# find the keypoints and descriptors with SIFT 
kp1, des1 = sift.detectAndCompute(img1, None) 
kp2, des2 = sift.detectAndCompute(img2, None) 
#img1 = cv2.drawKeypoints(img1, kp1, img1) 
#img2 = cv2.drawKeypoints(img2, kp2, img2) 
 
# FLANN parameters 
FLANN_INDEX_KDTREE = 0 
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5) 
search_params = dict(checks=50) 
flann = cv2.FlannBasedMatcher(index_params, search_params) 
matches = flann.knnMatch(des1, des2, k=2) 
 
good_points = [] 
pts1 = [] 
pts2 = [] 
# ratio test to retain only the good matches 
for i, (m, n) in enumerate(matches): 
    if m.distance < 0.7 * n.distance: 
       pts1.append(kp1[m.queryIdx].pt) 
       pts2.append(kp2[m.trainIdx].pt) 
 
pts1 = np.float32(pts1) 
pts2 = np.float32(pts2) 
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS) 
# Selecting only the inliers 
pts_left_image = pts1[mask.ravel() == 1] 
pts_right_image = pts2[mask.ravel() == 1] 
def draw_lines(img1, img2, lines, pts1, pts2): 
    h, w = img1.shape 
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) 
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR) 
 
    for line, pt_left, pt_right in zip(lines, pts1, pts2): 
        x_start, y_start = map(int, [0, -line[2] / line[1]]) 
        x_end, y_end = map(int, [w, -(line[2] + line[0] * w) / line[1]]) 
        color = tuple(np.random.randint(0, 255, 2).tolist()) 
        cv2.line(img1, (x_start, y_start), (x_end, y_end), color, 1) 
 
        cv2.circle(img1, tuple(pt_left), 5, color, -1) 
        cv2.circle(img2, tuple(pt_right), 5, color, -1) 
    return img1, img2 
 
# Drawing the lines on left image and the corresponding feature points on the right image 
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F) 
lines1 = lines1.reshape(-1, 3) 
img_left_lines, img_right_pts = draw_lines(img1, img2, lines1, pts1, pts2) 
# Drawing the lines on right image and the corresponding feature points on the left image 
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F) 
lines2 = lines2.reshape(-1, 3) 
img_right_lines, img_left_pts = draw_lines(img2, img1, lines2, pts2, pts1) 
 
cv2.imshow('Epi lines on left image', img_left_lines) 
cv2.imshow('Feature points on right image', img_right_pts) 
cv2.imshow('Epi lines on right image', img_right_lines) 
cv2.imshow('Feature points on left image', img_left_pts) 
cv2.waitKey() 
cv2.destroyAllWindows()
