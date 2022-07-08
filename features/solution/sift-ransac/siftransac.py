import cv2
import numpy as np

i = 0
vid = "/home/hamidreza/ocv/hw960/vid1.mp4"
img = cv2.imread("temp.jpg", cv2.IMREAD_GRAYSCALE)
cap = cv2.VideoCapture(vid)
cap.set(cv2.CAP_PROP_POS_FRAMES, 70)

detector = cv2.xfeatures2d.SIFT_create()
kp_image, desc_image = detector.detectAndCompute(img, None)

index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

while True:
    i = i+1
    res, frame = cap.read()
    gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kp_gframe, desc_gframe = detector.detectAndCompute(gframe , None)
    matches = flann.knnMatch(desc_image, desc_gframe, k=2)

    goods = []
    for m, n in matches:
        if m.distance < 0.6*n.distance:
            goods.append(m)

    if len(goods) > 10:
        q = np.float32([kp_image[m.queryIdx].pt for m in goods]).reshape(-1, 1, 2)
        t = np.float32([kp_gframe[m.trainIdx].pt for m in goods]).reshape(-1, 1, 2)

        matrix, mask = cv2.findHomography(q, t, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        h, w = img.shape
        p = np.float32([[0,0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(p, matrix)

        homo = cv2.polylines(gframe, [np.int32(dst)], True, (255, 0, 0), 3)

        cv2.imshow("Homography", homo)
        # cv2.imwrite("frame%d.jpg" %i, homo)
    else:
        cv2.imshow("Homography", gframe)


    if cv2.waitKey(1) == ord('a'):
        break

cap.release()
cv2.destroyAllWindows()
