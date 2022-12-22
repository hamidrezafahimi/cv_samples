import cv2
import numpy as np

i = 0

img = cv2.imread("temp.jpg", cv2.IMREAD_GRAYSCALE)
# scale_percent = 100 #percent of original size
# width = int(img.shape[1] * scale_percent / 100)
# height = int(img.shape[0] * scale_percent / 100)
# dim = (width, height)
# img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("vid1.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 70)

detector1 = cv2.xfeatures2d.SIFT_create()
kp_image1, desc_image1 = detector1.detectAndCompute(img, None)
detector = cv2.xfeatures2d.FREAK_create()
kp_image,desc_image = detector.compute(img,kp_image1)
#img = cv2.drawKeypoints(img, kp_image, img)

index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

desc_image = np.float32(desc_image)

while True:
    i = i+1
    res, frame = cap.read()
    gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kp_gframe1, desc_gframe1 = detector1.detectAndCompute(gframe , None)
    kp_gframe, desc_gframe = detector.compute(gframe , kp_gframe1)
    #frame = cv2.drawKeypoints(gframe, kp_gframe, gframe)

    desc_gframe = np.float32(desc_gframe)
    matches = flann.knnMatch(desc_image, desc_gframe, k=2)

    goods = []
    for m, n in matches:
        if m.distance < 0.9*n.distance:
            goods.append(m)

    #imag = cv2.drawMatches(img, kp_image, gframe, kp_gframe, goods, gframe)
    if len(goods) > 7:
        q = np.float32([kp_image[m.queryIdx].pt for m in goods]).reshape(-1, 1, 2)
        t = np.float32([kp_gframe[m.trainIdx].pt for m in goods]).reshape(-1, 1, 2)

        matrix, mask = cv2.findHomography(q, t, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        h, w = img.shape
        p = np.float32([[0,0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(p, matrix)

        homo = cv2.polylines(gframe, [np.int32(dst)], True, (255, 0, 0), 3)

        cv2.imshow("Homography", homo)
    else:
        #continue
         #cv2.imshow("Homography", gframe)
        print(i)

   # cv2.imshow("image", img)
    #cv2.imshow("frame", frame)
   # cv2.imshow("imag", imag)

    if cv2.waitKey(1) == ord('a'):
        break

cap.release()
cv2.destroyAllWindows()
