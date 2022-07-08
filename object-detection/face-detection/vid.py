import cv2
import sys
import numpy as np


# Get user supplied values
cap = cv2.VideoCapture(0)
cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

img = cv2.imread("temp.jpg", cv2.IMREAD_GRAYSCALE)
# scale_percent = 100 #percent of original size
# width = int(img.shape[1] * scale_percent / 100)
# height = int(img.shape[0] * scale_percent / 100)
# dim = (width, height)
# img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)


detector = cv2.xfeatures2d.SURF_create()
kp_image, desc_image = detector.detectAndCompute(img, None)
#img = cv2.drawKeypoints(img, kp_image, img)

index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

# flann = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


while True:
    _, frame =cap.read()

    gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fac = faceCascade.detectMultiScale(gframe, scaleFactor=1.1, minNeighbors=5,  minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)

    # Draw a rectangle around the faces
    # face = frame
    face = np.zeros((100,100,3))
    for (x, y, w, h) in fac:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face = frame[y:y+h,x:x+w]

    cv2.imshow("Faces found", frame)
    cv2.imshow("Faces extracted", face)

    kp_face, desc_face = detector.detectAndCompute(face , None)
    #frame = cv2.drawKeypoints(gframe, kp_gframe, gframe)
    matches = flann.knnMatch(desc_image, desc_face, k=2)
    # matches = flann.match(desc_image, desc_face)

    goods = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            goods.append(m)

    #imag = cv2.drawMatches(img, kp_image, gframe, kp_gframe, goods, gframe)
    if len(goods) > 10: print('temp found')
    else: print('not found')

    if cv2.waitKey(1) == ord('a'):
        break
cap.release()
cv2.destroyAllWindows()
