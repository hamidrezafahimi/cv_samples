import cv2
import sys
import numpy as np
import glob
import os

if len(sys.argv) < 2:
    print "error: no arguments"
    exit()

# Get user supplied values
cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

filenames = glob.glob("/home/hamidreza/cv/HW/face/ds/templates/*.jpg")
filenames.sort()
images = [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in filenames]

oframe = cv2.imread("/home/hamidreza/cv/HW/face/ds/" + str(sys.argv[1]))
frame = cv2.cvtColor(oframe, cv2.COLOR_BGR2GRAY)
# cv2.imshow("dd", frame)
# scale_percent = 100 #percent of original size
# width = int(img.shape[1] * scale_percent / 100)
# height = int(img.shape[0] * scale_percent / 100)
# dim = (width, height)
# img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
detector = cv2.xfeatures2d.SURF_create()
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

fac = faceCascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5,  minSize=(120, 120), flags = cv2.CASCADE_SCALE_IMAGE)

cnt1 = 0;
for img in images:

    kp_image, desc_image = detector.detectAndCompute(img, None)
    img_name = os.path.basename(filenames[cnt1])
    name = os.path.splitext(img_name)[0]
        # Draw a rectangle around the faces
        # face = frame
    face = np.zeros((10,10,3))
    cnt = 0;
    for (x, y, w, h) in fac:
        cnt = cnt + 1
        cv2.rectangle(oframe, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face = frame[y:y+h,x:x+w]
        num = "face " + str(cnt)
        cv2.imshow(num, face)
        kp_gframe, desc_gframe = detector.detectAndCompute(face , None)
        matches = flann.knnMatch(desc_image, desc_gframe, k=2)
        goods = []
        for m, n in matches:
            if m.distance < 0.7*n.distance:
                goods.append(m)

        #imag = cv2.drawMatches(img, kp_image, gframe, kp_gframe, goods, gframe)
        print len(goods)
        if len(goods) > 40:

            cv2.putText(oframe,name,(x,y),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)
        else: break

    cnt1 = cnt1 + 1

cv2.imshow("Faces found", oframe)


# matches = flann.match(desc_image, desc_gframe)



cv2.waitKey()

# cap.release()
cv2.destroyAllWindows()
