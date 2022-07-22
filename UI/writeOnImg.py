
import cv2 as cv

image = cv.imread('2/8.png')
FPS = 27
dt = 0.037
cv.putText(image,"Calculation Time: {:.03f} ms   FPS: {:.0f}".format(dt, FPS) ,(20,20), cv.FONT_HERSHEY_PLAIN, 1.4, [255,255,255], 2)
cv.imwrite('filename.png', image)
