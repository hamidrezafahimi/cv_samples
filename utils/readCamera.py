import cv2

cam = cv2.VideoCapture(0)

while True:
    check, frame = cam.read()