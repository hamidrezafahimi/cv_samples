

import glob
import cv2

images = [cv2.imread(file) for file in glob.glob("/home/hamidreza/cv/HW/hw7/JPEGS/birds/*.jpg")]

print ("Number of items in the list = ", len(images))
