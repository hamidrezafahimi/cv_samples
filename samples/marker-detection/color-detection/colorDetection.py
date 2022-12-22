#!/usr/bin/env python

import cv2 as cv
import pandas as pd
import math
import argparse
import yaml
import numpy as np
import time
from enum import Enum


# p = '../params/params.yaml'
p1 = '../params/params_1.yaml'
p2 = '../params/params_2.yaml'
p3 = '../params/params_3.yaml'
p4 = '../params/params_4.yaml'
# P = [p, p1, p2, p3, p4]
params = [p1, p2, p3, p4]

parser = argparse.ArgumentParser()
parser.add_argument("file")
args = parser.parse_args()



class Modes(Enum):
	SET = 1
	GET = 2
	DONE = 3



class ColorThreshold:

	def __init__(self):

		self._paramsFiles = params
		self._windowCreated = False
		self._presetUI = "Set Color Thresholds"
		self._key = None
		self._delay = 30
		self._mustKeepOn = False
		self._hsv_lists = []

		for k, p in enumerate(params):
			with open(p) as f:
				data = yaml.load(f, Loader = yaml.FullLoader)
				self._hsv_lists.append([int(data["lowH"]), int(data["lowS"]),
					int(data["lowV"]), int(data["highH"]), int(data["highS"]),
					int(data["highV"])])
				if k == 0:
					self._lowH = int(data["lowH"])
					self._lowS = int(data["lowS"])
					self._lowV = int(data["lowV"])
					self._highH = int(data["highH"])
					self._highS = int(data["highS"])
					self._highV = int(data["highV"])

		# print("ColorThreshold GUIDE:" + \
		# 	"\n \t q: quit" + \
		# 	"\n \t w: pause" + \
		# 	"\n \t e: play" + \
		# 	"\n \t r: save params")


	def lowHChanged(self, value):
		# print('lh ---')
		# print('val : ----',value)
		self._lowH = int(min(value, self._highH-1))
		# self.onSlidersChange()

	def lowSChanged(self, value):
		# print('ls ---')
		# print('val : ----',value)
		self._lowS = int(min(value, self._highS-1))
		# self.onSlidersChange()

	def lowVChanged(self, value):
		# print('lv ---')
		# print('val : ----',value)
		self._lowV = int(min(value, self._highS-1))
		# self.onSlidersChange()

	def highHChanged(self, value):
		# print('hh ---')
		# print('val : ----',value)
		self._highH = int(max(value, self._lowH+1))
		# self.onSlidersChange()

	def highSChanged(self, value):
		# print('hs ---')
		# print('val : ----',value)
		self._highS = int(max(value, self._lowS+1))
		# self.onSlidersChange()

	def highVChanged(self, value):
		# print('lv ---')
		# print('val : ----',value)
		self._highV = int(max(value, self._lowV+1))
		# self.onSlidersChange()


	def saveParams(self, idx=0):

		with open(self._paramsFiles[idx], "r") as f:
			dict = yaml.load(f, Loader=yaml.FullLoader)

		dict["lowH"] = self._lowH
		dict["lowS"] = self._lowS
		dict["lowV"] = self._lowV
		dict["highH"] = self._highH
		dict["highS"] = self._highS
		dict["highV"] = self._highV

		with open(self._paramsFiles[idx], "w") as f:
			yaml.dump(dict, f)


	def visualize(self, image):

		if not self._windowCreated:
			cv.namedWindow(self._presetUI, cv.WINDOW_GUI_NORMAL)
			cv.resizeWindow(self._presetUI, 1200, 400)
			cv.createTrackbar("Low H", self._presetUI, self._lowH, 180,
				self.lowHChanged)
			cv.createTrackbar("High H", self._presetUI, self._highH, 180,
				self.highHChanged)
			cv.createTrackbar("Low S", self._presetUI, self._lowS, 255,
				self.lowSChanged)
			cv.createTrackbar("High S", self._presetUI, self._highS, 255,
				self.highSChanged)
			cv.createTrackbar("Low V", self._presetUI, self._lowV, 255,
				self.lowVChanged)
			cv.createTrackbar("High V", self._presetUI, self._highV, 255,
				self.highVChanged)
			self._windowCreated = True

		image = cv.resize(image, (600, 400))
		cv.imshow(self._presetUI, image)

		self._key = cv.waitKey(self._delay)

		if not self._mustKeepOn:
			self._keepOn = False

		if self._key == ord('r'):
			self.saveParams()
			self.destroyWindow(self._presetUI)
			# self._keepOn = False

		elif self._key == ord('w'):
			self._mustKeepOn = True

		elif self._key == ord('e'):
			self._mustKeepOn = False
			# self._keepOn = False

		elif self._key == ord('q'):
			self._keepOn = False

		elif self._key == ord('1'):
			self.saveParams(1)

		elif self._key == ord('2'):
			self.saveParams(2)

		elif self._key == ord('3'):
			self.saveParams(3)

		elif self._key == ord('4'):
			self.saveParams(4)


	def filter(self, img, mode):

		if mode == Modes.SET:
			imageHSV = cv.cvtColor(image, cv.COLOR_BGR2HSV)
			h,s,v = cv.split(imageHSV)
			v = cv.equalizeHist(v)
			imageHSV = cv.merge([h, s, v])
			# print('lh',self._lowH)
			# print('ls',self._lowS)
			# print('lv',self._lowV )
			# print('hh',self._highH )
			# print('hs',self._highS )
			# print('hv',self._highV)
			lh = self._lowH
			ls = self._lowS
			lv = self._lowV
			hh = self._highH
			hs = self._highS
			hv = self._highV
			thresholdedImage = cv.inRange(imageHSV, (lh, ls, lv), (hh, hs, hv))

			return thresholdedImage, None

		elif mode == Modes.GET:
			# To do: The class is incomplete; Tasks:
			# 	- 	Save 4 different filter values for 4 different corner marker colors
			# 	- 	Do the thresholding using pre-loaded 4 different parameter sets
			# 		within the ROI and determine the pixel "coordinates" for each marker
			return None, coordinates


	def process(self, image, mode):

		if mode == Modes.DONE:
			return None

		self._keepOn = True
		while self._keepOn:
			filtered, nums = self.filter(image, mode)
			self.visualize(filtered)
			image = cv.resize(image, (600, 400))
			if mode == Modes.GET:
				return nums
