#!usr/bin/python
#coding=utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import os
import shutil
import cv2
import math
import heapq
import pandas as pd
import numpy as np
from PIL import Image
from glob import glob
from train import classes
# from collections import Counter

def getBgPos(img):
	posList = [
		(0, 0),
		(0, img.shape[1] - 1),
		(img.shape[0] - 1, 0),
		(img.shape[0] - 1, img.shape[1] - 1)
	]
	colorList = []
	for pos in posList:
		colorList.append(img[pos[0]][pos[1]])
	posColor = zip(posList, colorList)
	return sorted(posColor, key = lambda x: x[1])[-1]

def split(filename, outdir = 'data'):
	code = filename.split('/')[-1].split('.')[0]
	# origin_img = cv2.imread('source/103.png')
	origin_img = cv2.imread(filename)
	img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)
	copyImg = img.copy()
	h, w = img.shape[:2]
	mask = np.zeros([h+2, w+2], np.uint8)
	bgPos, color = getBgPos(copyImg)
	cv2.floodFill(copyImg, mask, (bgPos[1], bgPos[0]), 255, 50, 50, cv2.FLOODFILL_FIXED_RANGE)

	for x in xrange(copyImg.shape[0]):
		for y in xrange(copyImg.shape[1]):
			if copyImg[x][y] < 255 and abs(int(copyImg[x][y]) - color) > 50:
				mask[:] = 0
				cv2.floodFill(copyImg, mask, (y, x), 0, 10, 10, cv2.FLOODFILL_FIXED_RANGE)
			else:
				mask[:] = 0
				cv2.floodFill(copyImg, mask, (y, x), 255, 10, 10, cv2.FLOODFILL_FIXED_RANGE)

	newImg = np.zeros([h, w, 3], np.uint8)
	newImg[:] = 255
	for y in xrange(h):
		for x in xrange(w):
			if copyImg[y][x] != 255:
				newImg[y][x] = origin_img[y][x]
				mask[:] = 0
				tmp = copyImg.copy()
				cv2.floodFill(tmp, mask, (x, y), 128, 10, 10, cv2.FLOODFILL_FIXED_RANGE)
				if tmp[tmp==128].shape[0] < 30:
					newImg[np.where(tmp==128)] = 255
					tmp[np.where(tmp==128)] = 255
					copyImg = tmp
	cv2.imwrite('test1.jpg', newImg)
	gray_img = cv2.cvtColor(newImg, cv2.COLOR_BGR2GRAY)
	cv2Img = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY)[1]
	# swipWindow(newImg, cv2Img)

	verticalImg = np.zeros([h, w], np.uint8)
	verticalImg[:] = 255
	# area = []
	blackArray = []
	# blackWhiteDict = {}
	for x in xrange(w):
		black = 0
		for y in xrange(h):
			if cv2Img[y][x] == 0:
				black += 1

		for i in xrange(black):
			verticalImg[h-i-1][x] = 0
		blackArray.append(black)
	cv2.imwrite('test3.jpg', verticalImg)
	blackArray = pd.Series(blackArray)
	wn = int(len(blackArray)/4)
	wave_base = heapq.nsmallest(wn, enumerate(blackArray), key=lambda x: x[1])
	area = [i[0] for i in wave_base]
	area.extend([0, w - 1])
	area.sort()
	area = zip(area[0:-1], area[1::])
	# print area
	# count = 0
	retList = []
	for x1, x2 in area:
		sub_w = x2 - x1
		crop = cv2Img[:, x1:x2]
		blackRate = np.sum(crop==0) / float(crop.shape[0] * crop.shape[1])
		# stdValue = np.std(crop)
		# print sub_w, blackRate
		if sub_w > 5 and blackRate > 0.1:
			# count += 1
			retList.append((sub_w, x1, x2))
	# while True:
	# print len(retList)
	if len(retList) < 6:
		retList = sorted(retList, key = lambda x: x[0], reverse = True)
		for maxW in retList:
			xw, x1, x2 = maxW
			if xw > 20:
				retList.remove(maxW)
				imgGrayCrop = cv2Img[:, x1:x2]
				imgCrop = newImg[:, x1:x2]
				splitRet, win_size = swipWindow(imgCrop, imgGrayCrop, 2)
				for i in xrange(len(splitRet)):
					# if i == 0:
					# 	new1 = (win_size + 3, x1 + splitRet[i][0], x1 + splitRet[i][0] + win_size + 3)
					# else:
					# 	new1 = (win_size + 3, x1 + splitRet[i][0] - 3, x1 + splitRet[i][0] + win_size)
					new1 = (win_size, x1 + splitRet[i][0], x1 + splitRet[i][0] + win_size)
					retList.append(new1)
			# else:
			# 	break
					# retList.append(new2)
				# _, x1, x2 = maxW
				# mid = (maxW[1] + maxW[2]) / 2
				# new1, new2 = (mid - x1, x1, mid), (x2 - mid, mid, x2)
				# retList.remove(maxW)
				# retList.append(new1)
				# retList.append(new2)
	# else:
	# 	break
	if len(retList) > 6:
		stdList = []
		for i in xrange(len(retList)):
			_, x1, x2 = retList[i]
			crop = newImg[:, x1:x2]
			stdList.append((np.std(crop), retList[i]))
		stdList = sorted(stdList, key = lambda x: x[0])
		num = len(stdList) - 6
		for i in xrange(num):
			d = stdList[i][1]
			retList.remove(d)
		# else:
		# 	break
	retList = sorted(retList, key = lambda x: x[1])
	numList = []
	for i in xrange(len(retList)):
		_, x1, x2 = retList[i]
		if x2 - x1 < 15:
			x1 = max(x1 - 5, 0)
			x2 = min(x2 + 5, w)
		if x2 - x1 <= 44:
			crop = newImg[:, x1:x2]
			fn = '%s/%s_%s.jpg'%(outdir, code, i)
			saveImg = np.zeros((44, 44, 3), np.uint8)
			saveImg[:] = 255
			x = (44 - x2 + x1) / 2
			saveImg[0:44, x:x + crop.shape[1]] = crop
			numList.append(fn)
			# saveImg.save(fn)
			cv2.imwrite(fn, saveImg)
	return numList

def swipWindow(img, img_gray, num = -1):
	win_size = 20
	winList = []
	h, w = img_gray.shape[0:2]
	if num <= 0:
		num = int(math.ceil(w/20.0))
	else:
		win_size = int(math.ceil(w / float(num)))
	for i in xrange(0, w, win_size):
		crop = img_gray[:, i:i+win_size]
		stdValue = np.std(img[:, i:i+win_size])
		winList.append((i, stdValue, np.sum(crop==0)))
	winList = sorted(winList, key = lambda x: x[1] * x[2], reverse = True)
	# print winList
	winList = winList[0:num]
	return winList, win_size
	# for i in xrange(len(winList)):
	# 	crop = img[:, winList[i][0]:winList[i][0]+win_size]
	# 	fn = 'tmp/%s.jpg'%(i)
	# 	cv2.imwrite(fn, crop)

def makeTrainData():
	countDict = {}
	fileList = glob('data/*.jpg')
	fileList.sort()
	for fn in fileList:
		code, idx = fn.split('/')[-1].split('.')[0].split('_')
		idx = int(idx)
		lab = code[idx]
		countDict.setdefault(lab, 0)
		if not os.path.isdir('train/%s'%lab):
			os.makedirs('train/%s'%lab)
		# print countDict[lab]
		shutil.copy2(fn, 'train/%s/%s_%s.jpg'%(lab, countDict[lab], code))
		countDict[lab] = countDict[lab] + 1

def predict(model, code):
	from keras.preprocessing.image import load_img, img_to_array
	from keras.applications.vgg16 import preprocess_input
	img = load_img(code, target_size = (44, 44))
	x = img_to_array(img)
	x = np.expand_dims(x, axis = 0)
	x = preprocess_input(x)
	# im = cv2.imread('result/10000/h-4.png')
	# im = cv2.resize(im, (100, 100), interpolation = cv2.INTER_CUBIC)
	# im = im[np.newaxis, :, :, :]
	out = model.predict_classes(x)
	return classes[out[0]]

def test():
	from keras.models import load_model
	count = 0
	right = 0
	model = load_model('code_model.h5')
	fileList = glob('code/*.jpg')
	for fn in fileList:
		# print fn
		count = count + 1
		code = ''
		rightCode = fn.split('/')[-1].split('.')[0]
		nl = split(fn)
		for c in nl:
			code = code + predict(model, c)
		if code == rightCode:
			right = right + 1
		print fn, code, rightCode, float(right) / float(count)



if __name__ == '__main__':
	test()
	# split('error/1A6D12.jpg', 'tmp')
	# split('error/Y7425M.jpg', 'tmp')
	# split('code/1DSV28.jpg', 'tmp')

	# fileList = glob('code/*.jpg')
	# fileList.sort()
	# total = 0.0
	# right = 0.0
	# for fn in fileList:
	# 	total += 1.0
	# 	try:
	# 		nl = split(fn)
	# 	except:
	# 		nl = []
	# 	if len(nl) == 6:
	# 		right += 1.0
	# 		# print nl
	# 	else:
	# 		print fn, len(nl)
	# 		for i in nl:
	# 			os.remove(i)
	# print fn, total, right, right / total
	# makeTrainData()
	# rate = 0.3
	# import random
	# for i in os.listdir('train'):
	# 	if not i.startswith('.'):
	# 		label = 'train/%s'%i
	# 		fileList = glob(label + '/*.jpg')
	# 		random.shuffle(fileList)
	# 		num = int(rate * len(fileList))
	# 		for fn in fileList[0:num]:
	# 			shutil.copy2(fn, 'val/%s/%s'%(i, fn.split('/')[-1]))
