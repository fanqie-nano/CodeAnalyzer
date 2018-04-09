#!usr/bin/python
#coding=utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import numpy as np
import os
import cv2
import copy
# from operator import mul
# import pytesseract

from PIL import Image, ImageDraw, ImageChops

from Rotated import rotated

def getPeriphery(imgData, imgWidth, imgHeight, x, y, size = 1):
	x1 = x
	if x > size:
		x1 = x - size
	y1 = y
	if y > size:
		y1 = y - size
	x2 = min(x + size, imgWidth)
	y2 = min(y + size, imgHeight)
	ax = y2 + 1 - y1
	if y2 == imgHeight:
		ax = ax - 1
	ay = x2 + 1 - x1
	if x2 == imgWidth:
		ay = ay - 1
	area = np.zeros((ax, ay))
	# if (x1, y1) == (105, 64):
	# 	print area
	# 	print x2, y2
	for ix in xrange(x1, x2 + 1):
		if ix == imgWidth:continue
		for iy in xrange(y1, y2 + 1):
			if iy == imgHeight:continue
			area[iy - y1, ix - x1] = imgData[ix, iy]
			# if (x1, y1) == (105, 64):
			# 	print ix, iy
			# 	print imgData[ix, iy]
	return area, (x1, y1)

def getBlackRate(data):
	return float(np.sum(data == 255)) / reduce(lambda x, y: x * y, data.shape)

# def isLinkWay(data):
# 	for i, v in enumerate(data):
# 		r = getBlackRate(v)
# 		if r == 0:
# 			return False, i
# 	return True, -1

def isLinkWay(data):
	isBreak = 0
	for i, v in enumerate(data):
		r = getBlackRate(v)
		if isBreak == 0 and r > 0:
			isBreak = 1
		elif isBreak == 1 and r == 0:
			isBreak = 2
		elif isBreak == 2 and r > 0:
			isBreak = 3
	if isBreak == 3:
		return False
	else:
		if getBlackRate(data[0]) > 0:
			return True
	return False
'''
top = 0
right = 1
bottom = 2
left = 3
'''
def findWay(retDict, imgData, imgWidth, imgHeight, x, y, block = 5, step = 8):
	nextList = []
	walkList = [(x, y)]
	while True:
		area, leftTop = getPeriphery(imgData, imgWidth, imgHeight, x, y, block)
		# top = area[0:block]
		r = isLinkWay(area)
		pos = (x, y - step)
		if r and pos[1] > 0 and pos not in walkList and pos not in nextList:
			nextList.append(pos)
		# bottom = area[block:area.shape[0]]
		r = isLinkWay(area[::-1])
		pos = (x, y + step)
		if r and pos[1] < imgHeight and pos not in walkList and pos not in nextList:
			nextList.append(pos)
		# left = area.T[0:block]
		r = isLinkWay(area.T)
		pos = (x - step, y)
		if r and pos[0] > 0 and pos not in walkList and pos not in nextList:
			nextList.append(pos)
		# right = area.T[block:area.shape[0]]
		r = isLinkWay(area.T[::-1])
		pos = (x + step, y)
		if r and pos[0] < imgWidth and pos not in walkList and pos not in nextList:
			nextList.append(pos)
		retDict['%s,%s'%leftTop] = copy.deepcopy(area)
		if len(nextList) > 0:
			x, y = nextList.pop(0)
			walkList.append((x, y))
		else:
			break

def drawImg(imgDraw, retDict, color):
	for c in retDict:
		ltx, lty = int(c.split(',')[0]), int(c.split(',')[1])
		for y in range(len(retDict[c])):
			for x in range(len(retDict[c][y])):
				clr = retDict[c][y][x]
				if clr == 255:
					imgDraw.point((ltx + x, lty + y), color)

def cropImg(img, img_data, img_width, img_height, size = (600, 600)):
	split_box = [0, 0, 0, 0]
	for x in range(img_width):
		all_is_white = True
		for y in range(img_height):
			# if img_data[x, y] != 255:
			# 	print img_data[x, y]
			if img_data[x, y] == 0:
				all_is_white = False
				split_box[0] = x
				break
		if not all_is_white:
			break
	for y in range(img_height):
		all_is_white = True
		for x in range(img_width):
			if img_data[x, y] == 0:
				all_is_white = False
				split_box[1] = y
				break
		if not all_is_white:
			break
	for x in range(img_width - 1, -1, -1):
		all_is_white = True
		for y in range(img_height):
			if img_data[x, y] == 0:
				all_is_white = False
				split_box[2] = x
				break
		if not all_is_white:
			break
	for y in range(img_height - 1, -1, -1):
		all_is_white = True
		for x in range(img_width):
			if img_data[x, y] == 0:
				all_is_white = False
				split_box[3] = y
				break
		if not all_is_white:
			break
	newImg = Image.new('RGB', size, color = (255, 255, 255))
	new_img = img.crop(split_box)
	leftTop = (size[0] / 2 - new_img.size[0] / 2, size[1] / 2 - new_img.size[1] / 2)
	box = (leftTop[0], leftTop[1], leftTop[0] + new_img.size[0], leftTop[1] + new_img.size[1])
	newImg.paste(new_img, box)
	return newImg

def cv2ToImage(data):
	return Image.fromarray(data)

def imageToCv2(data):
	return np.array(data)

def erosion(cv2Image, kernel = (10, 10)):
	kernel = np.ones(kernel, np.uint8)
	return cv2.erode(cv2Image, kernel, 0)

def dilate(cv2Image, kernel = (10, 10)):
	kernel = np.ones(kernel, np.uint8)
	return cv2.dilate(cv2Image, kernel, 1)

def splitImage(imageFile, codeCount = 4, outPath = None):
	imagePieces = []
	cv2Img = cv2.imread(imageFile, 0)
	cv2Img = cv2.resize(cv2Img, (cv2Img.shape[1] * 5, cv2Img.shape[0] * 5), interpolation = cv2.INTER_CUBIC)
	cv2Img = cv2.bilateralFilter(cv2Img, 15, 100, 100)
	# cv2.imwrite('new.png', cv2Img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
	cv2Img = cv2.threshold(cv2Img, 150, 255, cv2.THRESH_BINARY_INV)[1]
	cv2Img = cv2.morphologyEx(cv2Img, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
	cv2Img = cv2.threshold(cv2Img, 150, 255, cv2.THRESH_BINARY)[1]
	# cv2.imwrite('new1.png', cv2Img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
	# cv2.imwrite('new.png', cv2.resize(cv2Img, (cv2Img.shape[1] * 6, cv2Img.shape[0] * 2), interpolation = cv2.INTER_CUBIC), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

	img = cv2ToImage(cv2Img)
	# img.save('test0.png')
	imgDraw = ImageDraw.Draw(img)
	imgWidth, imgHeight = img.size
	# imgHeight, imgWidth = cv2Img.shape
	imgData = img.load()

	pngNum = 1
	while True:
		retDict = {}
		findPoint = False
		for x in xrange(imgWidth):
			for y in xrange(imgHeight):
				if imgData[x, y] == 255:
					findWay(retDict, imgData, imgWidth, imgHeight, x, y, block = 4, step = 8)
					findPoint = True
					break
			if findPoint:
				break
		if not findPoint:
			break
		# print len(retDict)
		if len(retDict) < 20:
			drawImg(imgDraw, retDict, 0)
			continue
		# 	continue
		# else:
		newImg = Image.new('L', img.size, color = 0)
		newImgDraw = ImageDraw.Draw(newImg)
		drawImg(newImgDraw, retDict, 255)
		drawImg(imgDraw, retDict, 0)
		# newImg.save('%s.png'%pngNum)
		# newImg = newImg.resize((newImg.size[0] / 6, newImg.size[1] / 2), Image.ANTIALIAS)
		newImg = newImg.resize((newImg.size[0] * 5, newImg.size[1] * 5), Image.ANTIALIAS)

		# cv2Img = erosion(imageToCv2(newImg), kernel = (5, 5))
		# cv2Img = dilate(cv2Img, kernel = (5, 5))
		# cv2Img = cv2.morphologyEx(imageToCv2(newImg), cv2.MORPH_CLOSE, np.ones((6, 6), np.uint8))
		# cv2Img = imageToCv2(newImg)
		cv2Img = cv2.threshold(imageToCv2(newImg), 90, 255, cv2.THRESH_BINARY)[1]
		# cv2Img = erosion(cv2Img, kernel = (10, 10))
		# cv2Img = dilate(cv2Img, kernel = (15, 15))
		# cv2.imwrite('ttt.png', cv2Img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
		cv2Img = erosion(cv2Img)
		newImg = ImageChops.invert(cv2ToImage(cv2Img))
		# newImg = cv2ToImage(cv2Img)
		newImg = cropImg(newImg, newImg.load(), newImg.size[0], newImg.size[1])
		newImg = newImg.convert('RGB')
		cv2Img = rotated(newImg)
		# cv2Img = erosion(cv2Img, kernel = (5, 5))
		cv2Img = cv2.resize(cv2Img, (100, 100), interpolation = cv2.INTER_AREA)
		if outPath and os.path.isdir(outPath):
			outFile = os.path.join(outPath, '%s.png'%pngNum)
			pngNum = pngNum + 1
			cv2.imwrite(outFile, cv2Img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
		imagePieces.append(cv2ToImage(cv2Img))
	if len(imagePieces) == codeCount:
		return imagePieces
	return []
	# cv2Image = dilate(cv2Img)
	# img = cv2ToImage(cv2Image)
	# img.save('tmp.png')

# if __name__ == '__main__':
# 	start('source/21856.png', './')

# def exceptionHandler(signum, frame):
# 	raise AssertionError

# if __name__ == '__main__':
# 	# rightList = open('run.log', 'r').readlines()
# 	# rightList = [i.strip().split('/')[1].split('.')[0] for i in rightList]
# 	import shutil
# 	# import signal
# 	# path = 'source'
# 	# error = open('error.log', 'w')
# 	# for i in os.listdir(path):
# 	# 	if int(i.split('.')[0]) >= 100:
# 	# 		imageFile = os.path.join(path, i)
# 	# 		cvImg = cv2.imread(imageFile)
# 	# 		if cvImg is None:
# 	# 			os.remove(imageFile)
# 	# 			continue
# 	# 		if os.path.splitext(i)[0] in rightList:
# 	# 			continue
# 	# 		outPath = os.path.join('result', os.path.splitext(i)[0])
# 	# 		if os.path.isdir(outPath):
# 	# 			shutil.rmtree(outPath)
# 	# 		os.makedirs(outPath)
# 	# 		try:
# 	# 			signal.signal(signal.SIGALRM, exceptionHandler)
# 	# 			signal.alarm(15)
# 	# 			pngNum = start(imageFile, outPath)
# 	# 			signal.alarm(0)
# 	# 			if pngNum != 4:
# 	# 				error.write(imageFile + '\r\n')
# 	# 				error.flush()
# 	# 			print imageFile, pngNum
# 	# 		except:
# 	# 			error.write(imageFile + '\r\n')
# 	# 			error.flush()
# 	# error.close()
# 		# break
# 	for i in os.listdir('result'):
# 		if not i.startswith('.'):
# 		# if int(i.split('.')[0]) >= 100:
# 			p = 'result/%s/source.png'%(i.split('.')[0])
# 			print 'source/%s.png'%(i), p
# 			shutil.copy('source/%s.png'%(i), p)

# def main(filePath):
# 	import shutil
# 	import EVA2Slim
# 	tmp = 'tmp'
# 	if os.path.isdir(tmp):
# 		shutil.rmtree(tmp)
# 	os.makedirs(tmp)
# 	start(filePath, tmp)
# 	codeList = os.listdir(tmp)
# 	result = ''
# 	if len(codeList) == 4:
# 		codeList = [os.path.join(tmp, i) for i in codeList]
# 		for num in range(1, len(codeList) + 1):
# 			result = result + EVA2Slim.predict(os.path.join(tmp, '%s.png'%num))
# 	return result

# if __name__ == '__main__':
# 	# print main('ret-meug.png')
# 	# start('ret-meug.png', 'tmp')
# 	import random
# 	allCount = 0
# 	right = 0
# 	path = 'test'
# 	fileList = os.listdir(path)
# 	random.shuffle(fileList)
# 	fileList = fileList[0:200]
# 	for i in fileList:
# 		allCount = allCount + 1
# 		result = i.split('.')[0].split('ret-')[-1]
# 		print result
# 		ret = main(os.path.join(path, i))
# 		print ret
# 		if ret == result:
# 			right = right + 1
# 		print allCount, right, float(right) / allCount