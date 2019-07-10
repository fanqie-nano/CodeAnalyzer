# !usr/bin/python
# coding=utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from PIL import Image, ImageDraw
import cv2
import copy
import numpy as np
import selectivesearch

class ImageInfo(object):

	def __init__(self, x = 0, y = 0, w = 0, h = 0, img = None, sim = 0):
		self.x = x
		self.y = y
		self.w = w
		self.h = h
		self.img = img
		self.sim = sim

	def __eq__(self, obj):
		if self.x == obj.x and self.y == obj.y:
			return True
		return False

def cv2ToImage(data):
	return Image.fromarray(data)

def imageToCv2(data):
	return np.array(data)

def erosion(img, kernel = (2, 2)):
	cv2Image = imageToCv2(img)
	kernel = np.ones(kernel, np.uint8)
	cv2Image = cv2.erode(cv2Image, kernel, 0)
	return cv2ToImage(cv2Image)

def dilate(img, kernel = (2, 2)):
	cv2Image = imageToCv2(img)
	kernel = np.ones(kernel, np.uint8)
	cv2Image = cv2.dilate(cv2Image, kernel, 1)
	return cv2ToImage(cv2Image)

def difference(hist1, hist2):
	sum1 = 0
	for i in range(len(hist1)):
		if (hist1[i] == hist2[i]):
			sum1 += 1
		else:
			sum1 += 1 - float(abs(hist1[i] - hist2[i]))/ max(hist1[i], hist2[i])
	return sum1 / len(hist1)

def similary_calculate(img1, img2):
	img1 = img1.resize((30, 30)).convert('1')
	img2 = img2.resize((30, 30)).convert('1')
	hist1 = list(img1.getdata())
	hist2 = list(img2.getdata())
	diff = difference(hist1, hist2)
	return diff

def isInBox(x, y, box, win_siz):
	a, b, c, d = box
	if x >= a and x <= c and y >= b and y <= d:
		return True
	if x + win_siz >= a and x + win_siz <= c and y >= b and y <= d:
		return True
	if x >= a and x <= c and y + win_siz >= b and y + win_siz <= d:
		return True
	if x + win_siz >= a and x + win_siz <= c and y + win_siz >= b and y + win_siz <= d:
		return True
	return False

def getWriteRate(img):
	count = 0
	w, h = img.size
	data = img.load()
	for x in xrange(w):
		for y in xrange(h):
			if data[x, y] > 250:
				count += 1
	return float(count) / (w * h)

def splitSource(sourcePath):
	source = Image.open(sourcePath).convert('L')
	w, h = source.size
	template = source.crop((0, 0, w, h - 80))
	target = source.crop((0, h - 80, 200, h))
	return template, target

def wordFilter(img):
	newImg = Image.new('L', img.size, color = 0)
	imgDraw = ImageDraw.Draw(newImg)
	w, h = img.size
	data = img.load()
	for x in xrange(w):
		for y in xrange(h):
			if data[x, y] <= 10:
				imgDraw.point((x, y), 255)
	# newImg = erosion(newImg)
	# newImg = dilate(newImg)
	newImg = dilate(newImg)
	return newImg

def splitWord(img, target_size):
	cv2Img = imageToCv2(img)
	cv2Img = cv2.cvtColor(cv2Img, cv2.COLOR_GRAY2BGR)
	img_lbl, regions = selectivesearch.selective_search(cv2Img, scale = 500, sigma = 0.9, min_size = 30)

	candidates = []
	for r in regions:
		# 重复的不要
		if r['rect'] in candidates:
			continue
		# 太小和太大的不要
		if r['rect'][2] * r['rect'][3] < 400 or r['rect'][2] * r['rect'][3] > 6400:
			continue

		x, y, w, h = r['rect']
		if w > 0 and h > 0:
			# 太不方的不要
			if w / float(h) > 1.5 or h / float(w) > 1.5:
				continue
			candidates.append(r)

	candidates = [i['rect'] for i in candidates]
	candidates = sorted(candidates, key = lambda x: x[2] * x[3], reverse = True)
	candidates_sec = []
	for x, y, w, h in candidates:
		isAdd = 1
		for a, b, c, d in candidates_sec:
			if x >= a and x <= a + c and y >= b and y <= b + d and w <= c and h <= d:
				isAdd = 0
		if isAdd == 1:
			candidates_sec.append((x, y, w, h))
	ret = []
	for x, y, w, h in candidates_sec:
		tmp = img.crop((x, y, x + w, y + h))
		size = max(tmp.size)
		newImg = Image.new('L', (size, size), color = 0)
		leftTop = (size / 2 - tmp.size[0] / 2, size / 2 - tmp.size[1] / 2)
		box = (leftTop[0], leftTop[1], leftTop[0] + tmp.size[0], leftTop[1] + tmp.size[1])
		newImg.paste(tmp, box)
		newImg = newImg.resize((target_size, target_size))
		# newImg = dilate(newImg)
		ret.append(ImageInfo(x, y, w, h, newImg))
	ret = sorted(ret, key = lambda x: x.x)
	return ret

def main(sourcePath):
	template, target = splitSource(sourcePath)
	template = wordFilter(template)
	target = wordFilter(target)
	template.save('template.jpg')
	target.save('target.jpg')

	templateList = splitWord(template, 50)
	# for k, i in enumerate(templateList):
	# 	i.img.save('tmp/template_%s.jpg'%k)
	targetList = splitWord(target, 50)
	# for k, i in enumerate(targetList):
	# 	i.img.save('tmp/target_%s.jpg'%k)


	retList = [None, None, None, None]
	for k, v in enumerate(targetList):
		similary = []
		for a, b in enumerate(templateList):
			sim = similary_calculate(v.img, b.img)
			b = copy.deepcopy(b)
			b.sim = sim
			similary.append(b)
		similary = sorted(similary, key = lambda x: x.sim, reverse = True)
		retList[k] = copy.deepcopy(similary)

	while True:
		isBreak = True
		for kv, val in enumerate(retList):
			for k, i in enumerate(retList):
				if kv != k and i[0] == val[0] and i[0].sim <= val[0].sim:
					isBreak = False
					i.pop(0)
		if isBreak:
			break

	count = 0
	for i in retList:
		if i is not None:
			count += 1
			i[0].img.save('tmp/%s.jpg'%count)

if __name__ == '__main__':
	main('cap_union_new_getcapbysig.jpeg')