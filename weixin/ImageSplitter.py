#!usr/bin/python
#coding=utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf8')
from PIL import Image, ImageChops
import cv2
import numpy as np
from glob import glob
from copy import deepcopy
import os
import shutil
import random

def cropImg(img, img_data, img_width, img_height):
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
	new_img = img.crop(split_box)
	# newImg = Image.new('RGB', size, color = (255, 255, 255))
	# leftTop = (size[0] / 2 - new_img.size[0] / 2, size[1] / 2 - new_img.size[1] / 2)
	# box = (leftTop[0], leftTop[1], leftTop[0] + new_img.size[0], leftTop[1] + new_img.size[1])
	# newImg.paste(new_img, box)
	return new_img

def cv2ToImage(data):
	return Image.fromarray(data)

def imageToCv2(data):
	return np.array(data)

def split_img(imageFile):
	cv2Img = cv2.imread(imageFile, 0)
	cv2Img = cv2.resize(cv2Img, (cv2Img.shape[1] * 5, cv2Img.shape[0] * 5), interpolation = cv2.INTER_CUBIC)
	# cv2Img = cv2.bilateralFilter(cv2Img, 15, 100, 100)
	cv2Img = cv2.threshold(cv2Img, 210, 255, cv2.THRESH_BINARY_INV)[1]
	newImg = ImageChops.invert(cv2ToImage(cv2Img))
	# cv2.imwrite('new.png', cv2Img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
	# newImg = cv2ToImage(cv2Img)
	newImg = cropImg(newImg, newImg.load(), newImg.size[0], newImg.size[1])
	inNewImg = ImageChops.invert(newImg)
	# newImg.save('new.png')
	width = newImg.size[0] / 4
	height = newImg.size[1]
	imgList = []
	for i in range(4):
		split_box = (i * width - 10, 0, (i + 1) * width + 10, height)
		tmpImg = inNewImg.crop(split_box)
		tmpImg = tmpImg.resize((128, 128))
		tmpImg = ImageChops.invert(tmpImg)
		tmpImg = tmpImg.convert('RGB')
		imgList.append(deepcopy(tmpImg))
	return imgList
		# tmpImg.save('%s.png'%i)

def main():
	filenames = glob('data/*.png')
	for fn in filenames:
		name = os.path.basename(fn).split('.')[0]
		imgList = split_img(fn)
		if len(imgList) == len(name) == 4:
			for i in range(len(imgList)):
				if not os.path.isdir('train/%s'%name[i]):
					os.makedirs('train/%s'%name[i])
				idx = len(os.listdir('train/%s'%name[i]))
				img = imgList[i]
				img.save('train/%s/%s.png'%(name[i], idx))

def split_test():
	test_size = 0.3
	classes = glob('train/*')
	for clsname in classes:
		if os.path.isdir(clsname):
			os.makedirs('validation/%s'%clsname.split('/')[-1])
			l = os.listdir(clsname)
			random.shuffle(l)
			num = len(l)
			tn = int(num * test_size)
			for i in l[0:tn]:
				shutil.copy2(os.path.join(clsname, i), 'validation/%s'%clsname.split('/')[-1])

if __name__ == '__main__':
	# ydm()
	# main()
	split_test()