#!usr/bin/python
#coding=utf-8

import os
import urllib
import json
from PIL import Image
from io import StringIO
from glob import glob
import random

def downloadImg():
	base_url = r'https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result&queryWord=%E9%A3%8E%E6%99%AF%E5%A3%81%E7%BA%B8&cl=&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=-1&z=&ic=&hd=&latest=&copyright=&word=%E9%A3%8E%E6%99%AF%E5%A3%81%E7%BA%B8&s=&se=&tab=&width=1280&height=720&face=0&istype=2&qc=&nc=&fr=&expermode=&force=&cg=wallpaper&pn={pn}&rn=30&gsm=&1576739274838='
	pn = 0
	count = 0
	for page in xrange(10):
		content = urllib.request.urlopen(base_url.format(pn = pn)).read()
		data = json.loads(content)
		for i in data['data']:
			if 'middleURL' not in i: continue
			imgUrl = i['middleURL']
			img = Image.open(StringIO(urllib.request.urlopen(imgUrl).read()))
			img = img.convert('RGB')
			img.save('source/%s.jpg'%count)
			count += 1
		print(page, count)
		pn = pn + 30

def getMask(path, count):
	rate = random.uniform(0.3, 0.7)
	sizeRange = (40, 60)
	img = Image.open(path)
	# img = img.resize((100, 60))
	img.save('bg/%s'%(os.path.basename(path)))
	img = img.convert('RGBA')
	w, h = img.size
	maskSize = (random.randint(sizeRange[0], sizeRange[1]), random.randint(sizeRange[0], sizeRange[1]))
	mask_white = Image.new('RGBA', maskSize, color = (255, 255, 255))
	mask_black = Image.new('RGBA', maskSize, color = (0, 0, 0))
	x, y = random.randint(10, w - maskSize[0] - 10), random.randint(10, h - maskSize[1] - 10)
	maskBox = (x, y, x + maskSize[0], y + maskSize[1])
	crop = img.crop(maskBox)
	crop = Image.blend(crop, random.choice([mask_white, mask_black]), rate)
	# crop.save('crop.png')
	img.paste(crop, maskBox)
	img = img.convert('RGB')
	fn = 'data/mask_%s.jpg'%count
	# fn = os.path.abspath(fn)
	img.save(fn)
	return maskBox, fn

def makeVec():
	cmd = 'opencv_createsamples -vec ./train_set.vec -info ./info.txt -bg ./bg.txt -num 150 -w 100 -h 60'
	os.system(cmd)

def train():
	OUTPUT_FILE='./output'
	VEC_FILE='./train_set.vec'
	BG_FILE='./bg.txt'
	NUM_POS=20
	NUM_NEG=100
	NUM_STAGE=10   
	VAL_BUFSIZE=1024
	IDX_BUFSIZE=1024

	STAGE_TYPE='BOOST'
	FEATURE_TYPE='HAAR'
	# FEATURE_TYPE=LBP
	WEIGHT=177
	HEIGHT=100

	BT='GAB'
	MIN_HITRATE=0.995
	MAX_FALSE_ALARM_RATE=0.05
	WEIGHT_TRIM_RATE=0.95
	MAX_DEPTH=1
	MAX_WEAK_COUNT=100

	MODE='BASIC'


	cmd = 'opencv_traincascade -data {OUTPUT_FILE} -vec {VEC_FILE} -bg {BG_FILE} -numPos {NUM_POS} -numNeg {NUM_NEG} -numStages {NUM_STAGE} -precalcValBufSize {VAL_BUFSIZE} -precalcIdxBufSize {IDX_BUFSIZE} -stageType {STAGE_TYPE} -featureType {FEATURE_TYPE} -w {WEIGHT} -h {HEIGHT} -bt {BT} -minHitRate {MIN_HITRATE} -maxFalseAlarmRate {MAX_FALSE_ALARM_RATE} -weightTrimRate {WEIGHT_TRIM_RATE} -maxDepth {MAX_DEPTH} -maxWeakCount {MAX_WEAK_COUNT} -mode {MODE}'.format(
	    OUTPUT_FILE = OUTPUT_FILE, VEC_FILE = VEC_FILE, BG_FILE = BG_FILE, NUM_POS = NUM_POS, NUM_NEG = NUM_NEG, NUM_STAGE = NUM_STAGE, VAL_BUFSIZE = VAL_BUFSIZE, IDX_BUFSIZE = IDX_BUFSIZE, STAGE_TYPE = STAGE_TYPE, FEATURE_TYPE = FEATURE_TYPE, WEIGHT = WEIGHT, HEIGHT = HEIGHT, BT = BT, MIN_HITRATE = MIN_HITRATE, MAX_FALSE_ALARM_RATE = MAX_FALSE_ALARM_RATE, WEIGHT_TRIM_RATE = WEIGHT_TRIM_RATE, MAX_DEPTH = MAX_DEPTH, MAX_WEAK_COUNT = MAX_WEAK_COUNT, MODE = MODE)
	print(cmd)
	os.system(cmd)

def test(path):
	import cv2
	face_cascade = cv2.CascadeClassifier('./output/cascade.xml')
	img = cv2.imread(path)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray)
	for (x, y, w, h) in faces:
		# if w > 50 and h > 50 and w < 65 and y < 65:
		print(x, y, w, h)
		cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

	cv2.imwrite('test_result.jpg', img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

if __name__ == '__main__':
	# downloadImg()
	# splitImg('1.png')
	imageList = glob('source/*.jpg')
	imageList.sort()
	f = open('info.txt', 'w')
	b = open('bg.txt', 'w')
	for k, v in enumerate(imageList):
		box, fn = getMask(v, k)
		f.write('%s 1 %s %s %s %s\r\n'%(fn, box[0] - 5, box[1] - 5, box[2] - box[0] + 5, box[3] - box[1] + 5))
		b.write('bg/%s'%(os.path.basename(v)) + '\r\n')
		f.flush()
		b.flush()
	f.close()
	b.close()