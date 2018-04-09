#!usr/bin/python
#coding=utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import time
import urllib2
import os
import random
import shutil
import re

##### 1

# def downloadSource():
# 	url = 'http://s.weibo.com/ajax/pincode/pin?type=sass%s'

# 	for i in range(40425, 100000):
# 		content = urllib2.urlopen(url%int(time.time() * 10000)).read()
# 		fn = os.path.join('source', '%s.png'%i)
# 		f = open(fn, 'wb')
# 		f.write(content)
# 		f.close()
# 		print fn
# 		time.sleep(random.random())
# downloadSource()
##### 2

def ydm():
	from YDMHttp import getIdentifyingCode
	pIntPng = re.compile(r'\d.png')

	path = 'result'

	done = open('result.txt', 'r').readlines()
	done = set([i.strip().split(' ---> ')[0].strip() for i in done])
	print done
	f = open('result.txt', 'a+')
	fileList = os.listdir(path)
	for i in fileList:
		# if int(i) >= 100:
		if i.startswith('.'):
			continue
		d = os.path.join(path, i, 'source.png')
		if d in done:
			continue
		s = os.listdir(os.path.join(path, i))
		s = set(filter(pIntPng.match, s))
		if not set(['1.png', '2.png', '3.png', '4.png']) == s:
			shutil.rmtree(os.path.join(path, i))
			continue
		ret = getIdentifyingCode(d)
		nd = os.path.join(path, i, 'ret-%s.png'%ret)
		shutil.copy(d, nd)
		f.write('%s ---> %s\r\n'%(d, nd))
		f.flush()
		print d, '--->', nd
	f.close()
	# break

##### 3

def rename():
	path = 'result'

	for p in os.listdir(path):
		if p.startswith('.'):
			continue
		d = os.path.join(path, p)
		for i in os.listdir(d):
			if i.startswith('ret-'):
				code = list(i.split('-')[-1].split('.')[0])
				if len(code) == 4:
					code
					break
				else:
					print d, 'error'
		if len(os.listdir(d)) != 6:
			print d
		else:
			##### 4
			# for i in os.listdir(d):
			# 	if not i.startswith('source') and not i.startswith('ret-'):
			# 		idx = int(i.split('.')[0])
			# 		on = os.path.join(path, p, i)
			# 		nn = os.path.join(path, p, '%s-%s'%(code[idx - 1], i))
			# 		# print on, nn
			# 		shutil.move(on, nn)
			##### 5
			if len(code) != 4:
				continue
			try:
				for i in os.listdir(d):
					# print d, code
					if not i.startswith('source') and not i.startswith('ret-'):
						idx = int(i.split('.')[0])
						on = os.path.join(path, p, i)
						nn = os.path.join(path, p, '%s-%s'%(code[idx - 1], i))
						print on, nn
						shutil.move(on, nn)
			except:pass

##### 6
def moveToTrain():
	train = 'train'
	import re
	import json
	import cv2
	if os.path.isfile('retDict.json'):
		retDict = json.load(open('retDict.json', 'r'))
	else:
		retDict = {}
	pRet = re.compile(r'\w-\d.png')
	path = 'result'
	for i in os.listdir(path):
		if i.startswith('.'):continue
		p = os.path.join(path, i)
		for f in os.listdir(p):
			m = pRet.match(f)
			if m is not None:
				r = os.path.join(p, f)
				k = f.split('.')[0].split('-')[0]
				retDict.setdefault(k, 0)
				retDict[k] = retDict[k] + 1
				np = os.path.join(train, k)
				if not os.path.isdir(np):
					os.makedirs(np)
				nf = os.path.join(np, '%s.png'%retDict[k])
				# shutil.copy(r, nf)
				try:
					image = cv2.imread(r)
					res = cv2.resize(image, (600, 600), interpolation = cv2.INTER_AREA)
					cv2.imwrite(nf, res)
				except:
					print r
				# print '%s >>> %s'%(r, nf)
	r = open('retDict.json', 'w')
	r.write(json.dumps(retDict, indent = 4))
	r.close()

# moveToTrain()

def moveToValidation():
	import os
	path = 'train'
	for i in os.listdir(path):
		if i.startswith('.'):
			continue
		pr = os.path.join('train', i)
		p = os.path.join('validation', i)
		if not os.path.isdir(p):
			os.makedirs(p)
		total = len(os.listdir(pr))
		pro = 0.2
		if total < 20:
			pro = 0.5
		size = int(total * pro)
		print pr, size
		f = random.sample(os.listdir(pr), size)
		for t in f:
			shutil.move(os.path.join(pr, t), os.path.join(p, t))

# moveToValidation()

def resizeTrain():
	import os
	import cv2
	path = 'train'
	for i in os.listdir(path):
		if i.startswith('.'):continue
		for f in os.listdir(os.path.join(path, i)):
			if f.startswith('.'):continue
			fn = os.path.join(path, i, f)
			image = cv2.imread(fn)
			res = cv2.resize(image, (300, 300), interpolation = cv2.INTER_AREA)
			cv2.imwrite(fn, res)
resizeTrain()