#!usr/bin/python
#coding=utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf8')
from keras.models import load_model
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array

from ImageSplitter import splitImage

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

class CodeAnalyzer(object):
	def __init__(self):
		self.model = load_model('code_model.h5')

	def start(self, imageFile):
		result = []
		imageList = splitImage(imageFile)
		for x in imageList:
			x = img_to_array(x)
			x = np.expand_dims(x, axis = 0)
			x = preprocess_input(x)
			out = self.model.predict_classes(x)
			result.append(classes[out[0]])
		return ''.join(result)

if __name__ == '__main__':
	print CodeAnalyzer().start(sys.argv[1])