#!/usr/bin/python
#coding=UTF-8

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import os
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import load_model
import numpy as np
from glob import glob
from PIL import Image
from cStringIO import StringIO

import ImageSplitter

size = (128, 128)

def VGG_16(classes, size = (128, 128), weights_path = None):

	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape = (size[0], size[1], 3), activation = 'relu'))
	model.add(MaxPooling2D((2, 2), strides = (2, 2)))

	model.add(Conv2D(32, (3, 3), activation = 'relu'))
	model.add(MaxPooling2D((2, 2), strides = (2, 2)))

	model.add(Conv2D(64, (3, 3), activation = 'relu'))
	model.add(MaxPooling2D((2, 2), strides = (2, 2)))

	model.add(Flatten())
	model.add(Dense(128, activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(classes, activation = 'softmax'))

	if weights_path:
		model.load_weights(weights_path)

	return model


train_data_dir = 'train'
validation_data_dir = 'validation'

classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'w', 'x', 'y', 'z']

def start():

	model = VGG_16(len(classes))
	# model = VGG16(weights = None, input_shape = (224, 224, 3), classes = len(classes))

	# sgd = SGD(lr = 0.1, decay = 1e-6, momentum = 0.9, nesterov = True)
	# model.compile(optimizer = sgd, loss = 'categorical_crossentropy')

	model.compile(loss='categorical_crossentropy',
				optimizer='rmsprop',
				metrics=['accuracy'])

	# 实例化图片生成器，说明图片是咋变化的
	train_datagen = ImageDataGenerator(
			# rotation_range=40,
			# width_shift_range=0.2,
			# height_shift_range=0.2,
			rescale=1./255,
			shear_range=0.2,
			zoom_range=0.2,)
			# horizontal_flip=True,
			# fill_mode='nearest')

	# 验证的数据不用变，只需要像素变成0-1区间的
	test_datagen = ImageDataGenerator(rescale=1./255)

	# 对这个实例化的生成器使用方法，说明它要干什么，从文件夹读图，标签。以及设置要生成目标的参数
	train_generator = train_datagen.flow_from_directory(
			train_data_dir,#从这个目录里面读取图片
			classes = classes,
			target_size = size,
			batch_size = 32,
			class_mode = 'categorical')

	# x = cv2.imread('train/asuka/0.jpg')  # this is a PIL image
	# # x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
	# x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

	# # the .flow() command below generates batches of randomly transformed images
	# # and saves the results to the `preview/` directory
	# i = 0
	# for batch in train_datagen.flow(x, batch_size=1, save_to_dir='preview', save_prefix='asuka', save_format='jpeg'):
	# 	i += 1
	# 	if i > 20:
	# 		break
	# return

	# for i in train_generator:
	# 	print i[0], i[1]
	#categorical会返回2D的one-hot编码标签,binary返回1D的二值标签

	validation_generator = test_datagen.flow_from_directory(
			validation_data_dir,
			classes = classes,
			target_size = size,
			batch_size = 32,  
			class_mode = 'categorical')

	# train_labels = np.array([0] * 100 + [1] * 100 + [2] * 100 + [3] * 100 + [4] * 96)
	# print train_labels
	# train_labels = keras.utils.to_categorical(train_labels, 5)
	# print train_labels

	# hist = model.fit_generator(
	# 		generator = train_generator,
	# 		steps_per_epoch = 32,
	# 		epochs = 1,#2000
	# 		# nb_epoch=nb_epoch,#50
	# 		validation_data = validation_generator,
	# 		validation_steps = 800)#800

	hist = model.fit_generator(
			generator = train_generator,
			steps_per_epoch = 2000,
			epochs = 1,#2000
			# nb_epoch=nb_epoch,#50
			validation_data = validation_generator,
			validation_steps = 800)#800

	model.save_weights('code_slim.h5')
	model.save('code_model.h5')
	import json
	f = open('history.txt', 'w')
	f.write(json.dumps(hist.history))
	f.close()
	# import os
	# os.system('sudo shutdown -h now')

# start()

# import cv2
def predict(img, model):
	# model = VGG_16(len(classes), 'code_slim.h5')
	# model = load_model('model.h5')
	# model = load_model('code_model.h5')
	# img = load_img(code, target_size = (128, 128))
	x = img_to_array(img)
	x = np.expand_dims(x, axis = 0)
	x = preprocess_input(x)
	# im = cv2.imread('result/10000/h-4.png')
	# im = cv2.resize(im, (100, 100), interpolation = cv2.INTER_CUBIC)
	# im = im[np.newaxis, :, :, :]
	out = model.predict_classes(x)
	return classes[out[0]]

def test():
	allNum = 0
	right = 0
	model = load_model('code_model.h5')
	filenames = glob('data/*.png')
	for fn in filenames:
		allNum += 1
		code = ''
		imgList = ImageSplitter.split_img(fn)
		for img in imgList:
			try:
				ret = predict(img, model)
			except:
				ret = ''
			code = code + ret
		if code == os.path.basename(fn).split('.')[0]:
			right += 1
		print fn, code
		print right, allNum
	print right, allNum

if __name__ == '__main__':
	# start()
	# img = Image.open('validation/b/104.png')
	# img = img.convert('RGB')
	# print img
	# print predict(img)
	test()