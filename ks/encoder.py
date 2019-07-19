# !usr/bin/python
# coding=utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import random
from glob import glob
import numpy as np
# from keras.optimizers import SGD, Adam
from keras.layers import Conv2D, Input, UpSampling2D, MaxPooling2D
from keras.preprocessing.image import load_img, img_to_array
# from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import load_model, Model
# from keras.callbacks import ModelCheckpoint, EarlyStopping

# import matplotlib.pyplot as plt

INPUT_SHAPE = (28, 28, 1)

def make_model():

	input_img = Input(shape=INPUT_SHAPE)  # adapt this if using `channels_first` image data format

	x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
	encoded = MaxPooling2D((2, 2), padding='same')(x)

	# at this point the representation is (4, 4, 8) i.e. 128-dimensional

	x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
	x = UpSampling2D((2, 2))(x)
	x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
	x = UpSampling2D((2, 2))(x)
	x = Conv2D(16, (3, 3), activation='relu')(x)
	x = UpSampling2D((2, 2))(x)
	decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

	autoencoder = Model(input_img, decoded)
	encoder = Model(input_img, encoded)
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
	return autoencoder, encoder, input_img

def build(model_path = None):
	autoencoder, encoder_model, input_img = make_model()
	if model_path:
		autoencoder = load_model(model_path)
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
	x_train = getTrain('source')
	x_test = getTrain('test_source')

	# modelCheckpoint = ModelCheckpoint('log/ep{epoch:03d}-loss{loss:.4f}.h5', monitor='loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
	# earlyStopping = EarlyStopping(monitor='loss', min_delta=0.0005, patience=5, verbose=1, mode='auto', baseline=None, restore_best_weights=False)

	autoencoder.fit(x_train, x_train,
		epochs = 128,
		batch_size = 128,
		shuffle = True,
		validation_data = (x_test, x_test))
	autoencoder.save('model.h5')
	encoder_model.save('encoder_model.h5')

def getTrain(path):
	data_X = []
	total_train = glob('%s/*.jpg'%path)
	random.shuffle(total_train)
	for i in total_train:
		img = img_to_array(load_img(i, target_size = INPUT_SHAPE, color_mode = 'grayscale'))
		img = img.astype('float32') / 255.
		data_X.append(img)
	return np.array(data_X)

def train_generate_batch_data(length = 50):
	data_X = []
	total_train = glob('source/*.jpg')
	# total_train.sort()
	while True:
		random.shuffle(total_train)
		for i in total_train:
			img = img_to_array(load_img(i, target_size = INPUT_SHAPE, color_mode = 'grayscale'))
			img = img.astype('float32') / 255.
			data_X.append(img)
			if len(data_X) == length:
				yield np.array(data_X), np.array(data_X)
				data_X = []

def test():
	x_test = getTrain('test_source')
	encoder = load_model('encoder_model.h5')
	ret = encoder.predict(x_test)
	print ret.shape

if __name__ == '__main__':
	build()
