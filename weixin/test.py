# from captcha.image import ImageCaptcha
import cv2
import numpy as np
import os
import random
import string
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Permute
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.models import load_model

chars = '0123456789abcdefghijklmnopqrstuvwxyz' # 验证码字符集
char_map = {chars[c]: c for c in range(len(chars))} # 验证码编码（0到len(chars) - 1)
idx_map = {value: key for key, value in char_map.items()} # 编码映射到字符
idx_map[-1] = '' # -1映射到空

# def generate_img(img_dir = 'data'):
#     for length in range(4, 8): # 验证码长度
#         if not os.path.exists(f'{img_dir}/{length}'):
#             os.makedirs(f'{img_dir}/{length}')
#         for _ in range(10000): 
#             img_generator = ImageCaptcha()
#             char = ''.join([random.choice(chars) for _ in range(length)])
#             img_generator.write(chars=char, output=f'{img_dir}/{length}/{char}.jpg')


def load_img():
    labels = [] # 验证码真实标签{长度：标签列表}
    # print(labels)
    imgs = [] # 图片BGR数据字典{长度：BGR数据列表}
    ### 读取图片
    for file in os.listdir('data'):
        img = cv2.imread('data/%s'%file)
        labels.append(file[:-4])
        height, width, _ = img.shape
        h_resize = 32
        w_resize = 78
        img_gray = cv2.cvtColor(cv2.resize(img, (w_resize, h_resize)), cv2.COLOR_BGR2GRAY) # 缩小图片固定宽度为32，并转为灰度图
        imgs.append(img_gray)

    ### 编码真实标签
    labels_encode = []
    for label in labels:
        label = [char_map[i] for i in label]
        labels_encode.append(label)
    return np.array(imgs), np.array(labels), np.array(labels_encode)


def ctc_loss(args):
    return K.ctc_batch_cost(*args)
def ctc_decode(softmax):
    return K.ctc_decode(softmax, K.tile([K.shape(softmax)[1]], [K.shape(softmax)[0]]))[0][0]


def generate_data(imgs, labels_encode, batch_size):
    while True:
        test_idx = np.random.choice(range(len(imgs)), batch_size)
        batch_imgs = imgs[test_idx]
        batch_labels = labels_encode[test_idx]
        yield ([batch_imgs, batch_labels], None) # 元组的第一个元素为输入，第二个元素为训练标签，即自定义loss函数时的y_true
           

def char_decode(label_encode): 
    return [''.join([idx_map[column] for column in row]) for row in label_encode]

    
# def generate_test_data(batch_size):
#     img_generator = ImageCaptcha()
#     while True:
#         test_labels_batch = []
#         test_imgs_batch = []
#         length = random.randint(4, 7)
#         for _ in range(batch_size):
#             char = ''.join([random.choice(chars) for _ in range(length)])
#             img = img_generator.generate_image(char)
#             img = np.asarray(img) 
#             test_labels_batch.append(char)
#             h = 32
#             w = int(img.shape[1] * 32 / img.shape[0])
#             img_gray = cv2.cvtColor(cv2.resize(img, (w, h)), cv2.COLOR_BGR2GRAY)
#             test_imgs_batch.append(img_gray)
#         yield([np.array(test_imgs_batch), np.array(test_labels_batch)])


def test():
    model = load_model('model.h5')
    error_cnt = 0
    # iterator = generate_test_data(test_batch_size)
    imgs, labels, labels_encode = load_img()
    for idx in range(0, len(labels), 32):
        labels_pred = model.predict(np.array(imgs[idx:idx+32]))
        labels_pred = char_decode(labels_pred)
        for label, label_pred in zip(labels[idx:idx+32], labels_pred):
            if label != label_pred:
                error_cnt += 1
    print(error_cnt)
    print(len(labels))
    # for _ in range(test_iter_num):
    #     test_imgs_batch, test_labels_batch = next(iterator)
    #     labels_pred = model.predict_on_batch(np.array(test_imgs_batch))
    #     labels_pred = char_decode(labels_pred)   
    #     for label, label_pred in zip(test_labels_batch, labels_pred):
    #         if label != label_pred:
    #             error_cnt += 1
    #             print(f'{label} -> {label_pred}')
    # print(f'总样本数：{test_batch_size * test_iter_num} | '
    #       f'错误数：{error_cnt} | '
    #       f'准确率：{1 - error_cnt / test_batch_size / test_iter_num}')


def train():
    # generate_img()
    imgs, labels, labels_encode = load_img()
    
    labels_input = Input([None], dtype='int32')
    sequential = Sequential([
        Reshape([32, -1, 1], input_shape=[32, None]),
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
        Permute((2, 1, 3)),
        TimeDistributed(Flatten()),
        LSTM(units=128, return_sequences=True),
        LSTM(units=128, return_sequences=True),
        TimeDistributed(Dense(len(chars) + 1, activation='softmax'))
    ])
    input_length = Lambda(lambda x: K.tile([[K.shape(x)[1]]], [K.shape(x)[0], 1]))(sequential.output)
    label_length = Lambda(lambda x: K.tile([[K.shape(x)[1]]], [K.shape(x)[0], 1]))(labels_input)
    output = Lambda(ctc_loss)([labels_input, sequential.output, input_length, label_length])
    fit_model = Model(inputs=[sequential.input, labels_input], outputs=output)
    ctc_decode_output = Lambda(ctc_decode)(sequential.output)
    model = Model(inputs=sequential.input, outputs=ctc_decode_output)
    adadelta = Adadelta(lr=0.05)
    fit_model.compile(
        loss=lambda y_true, y_pred: y_pred,
        optimizer=adadelta)
    fit_model.summary()
    import sys
    sys.exit()
    
    fit_model.fit_generator(
    generate_data(imgs, labels_encode, 32), 
    epochs=100, 
    steps_per_epoch=100, 
    verbose=1)
    fit_model.save('fit_model.h5')
    model.save('model.h5')
    test()
    
if __name__ == '__main__':
    train()
