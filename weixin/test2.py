# from captcha.image import ImageCaptcha
import cv2
import numpy as np
import os
import random
import string
import tensorflow as tf
# import tensorflow.keras as keras
# import tensorflow.keras.backend as K
import keras.backend as K
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import LSTM
from keras.layers import MaxPooling2D
from keras.layers import Permute
from keras.layers import Reshape, Dropout
from keras.layers import TimeDistributed, Activation
from keras.models import Model, Sequential
from keras.optimizers import Adadelta
from keras.models import load_model
from keras.layers.recurrent import GRU
from keras.layers.merge import add, concatenate
from keras.optimizers import SGD

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
        img_gray = cv2.transpose(img_gray,(w_resize,h_resize))
        img_gray = cv2.flip(img_gray,1)
        img_gray = np.expand_dims(img_gray, axis = 2)
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

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    # y_pred = y_pred[:, 2:, :] 测试感觉没影响
    y_pred = y_pred[:, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def generate_data(imgs, labels_encode, batch_size):
    while True:
        test_idx = np.random.choice(range(len(imgs)), batch_size)
        batch_imgs = imgs[test_idx]
        batch_labels = labels_encode[test_idx]
        # print(batch_imgs.shape, batch_labels.shape)
        input_length = np.zeros([batch_size, 1])
        label_length = np.zeros([batch_size, 1])
        for i in range(batch_size):
            input_length[i] = 78 // 4 - 2
            label_length[i] = 4
        inputs = {'the_input': batch_imgs,
                  'the_labels': batch_labels,
                  'input_length': input_length,
                  'label_length': label_length
                  # 'source_str': source_str  # used for visualization only
                  }
        outputs = {'ctc': np.zeros(batch_size)}  # dummy data for dummy loss function
        # return (inputs, outputs)
        yield inputs, outputs # 元组的第一个元素为输入，第二个元素为训练标签，即自定义loss函数时的y_true


def generate_data_2(imgs, labels_encode, batch_size):
    while True:
        test_idx = np.random.choice(range(len(imgs)), batch_size)
        batch_imgs = imgs[test_idx]
        batch_labels = labels_encode[test_idx]
        # print(batch_imgs.shape, batch_labels.shape)
        input_length = np.zeros([batch_size, 1])
        label_length = np.zeros([batch_size, 1])
        for i in range(batch_size):
            input_length[i] = 4
            label_length[i] = 4
        inputs = {'the_input': batch_imgs,
                  'the_labels': batch_labels,
                  'input_length': input_length,
                  'label_length': label_length
                  # 'source_str': source_str  # used for visualization only
                  }
        outputs = {'ctc': np.zeros(batch_size)}  # dummy data for dummy loss function
        # return (inputs, outputs)
        yield inputs, outputs # 元组的第一个元素为输入，第二个元素为训练标签，即自定义loss函数时的y_true
           

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
        X = np.array(imgs[idx:idx+32])
        y_pred = model.predict(X)
        shape = y_pred[:, :, :].shape
        out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(shape[0]) * shape[1])[0][0])[:,:4]
        for i in range(len(X)):
            str_out = ''.join([chars[x] for x in out[i] if x!=-1 ])
            if labels[i] != str_out:
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

def train_test():
    # generate_img()
    imgs, labels, labels_encode = load_img()
    
    # labels_input = Input([None], dtype='int32')

    img_w = 156
    img_h = 64
    conv_filters = 16
    kernel_size = (3, 3)
    input_shape = (img_w, img_h, 1)
    pool_size = 2
    time_dense_size = 32
    rnn_size = 512

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv1')(input_data)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv2')(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

    conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filters)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

    # cuts down input size going into RNN:
    inner = Dense(time_dense_size, activation=act, name='dense1')(inner)

    # Two layers of bidirectional GRUs
    # GRU seems to work as well, if not better than LSTM:
    gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(inner)
    gru1_merged = add([gru_1, gru_1b])
    gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)

    # transforms RNN output to character activations:
    inner = Dense(len(chars) + 1, kernel_initializer='he_normal',
                  name='dense2')(concatenate([gru_2, gru_2b]))
    y_pred = Activation('softmax', name='softmax')(inner)
    base_model = Model(inputs=input_data, outputs=y_pred)

    labels = Input(name='the_labels', shape=[4], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    # clipnorm seems to speeds up convergence
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    fit_model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    fit_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)




    # adadelta = Adadelta(lr=0.05)
    # fit_model.compile(
    #     loss=lambda y_true, y_pred: y_pred,
    #     optimizer=adadelta)
    # fit_model.summary()
    # import sys
    # sys.exit()
    
    fit_model.fit_generator(
    generate_data(imgs, labels_encode, 32), 
    epochs=10, 
    steps_per_epoch=100, 
    verbose=1)
    fit_model.save('fit_model.h5')
    base_model.save('model.h5')
    # test()
    
if __name__ == '__main__':
    # train_test()
    test()
