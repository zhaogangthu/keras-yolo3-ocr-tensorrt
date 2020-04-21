# -*- coding: utf-8 -*-
# @Time    : 2020/2/20 16:41
# @Author  : zhaogang

from keras.layers import Input,Conv2D,MaxPooling2D,ZeroPadding2D
from keras.layers import Flatten,BatchNormalization,Permute,TimeDistributed,Dense,Bidirectional,GRU
from keras.models import Model


import numpy as np
from PIL import Image
import keras.backend  as K

from imp import reload
import densent_ocr.densenet as densenet
reload(densenet)

from keras.layers import Lambda
from keras.optimizers import SGD

import tensorflow as tf
import keras.backend.tensorflow_backend as K
from matplotlib import pyplot as plt
import time

class DenseNetOcrPredict(object):
    def __init__(self,modelPath):
        char = ''
        #with open('densent_ocr/char_std_5990.txt', encoding='utf-8') as f:
        #with open('crnn_data/char.txt', encoding='utf-8') as f:
        with open('crnn_data/char_0_1000000.txt', encoding='utf-8') as f:
            for ch in f.readlines():
                ch = ch.strip('\r\n')
                char = char + ch

        # char = char[1:] + '卍'
        # nclass = len(char)
        # modelPath = 'densent_ocr/model/weights-densent-02.hdf5'
        nclass = len(char)+1

        print('nclass:', len(char))
        self.id_to_char = {i: j for i, j in enumerate(char)}

        input = Input(shape=(32, None, 1), name='the_input')
        y_pred = densenet.dense_cnn(input, nclass)
        self.basemodel = Model(inputs=input, outputs=y_pred)
        self.basemodel.load_weights(modelPath)

        x = self.basemodel.output  # [batch_sizes, series_length, classes]
        input_length = Input(batch_shape=[None], dtype='int32')
        ctc_decode = K.ctc_decode(x, input_length=input_length)
        self.decode = K.function([x, input_length], [ctc_decode[0][0]])


    def predict(self,img_path):
        if isinstance(img_path,str):
            img = Image.open(img_path)
        else:
            img=img_path
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
        im = img.convert('L')
        scale = im.size[1] * 1.0 / 32
        w = im.size[0] / scale
        w = int(w)


        im = im.resize((w, 32), Image.ANTIALIAS)
        img = np.array(im).astype(np.float32) / 255.0 - 0.5
        X = img.reshape((32, w, 1))
        X = np.array([X])
        if X.shape[2]<X.shape[1]:
            return '',im
        try:

            y_pred = self.basemodel.predict(X)
        except:
            print('a')


        y_pred = y_pred[:, :, :]
        out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1])[0][0])[:, :]
        out = u''.join([self.id_to_char[x] for x in out[0]])

        return out, im

    def predict_batch(self,img_list):
        t1=time.time()
        y_pred = self.basemodel.predict(img_list)
        y_pred = y_pred[:, :, :]
        # t1=time.time()
        # ctc_decode=K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1], )[0][0]
        # t2=time.time()
        # print('解码时间1：',t2-t1)
        # out = K.get_value(ctc_decode)[:, :]
        # t3=time.time()
        # print('解码时间2：',t3-t2)
        t2=time.time()
        print('推理时间：',t2-t1)
        tmp=(np.ones(y_pred.shape[0]) * y_pred.shape[1])
        out =self.decode([y_pred,tmp])[0]
        out = [u''.join([self.id_to_char[x] for x in out[i] if x!=-1]).replace('曌',' ') for i in range(len(out))]
        t3=time.time()
        print('解码时间：',t3-t2)
        print('解码数量：',len(out))

        return out


if __name__ == '__main__':
    basemodel=DenseNetOcrPredict('model/weights-densent-06-acc_0.4026.hdf5')
    t1=time.time()
    out,im=basemodel.predict('D:/software/keras_ocr-master/crnn_data/image_100000_120000/100017_0.jpg')
    t2=time.time()
    print(t2-t1)
    print(out)