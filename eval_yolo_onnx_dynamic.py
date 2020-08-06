# -*- coding: utf-8 -*-
# @Time    : 2020/7/31 16:32
# @Author  : zhaogang

# -*- coding: utf-8 -*-
# @Time    : 2020/4/16 15:28
# @Author  : zhaogang

# -*- coding: utf-8 -*-
# @Time    : 2020/4/15 14:55
# @Author  : zhaogang

# -*- coding: utf-8 -*-
# @Time    : 2020/3/26 17:20
# @Author  : zhaogang

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf

tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5
tf_config.gpu_options.allow_growth = True  # 自适应
sess = tf.Session(config=tf_config)
graph = tf.get_default_graph()

import cv2

import numpy as np
import time
import matplotlib.pyplot as plt
from math import *

from tensorRT_yolo3.yolo_test_onnx_dynamic import YoloPredict
import ctpn.utils as utils
from ctpn.text_proposal_connector_oriented import TextProposalConnectorOriented
from densent_ocr.densenet_ocr_predict import DenseNetOcrPredict
# from pytesseract import image_to_string
import json
from PIL import Image
from tensorflow.python.keras.backend import set_session


def dumpRotateImage(img, degree, pt1, pt2, pt3, pt4):
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) +
                    height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) +
                   width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(
        img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    pt1 = list(pt1)
    pt3 = list(pt3)

    [[pt1[0]], [pt1[1]]] = np.dot(matRotation,
                                  np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation,
                                  np.array([[pt3[0]], [pt3[1]], [1]]))
    ydim, xdim = imgRotation.shape[:2]
    imgOut = imgRotation[max(1, int(pt1[1])):min(ydim - 1, int(pt3[1])),
             max(1, int(pt1[0])):min(xdim - 1, int(pt3[0]))]
    # height,width=imgOut.shape[:2]
    return imgOut


def predict(ctpn_model, densenet_model, imgpath, gold_data):
    img = cv2.imread(imgpath)
    # print(img)
    try:
        h, w, c = img.shape
    except:
        return

    m_img = img - utils.IMAGE_MEAN
    m_img = np.expand_dims(m_img, axis=0)
    # print(m_img.shape)
    # print(m_img[0][300][300])
    texts = ctpn_model.predict(m_img)

    for text in texts:
        #  print(text)
        if text[4] < text[0] or text[5] < text[1]:
            continue
        degree = degrees(atan2(text[3] - text[1], text[2] - text[0]))  ##图像倾斜角度

        partImg = dumpRotateImage(img, degree, [text[0], text[1]], [text[2], text[3]], [text[6], text[7]],
                                  [text[4], text[5]])

        ###
        plt.imshow(partImg)
        plt.show()
        cv2.imwrite('demo/12221.jpg', partImg)
        ###
        out, im = densenet_model.predict(partImg)
        #   print(out)
        if out == '':
            continue

        txt_name = 'res_img_' + imgpath.split('/')[-1].split('.')[0] + '.txt'
        with open('eval_data/submit/' + txt_name, 'a', encoding='utf8') as f:
            f.write(str(text[0]) + ',' + str(text[1]) + ',' + str(text[4]) + ',' + str(text[5]) + ',' + out + '\n')

    ######gold_data
    with open(gold_data, encoding='utf8') as f:
        for line in f:
            line_json = json.loads(line)
            ocr_locations = line_json['ocrLocations']
            for ocr_loc in ocr_locations:
                w = float(ocr_loc['w'])
                h = float(ocr_loc['h'])
                x = float(ocr_loc['x'])
                y = float(ocr_loc['y'])
                x1 = x
                y1 = y

                x3 = x + w
                y3 = y + h

                gold_text = ocr_loc['text']

                txt_name = 'gt_img_' + imgpath.split('/')[-1].split('.')[0] + '.txt'
                with open('eval_data/gt/' + txt_name, 'a', encoding='utf8') as f2:
                    f2.write(str(x1) + ',' + str(y1) + ',' + str(x3) + ',' + str(y3) + ',' + gold_text + '\n')


error_count1 = 0
error_count2 = 0


def predict_batch(yolo_model, densenet_model, imgpath, gold_data):
    ######gold_data
    with open(gold_data, encoding='utf8') as f:
        for line in f:
            line_json = json.loads(line)
            ocr_locations = line_json['ocrLocations']
            for ocr_loc in ocr_locations:
                w = float(ocr_loc['w'])
                h = float(ocr_loc['h'])
                x = float(ocr_loc['x'])
                y = float(ocr_loc['y'])
                x1 = x
                y1 = y

                x3 = x + w
                y3 = y + h

                gold_text = ocr_loc['text']

                txt_name = 'gt_img_' + imgpath.split('/')[-1].split('.')[0] + '.txt'
                with open('eval_data/gt/' + txt_name, 'a', encoding='utf8') as f2:
                    f2.write(str(x1) + ',' + str(y1) + ',' + str(x3) + ',' + str(y3) + ',' + gold_text + '\n')

    img = np.array(Image.open(imgpath).convert('RGB'))
    t1 = time.time()
    texts = yolo_model.predict2(img)
    t2 = time.time()
    print('yolo时间：', t2 - t1)
    # print(texts)
    # ######################################
    #     plt.imshow(img)
    #     plt.show()
    #     cv2.imwrite('demo/110.jpg', img)
    #     yolo_model.plot_box(img, texts)
    # #############################
    partImg_list = []
    text_list = []
    for text in texts:
        # print(text)
        if text[4] < text[0] or text[5] < text[1]:
            continue
        degree = degrees(atan2(text[3] - text[1], text[2] - text[0]))  ##图像倾斜角度

        partImg = dumpRotateImage(img, degree, [text[0], text[1]], [text[2], text[3]], [text[4], text[5]],
                                  [text[6], text[7]])

        # ####
        # plt.imshow(partImg)
        # plt.show()
        # cv2.imwrite('demo/110.jpg', partImg)
        # ################
        try:
            img2 = Image.fromarray(partImg.astype('uint8')).convert('RGB')
        except:
            print('error:', 1)
            global error_count1
            error_count1 += 1
            continue
        im = img2.convert('L')
        # print(im.size)
        scale = im.size[1] * 1.0 / 32
        w = im.size[0] / scale
        w = int(w)
        try:
            im = im.resize((w, 32), Image.ANTIALIAS)
        except:
            print('error:', 2)
            global error_count2
            error_count2 += 1
            continue

        new_im = Image.new("L", (560, 32))
        new_im.paste(im, (0, 0))
        img1 = new_im

        img1 = np.array(img1).astype(np.float32) / 255.0 - 0.5

        if img1.shape[1] < img1.shape[0]:
            continue

        partImg_list.append(img1)
        text_list.append(text)
        ###############
    if len(partImg_list) == 0:
        return 0

    partImg_list = np.array(partImg_list)
    partImg_list = np.expand_dims(partImg_list, 3)

    out_list = densenet_model.predict_batch(partImg_list)
    # print(out_list)
    for i, text in enumerate(text_list):

        out = out_list[i]
        # print(out)
        if out == '':
            continue
        txt_name = 'res_img_' + imgpath.split('/')[-1].split('.')[0] + '.txt'

        with open('eval_data/submit/' + txt_name, 'a', encoding='utf8') as f:
            f.write(str(text[0]) + ',' + str(text[1]) + ',' + str(text[4]) + ',' + str(text[5]) + ',' + out + '\n')

    return len(texts)


if __name__ == '__main__':

    imgpath = 'image_100000_120000/100006.jpeg'
    gold_data = 'label_100000_120000/100010.txt'
    # ctpn_model = CtpnPredict(path='ctpn/model_v100_300/weights-ctpnlstm-01-loss-0.1225.hdf5')
    # densenet_model = DenseNetOcrPredict('densent_ocr/model_0_1000000_300/weights-densent-02-acc_0.2627.hdf5')

    with graph.as_default():

        set_session(sess)

        yolo_model = YoloPredict(path='')

        densenet_model = DenseNetOcrPredict('densent_ocr/model_0_1000000/weights-densent-20-acc_0.4191.hdf5')

        # predict_batch(yolo_model, densenet_model, imgpath,gold_data)

        files = os.listdir('image_100000_120000/')
        files = sorted(files)
        t1 = time.time()
        num = 0
        for count, file in enumerate(files):
            if count > 10000:
                break
            print(file)
            file_name = file.split('/')[-1].split('.')[0]
            imgpath = 'image_100000_120000/' + file
            gold_data = 'label_100000_120000/' + file_name + '.txt'
            texts_num = predict_batch(yolo_model, densenet_model, imgpath, gold_data)
            num += texts_num
        t2 = time.time()
        print('总计用时：', t2 - t1)
        print('平均用时：', (t2 - t1) / len(files))
        print('总text数量：', num)
        print('error1:', error_count1)
        print('error2:', error_count2)


