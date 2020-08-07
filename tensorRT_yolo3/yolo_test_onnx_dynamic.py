# -*- coding: utf-8 -*-
# @Time    : 2020/7/31 16:31
# @Author  : zhaogang

import os
import tensorflow as tf
import numpy as np
import time
import tensorRT_yolo3.common as common

Input = tf.keras.layers.Input
Lambda = tf.keras.layers.Lambda
load_model = tf.keras.models.load_model
Model = tf.keras.models.Model
K = tf.keras.backend
# Concatenate = tf.keras.layers.Concatenate
concatenate = tf.keras.layers.concatenate
from yolo3.keras_yolo3 import preprocess_true_boxes, yolo_text
from PIL import Image
import cv2

# K.clear_session()
# tf.reset_default_graph()
graph = tf.get_default_graph()
sess = K.get_session()
import matplotlib.pyplot as plt
from yolo3.detector.text_proposal_connector import TextProposalConnector
import tensorrt as trt


def normalize(data):
    if data.shape[0] == 0:
        return data
    max_ = data.max()
    min_ = data.min()
    return (data - min_) / (max_ - min_) if max_ - min_ != 0 else data - min_


def nms(boxes, threshold, method='Union'):
    if boxes.size == 0:
        return np.empty((0, 3))
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    I = np.argsort(s)
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while I.size > 0:
        i = I[-1]
        pick[counter] = i
        counter += 1
        idx = I[0:-1]
        xx1 = np.maximum(x1[i], x1[idx])
        yy1 = np.maximum(y1[i], y1[idx])
        xx2 = np.minimum(x2[i], x2[idx])
        yy2 = np.minimum(y2[i], y2[idx])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if method is 'Min':
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)
        I = I[np.where(o <= threshold)]
    pick = pick[0:counter]
    return pick


def get_boxes(bboxes):
    """
        boxes: bounding boxes
    """
    text_recs = np.zeros((len(bboxes), 8), np.int)
    index = 0
    for box in bboxes:

        b1 = box[6] - box[7] / 2
        b2 = box[6] + box[7] / 2
        x1 = box[0]
        y1 = box[5] * box[0] + b1
        x2 = box[2]
        y2 = box[5] * box[2] + b1
        x3 = box[0]
        y3 = box[5] * box[0] + b2
        x4 = box[2]
        y4 = box[5] * box[2] + b2

        disX = x2 - x1
        disY = y2 - y1
        width = np.sqrt(disX * disX + disY * disY)
        fTmp0 = y3 - y1
        fTmp1 = fTmp0 * disY / width
        x = np.fabs(fTmp1 * disX / width)
        y = np.fabs(fTmp1 * disY / width)
        if box[5] < 0:
            x1 -= x
            y1 += y
            x4 += x
            y4 -= y
        else:
            x2 += x
            y2 += y
            x3 -= x
            y3 -= y

        text_recs[index, 0] = x1
        text_recs[index, 1] = y1
        text_recs[index, 2] = x2
        text_recs[index, 3] = y2
        text_recs[index, 4] = x3
        text_recs[index, 5] = y3
        text_recs[index, 6] = x4
        text_recs[index, 7] = y4
        index = index + 1

    return text_recs


class TextDetector:
    """
        Detect text from an image
    """

    def __init__(self, MAX_HORIZONTAL_GAP=30, MIN_V_OVERLAPS=0.6, MIN_SIZE_SIM=0.6):
        """
        pass
        """
        self.text_proposal_connector = TextProposalConnector(MAX_HORIZONTAL_GAP, MIN_V_OVERLAPS, MIN_SIZE_SIM)

    def detect_region(self, text_proposals, scores, size,
                      TEXT_PROPOSALS_MIN_SCORE=0.7,
                      TEXT_PROPOSALS_NMS_THRESH=0.3,
                      TEXT_LINE_NMS_THRESH=0.3, ):
        """
        Detecting texts from an image
        :return: the bounding boxes of the detected texts
        @@param:TEXT_PROPOSALS_MIN_SCORE:TEXT_PROPOSALS_MIN_SCORE=0.7##过滤字符box阀值
        @@param:TEXT_PROPOSALS_NMS_THRESH:TEXT_PROPOSALS_NMS_THRESH=0.3##nms过滤重复字符box
        @@param:TEXT_LINE_NMS_THRESH:TEXT_LINE_NMS_THRESH=0.3##nms过滤行文本重复过滤阀值
        @@param:MIN_RATIO:MIN_RATIO=1.0#0.01 ##widths/heights宽度与高度比例
        @@param:LINE_MIN_SCORE:##行文本置信度
        @@param:TEXT_PROPOSALS_WIDTH##每个字符的默认最小宽度
        @@param:MIN_NUM_PROPOSALS,MIN_NUM_PROPOSALS=1##最小字符数
        """
        # text_proposals, scores=self.text_proposal_detector.detect(im, cfg.MEAN)
        keep_inds = np.where(scores > TEXT_PROPOSALS_MIN_SCORE)[0]  ###
        text_proposals, scores = text_proposals[keep_inds], scores[keep_inds]

        sorted_indices = np.argsort(scores.ravel())[::-1]
        text_proposals, scores = text_proposals[sorted_indices], scores[sorted_indices]

        # nms for text proposals
        keep_inds = nms(np.hstack((text_proposals, scores)), TEXT_PROPOSALS_NMS_THRESH,
                        GPU_ID=self.GPU_ID)  ##nms 过滤重复的box
        text_proposals, scores = text_proposals[keep_inds], scores[keep_inds]

        groups_boxes, groups_scores = self.text_proposal_connector.get_text_region(text_proposals, scores, size)
        return groups_boxes, groups_scores

    def detect(self, text_proposals, scores, size,
               TEXT_PROPOSALS_MIN_SCORE=0.7,
               TEXT_PROPOSALS_NMS_THRESH=0.3,
               TEXT_LINE_NMS_THRESH=0.3,

               ):
        """
        Detecting texts from an image
        :return: the bounding boxes of the detected texts
        @@param:TEXT_PROPOSALS_MIN_SCORE:TEXT_PROPOSALS_MIN_SCORE=0.7##过滤字符box阀值
        @@param:TEXT_PROPOSALS_NMS_THRESH:TEXT_PROPOSALS_NMS_THRESH=0.3##nms过滤重复字符box
        @@param:TEXT_LINE_NMS_THRESH:TEXT_LINE_NMS_THRESH=0.3##nms过滤行文本重复过滤阀值
        @@param:MIN_RATIO:MIN_RATIO=1.0#0.01 ##widths/heights宽度与高度比例
        @@param:LINE_MIN_SCORE:##行文本置信度
        @@param:TEXT_PROPOSALS_WIDTH##每个字符的默认最小宽度
        @@param:MIN_NUM_PROPOSALS,MIN_NUM_PROPOSALS=1##最小字符数

        """
        # text_proposals, scores=self.text_proposal_detector.detect(im, cfg.MEAN)
        keep_inds = np.where(scores > TEXT_PROPOSALS_MIN_SCORE)[0]  ###

        text_proposals, scores = text_proposals[keep_inds], scores[keep_inds]
        # print('-----------------------------')
        # print('text:',text_proposals)
        # print('1')
        # print('scores1:',sorted(scores))
        sorted_indices = np.argsort(scores.ravel())[::-1]
        # print('sorted:',sorted_indices)
        #  print(scores)
        text_proposals, scores = text_proposals[sorted_indices], scores[sorted_indices]
        # print(text_proposals)
        # nms for text proposals
        if len(text_proposals) > 0:
            keep_inds = nms(np.hstack((text_proposals, scores)), TEXT_PROPOSALS_NMS_THRESH)  ##nms 过滤重复的box
            text_proposals, scores = text_proposals[keep_inds], scores[keep_inds]

            scores = normalize(scores)

            text_lines = self.text_proposal_connector.get_text_lines(text_proposals, scores, size)  ##合并文本行
            keep_inds = nms(text_lines, TEXT_LINE_NMS_THRESH)  ##nms
            if len(keep_inds) == 0:
                return []
            text_lines = text_lines[keep_inds]
            return text_lines
        else:
            return []


class ModelData(object):
    MODEL_FILE = "model/weights-densent-05-loss_344.7720.uff"
    INPUT_NAME = "input_1"
    INPUT_SHAPE = (1, -1, -1, 3)
    OUTPUT_NAME = ["conv2d_58/BiasAdd", "conv2d_66/BiasAdd", "conv2d_74/BiasAdd"]


class YoloPredict(object):
    def __init__(self, path):
        self.init_tensorrt_engine()

        global graph
        global sess
        with graph.as_default():
            anchors = '8,11, 8,16, 8,23, 8,33, 8,48, 8,97, 8,139, 8,198, 8,283'
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
            num_anchors = len(anchors)
            class_names = ['none', 'text', ]  ##text
            num_classes = len(class_names)
            self.image_shape = K.placeholder(shape=(2,))  ##图像原尺寸:h,w
            self.input_shape = K.placeholder(shape=(2,))  ##图像resize尺寸:h,w

            self.out1 = Input(batch_shape=(1, None, None, 21))
            self.out2 = Input(batch_shape=(1, None, None, 21))
            self.out3 = Input(batch_shape=(1, None, None, 21))

            self.box_score = self.box_layer([self.out1, self.out2, self.out3, self.image_shape, self.input_shape],
                                            anchors, num_classes)

    def init_tensorrt_engine(self):
        self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        self.engine = self.get_engine("tensorRT_yolo3/model/weights-densent-05-loss_344.7720_.h5_none.onnx",
                                 "tensorRT_yolo3/model/engine_dynamic_fp16_2.onnx")
        #self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(engine)
        self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers2(self.engine, 2080, 2080)
        #self.inputs, self.outputs, self.bindings, self.stream=None,None,None,None
        self.context = self.engine.create_execution_context()

    def get_engine(self, onnx_file_path, engine_file_path=""):
        """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

        def build_engine():
            """Takes an ONNX file and creates a TensorRT engine to run inference with"""
            with trt.Builder(self.TRT_LOGGER) as builder, builder.create_network(
                    common.EXPLICIT_BATCH) as network, \
                    trt.OnnxParser(network, self.TRT_LOGGER) as parser,\
                    builder.create_builder_config() as config:
                builder.max_batch_size = 1
                # Parse model file
                builder.fp16_mode = True
                builder.strict_type_constraints = True

                config.set_flag(trt.BuilderFlag.FP16)
                config.max_workspace_size=common.GiB(8)
                if not os.path.exists(onnx_file_path):
                    print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(
                        onnx_file_path))
                    exit(0)
                print('Loading ONNX file from path {}...'.format(onnx_file_path))
                with open(onnx_file_path, 'rb') as model:
                    print('Beginning ONNX file parsing')
                    if not parser.parse(model.read()):
                        print('ERROR: Failed to parse the ONNX file.')
                        for error in range(parser.num_errors):
                            print(parser.get_error(error))
                        return None
                # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
                network.get_input(0).shape = [1, 3, -1, -1]
                print('Completed parsing of ONNX file')
                print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))

                ############################################
                #config=builder.create_builder_config()
                profile = builder.create_optimization_profile()
                profile.set_shape(network.get_input(0).name, (1,3, 32, 32), (1,3, 608, 608), (1,3, 2050, 2050))
                config.add_optimization_profile(profile)
                ##################################################

                engine = builder.build_engine(network,config)
                print("Completed creating Engine")
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
                return engine

        if os.path.exists(engine_file_path):
            # If a serialized engine exists, use it instead of building an engine.
            print("Reading engine from file {}".format(engine_file_path))
            with open(engine_file_path, "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        else:
            return build_engine()

    def load_normalized_test_case(self, img):

        im = Image.fromarray(img)
        w, h = im.size

        w_, h_ = self.resize_im(w, h, scale=608, max_scale=2048)  ##短边固定为608,长边max_scale<4000
        print('w_',w_)
        print('h_',h_)
       # w_, h_ = 608, 608

        boxed_image = im.resize((w_, h_), Image.BICUBIC)

        image_data = np.array(boxed_image, dtype='float32', order='C')

        image = image_data / 255.

        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.array(image, dtype=np.float32, order='C')

        return image, w, h, w_, h_

    def resize_im(self, w, h, scale=416, max_scale=608):
        f = float(scale) / min(h, w)
        if max_scale is not None:
            if f * max(h, w) > max_scale:
                f = float(max_scale) / max(h, w)
        newW, newH = int(w * f), int(h * f)

        return newW - (newW % 32), newH - (newH % 32)

    def yolo_head(self, feats, anchors, num_classes, input_shape, calc_loss=False):
        """Convert final layer features to bounding box parameters."""
        num_anchors = len(anchors)
        # Reshape to batch, height, width, num_anchors, box_params.
        anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

        grid_shape = K.shape(feats)[1:3]  # height, width
        grid_y = tf.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                         [1, grid_shape[1], 1, 1])
        grid_x = tf.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                         [grid_shape[0], 1, 1, 1])
        grid = K.concatenate([grid_x, grid_y])
        grid = K.cast(grid, K.dtype(feats))

        feats = K.reshape(
            feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

        # Adjust preditions to each spatial grid point and anchor size.
        box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
        box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
        box_confidence = K.sigmoid(feats[..., 4:5])
        box_class_probs = K.sigmoid(feats[..., 5:])

        if calc_loss == True:
            return grid, feats, box_xy, box_wh
        return box_xy, box_wh, box_confidence, box_class_probs

    def box_layer(self, inputs, anchors, num_classes):
        y1, y2, y3, image_shape, input_shape = inputs
        out = [y1, y2, y3]

        num_layers = len(out)
        anchor_mask = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        boxes = []
        scores = []
        input_shape = K.cast(input_shape, tf.float32)
        image_shape = K.cast(image_shape, tf.float32)
        # new_shape   = K.round(image_shape * K.min(input_shape/image_shape))
        # offset = (input_shape-new_shape)/2./input_shape
        # scale = input_shape/new_shape

        for lay in range(num_layers):
            box_xy, box_wh, box_confidence, box_class_probs = self.yolo_head(out[lay], anchors[anchor_mask[lay]],
                                                                             num_classes, input_shape)
            # box_xy = (box_xy - offset) * scale
            # box_wh = box_wh*scale

            box_score = box_confidence * box_class_probs
            box_score = K.reshape(box_score, [-1, num_classes])

            box_mins = box_xy - (box_wh / 2.)
            box_maxes = box_xy + (box_wh / 2.)
            box = K.concatenate([
                box_mins[..., 0:1],  # xmin
                box_mins[..., 1:2],  # ymin
                box_maxes[..., 0:1],  # xmax
                box_maxes[..., 1:2]  # ymax
            ], axis=-1)

            box = K.reshape(box, [-1, 4])

            boxes.append(box)

            scores.append(box_score)

        boxes = K.concatenate(boxes, axis=0)
        scores = K.concatenate(scores, axis=0)

        boxes *= K.concatenate([image_shape[::-1], image_shape[::-1]])

        return boxes, scores[..., 1]

    def predict(self, img, prob=0.05):

        image, w, h, w_, h_ = self.load_normalized_test_case(img)

        # print(image.shape)
        print('w,h:',w,h)
        print('w_,h_:',w_,h_)
        #self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers2(self.engine,h_,w_)
        #image=image.astype('float16')
        self.inputs[0].host = image
        trt_outputs = common.do_inference_v3(self.context,
                                             bindings=self.bindings,
                                             inputs=self.inputs,
                                             outputs=self.outputs,
                                             stream=self.stream,
                                             h_=h_,
                                             w_=w_
                                             )

        print("Prediction: " + str(trt_outputs))
        scale_w=w_/608
        scale_h=h_/608
        print('scale_w:',scale_w)
        print('scale_h:', scale_h)
        print(len(trt_outputs[0]))

        output_shapes = [(1, int(h_/32), int(w_/32), 21),
                         (1, int(h_/16), int(w_/16), 21),
                         (1, int(h_/8), int(w_/8), 21)]

        print(output_shapes)
        trt_outputs = [output[:shape[1]*shape[2]*shape[3]].reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]
        # trt_outputs[0]=np.transpose(trt_outputs[0],[0,2,3,1])
        # trt_outputs[1] = np.transpose(trt_outputs[1], [0,2, 3, 1])
        # trt_outputs[2] = np.transpose(trt_outputs[2], [0,2, 3, 1])
        # print('trt_shape:',trt_outputs[0].shape)
        global graph
        global sess
        with graph.as_default():
            ##定义 graph变量 解决web.py 相关报错问题
            """
            pred = textModel.predict_on_batch([image_data,imgShape,inputShape])
            box,scores = pred[:,:4],pred[:,-1]

            """
            # try:
            box, scores = sess.run(
                [self.box_score],
                feed_dict={
                    self.out1: trt_outputs[2],
                    self.out2: trt_outputs[1],
                    self.out3: trt_outputs[0],
                    self.input_shape: [h_, w_],
                    self.image_shape: [h, w],
                    K.learning_phase(): 0
                })[0]
            # except:
            #     print('a')
        # print(box.shape)
        # print(scores.shape)
        # print('box[0]:',box[0:18])
        # print('w,h',w,' ',h)
        # print('prob:',prob)
        keep = np.where(scores > prob)
        # print('keep:',keep)
        box[:, 0:4][box[:, 0:4] < 0] = 0
        box[:, 0][box[:, 0] >= w] = w - 1
        box[:, 1][box[:, 1] >= h] = h - 1
        box[:, 2][box[:, 2] >= w] = w - 1
        box[:, 3][box[:, 3] >= h] = h - 1
        box = box[keep[0]]

        scores = scores[keep[0]]
        # scores=sorted(scores)
        # print(scores)
        return box, scores

    def plot_box(self, img, boxes):
        plt.imshow(img)
        plt.show()
        cv2.imwrite('img_1003.jpg', img)

        blue = (255, 0, 0)  # 18
        tmp = np.copy(img)
        for box in boxes:
            cv2.rectangle(tmp, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), blue, 2)  # 19

        plt.imshow(tmp)
        plt.show()
        cv2.imwrite('img_1004.jpg', tmp)

        return Image.fromarray(tmp)

    def detect(self, text_proposals, scores, size,
               TEXT_PROPOSALS_MIN_SCORE=0.7,
               TEXT_PROPOSALS_NMS_THRESH=0.3,
               TEXT_LINE_NMS_THRESH=0.3,

               ):
        """
        Detecting texts from an image
        :return: the bounding boxes of the detected texts
        @@param:TEXT_PROPOSALS_MIN_SCORE:TEXT_PROPOSALS_MIN_SCORE=0.7##过滤字符box阀值
        @@param:TEXT_PROPOSALS_NMS_THRESH:TEXT_PROPOSALS_NMS_THRESH=0.3##nms过滤重复字符box
        @@param:TEXT_LINE_NMS_THRESH:TEXT_LINE_NMS_THRESH=0.3##nms过滤行文本重复过滤阀值
        @@param:MIN_RATIO:MIN_RATIO=1.0#0.01 ##widths/heights宽度与高度比例
        @@param:LINE_MIN_SCORE:##行文本置信度
        @@param:TEXT_PROPOSALS_WIDTH##每个字符的默认最小宽度
        @@param:MIN_NUM_PROPOSALS,MIN_NUM_PROPOSALS=1##最小字符数

        """
        # text_proposals, scores=self.text_proposal_detector.detect(im, cfg.MEAN)
        keep_inds = np.where(scores > TEXT_PROPOSALS_MIN_SCORE)[0]  ###

        text_proposals, scores = text_proposals[keep_inds], scores[keep_inds]

        print(keep_inds)
        print('===========')

        sorted_indices = np.argsort(scores.ravel())[::-1]
        text_proposals, scores = text_proposals[sorted_indices], scores[sorted_indices]

        # nms for text proposals
        if len(text_proposals) > 0:
            keep_inds = self.nms(np.hstack((text_proposals, scores)), TEXT_PROPOSALS_NMS_THRESH)  ##nms 过滤重复的box
            text_proposals, scores = text_proposals[keep_inds], scores[keep_inds]

            scores = self.normalize(scores)

            text_lines = self.text_proposal_connector.get_text_lines(text_proposals, scores, size)  ##合并文本行
            keep_inds = self.nms(text_lines, TEXT_LINE_NMS_THRESH)  ##nms
            text_lines = text_lines[keep_inds]
            return text_lines
        else:
            return []

    def text_detect(self, boxes, scores, img):

        MAX_HORIZONTAL_GAP = 30
        MIN_V_OVERLAPS = 0.6
        MIN_SIZE_SIM = 0.6

        # TEXT_PROPOSALS_MIN_SCORE = 0.7
        # TEXT_PROPOSALS_NMS_THRESH = 0.3
        # TEXT_LINE_NMS_THRESH = 0.3

        TEXT_PROPOSALS_MIN_SCORE = 0.35
        TEXT_PROPOSALS_NMS_THRESH = 0.15
        TEXT_LINE_NMS_THRESH = 0.15

        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        textdetector = TextDetector(MAX_HORIZONTAL_GAP, MIN_V_OVERLAPS, MIN_SIZE_SIM)
        shape = img.shape[:2]

        # print(boxes)
        boxes = textdetector.detect(boxes,
                                    scores[:, np.newaxis],
                                    shape,
                                    TEXT_PROPOSALS_MIN_SCORE,
                                    TEXT_PROPOSALS_NMS_THRESH,
                                    TEXT_LINE_NMS_THRESH,
                                    )

        text_recs = get_boxes(boxes)
        newBox = []
        rx = 1
        ry = 1
        for box in text_recs:
            x1, y1 = (box[0], box[1])
            x2, y2 = (box[2], box[3])
            x3, y3 = (box[6], box[7])
            x4, y4 = (box[4], box[5])
            newBox.append([x1 * rx, y1 * ry, x2 * rx, y2 * ry, x3 * rx, y3 * ry, x4 * rx, y4 * ry])
        return newBox

    def predict2(self, img):
        box, scores = self.predict(img, prob=0.35)
        # print(box)
        #  print(scores)
        #  print(len(scores))
        newbox = self.text_detect(box, scores, img)
        print('-------------------')
        sorted_indices = np.argsort(scores.ravel())[::-1]
        #box, scores = box[sorted_indices], scores[sorted_indices]
        # print(box)
        # print('---------------')
        # print(scores)
        # print('--------------')
        # print(newbox)
        return newbox

    # def predict2(self,img,graph,sess):
    #     box, scores, img2 = self.predict(img, graph,sess,prob=0.01)
    #     newbox = self.text_detect(box, scores, img, img2)
    #     return newbox,img2


if __name__ == '__main__':
    basemodel = YoloPredict(path='model/weights-densent-05-loss_344.7720.hdf5')
    p = 'D:/software/Image-master/ocr/image/115.jpg'
    img = np.array(Image.open(p))
    t1 = time.time()
    newbox = basemodel.predict2(img)
    t2 = time.time()
    print('时延：', t2 - t1)

    basemodel.plot_box(img, newbox)


