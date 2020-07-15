# -*- coding: utf-8 -*-
# @Time    : 2020/4/16 15:06
# @Author  : zhaogang

# -*- coding: utf-8 -*-
# @Time    : 2020/4/13 15:47
# @Author  : zhaogang

import os,sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(parentdir)
sys.path.insert(0,parentdir)

import tensorflow as tf
import numpy as np
from yolo3.keras_yolo3 import yolo_text
from tensorflow.keras.models import load_model
K = tf.keras.backend
Model = tf.keras.models.Model
Input= tf.keras.layers.Input

from tensorflow.keras.models import load_model
import tensorflow as tf
import os
import os.path as osp
from tensorflow.keras import backend as K

from tensorflow.python.platform import gfile
K.set_learning_phase(0)



def load_pb(pb_file_path):
    sess = tf.Session()
    with gfile.FastGFile(pb_file_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')


def h5_to_pb(h5_model,output_dir,model_name,out_prefix = "output_",log_tensorboard = True):
    if osp.exists(output_dir) == False:
        os.mkdir(output_dir)
    out_nodes = []
    for i in range(len(h5_model.outputs)):
        out_nodes.append(out_prefix + str(i + 1))
        tf.identity(h5_model.output[i],out_prefix + str(i + 1))
    sess = K.get_session()
    from tensorflow.python.framework import graph_util,graph_io
    init_graph = sess.graph.as_graph_def()
    main_graph = graph_util.convert_variables_to_constants(sess,init_graph,out_nodes)
    graph_io.write_graph(main_graph,output_dir,name = model_name,as_text = False)
    if log_tensorboard:
        from tensorflow.python.tools import import_pb_to_tensorboard
        import_pb_to_tensorboard.import_to_tensorboard(osp.join(output_dir,model_name),output_dir)


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
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

def box_layer(inputs, anchors, num_classes):
    y1, y2, y3, image_shape, input_shape = inputs
    out = [y1, y2, y3]

    num_layers = len(out)
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    boxes = []
    scores = []
    input_shape = K.cast(input_shape, tf.float32)
    image_shape = K.cast(image_shape, tf.float32)
    # new_shape   = K.round(image_shape * K.min(input_shape/image_shape))
    # offset = (input_shape-new_shape)/2./input_shape
    # scale = input_shape/new_shape

    for lay in range(num_layers):
        box_xy, box_wh, box_confidence, box_class_probs = yolo_head(out[lay], anchors[anchor_mask[lay]],
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

name='weights-densent-05-loss_344.7720.hdf5'
anchors = '8,11, 8,16, 8,23, 8,33, 8,48, 8,97, 8,139, 8,198, 8,283'
anchors = [float(x) for x in anchors.split(',')]
anchors = np.array(anchors).reshape(-1, 2)
num_anchors = len(anchors)
class_names = ['none', 'text', ]  ##text
num_classes = len(class_names)
textModel = yolo_text(num_classes, anchors, train=False)

textModel.load_weights('../yolo3/model/'+name)



textModel.save('model/weights-densent-05-loss_344.7720.h5')
model=load_model('model/weights-densent-05-loss_344.7720.h5')
print(model.summary())
# image_shape = Input(shape=(2,))  ##图像原尺寸:h,w
# input_shape = Input(shape=(2,))  ##图像resize尺寸:h,w
# box_score = box_layer([*textModel.output, image_shape, input_shape], anchors, num_classes)
#
# yolo_inputs=[textModel.input,image_shape, input_shape]
# yolo_outputs=box_score
# yolo_model = Model(yolo_inputs,box_score)
#
#
# print(yolo_model.summary())
# #print(textModel.summary())
# output_dir='model/'
# output_graph_name='weights-densent-05-loss_344.7720.pb'
# h5_to_pb(yolo_model,output_dir = output_dir,model_name = output_graph_name)
#
# K.clear_session()
# load_pb('model/weights-densent-05-loss_344.7720.pb')


