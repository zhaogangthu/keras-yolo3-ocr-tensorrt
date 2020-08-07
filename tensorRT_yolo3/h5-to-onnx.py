# -*- coding: utf-8 -*-
# @Time    : 2020/4/16 15:08
# @Author  : zhaogang
import os
os.environ['TF_KERAS'] = '1'
from tensorflow.python.keras import saving
load_model = saving.load_model
import keras
import keras2onnx
import onnx

model = load_model('model/weights-densent-05-loss_344.7720.h5')
print(model.summary())
#onnx_model = keras2onnx.convert_keras(model, model.name,target_opset=10,channel_first_inputs= 'input_1')
onnx_model = keras2onnx.convert_keras(model, model.name,target_opset=10,channel_first_inputs= 'input_1')#11不支持pad

temp_model_file = 'model/weights-densent-05-loss_344.7720.h5.onnx'
onnx.save_model(onnx_model, temp_model_file)

