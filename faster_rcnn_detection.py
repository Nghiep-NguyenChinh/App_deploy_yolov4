from flask import Flask, render_template, request
import os 
import sys
from time import gmtime, strftime
# from pywebio.platform.flask import webio_view
# from pywebio import STATIC_PATH
# from pywebio.input import *
# from pywebio.output import *

import cv2
import numpy as np
import pickle
from Faster_RCNN.keras_frcnn import config
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import Faster_RCNN.test as FRCNN
import YOLO.detection as YOLO

app = Flask(__name__, template_folder="templates")


src_folder = "src_img"
output_folder = os.path.join('static', 'result_img')
app.config['UPLOAD_FOLDER'] = output_folder

if not os.path.isdir(src_folder):
	os.mkdir(src_folder)
if not os.path.isdir(output_folder):
	os.makedirs(output_folder)

#~~~~~~~~~~~~~~~~~~~~~Faster-RCNN~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sys.path.append('Faster_RCNN')
config_output_filename = "Faster_RCNN/config.pickle"
with open(config_output_filename, 'rb') as f_in:
    C = pickle.load(f_in)

C.best_model_path = "Faster_RCNN/best_weights"
# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

if C.network == 'resnet50':
    import keras_frcnn.resnet as nn
elif C.network == 'xception':
    import keras_frcnn.xception as nn
elif C.network == 'inception_resnet_v2':
    import keras_frcnn.inception_resnet_v2 as nn
elif C.network == 'vgg':
    import keras_frcnn.vgg as nn

class_mapping = C.class_mapping

if 'bg' not in class_mapping:
    class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}

C.num_rois = 32

if C.network == 'resnet50':
    num_features = 1024
elif C.network == 'xception':
    num_features = 1024
elif C.network == 'inception_resnet_v2':
    num_features = 1088
elif C.network == 'vgg':
    num_features = 512

if K.image_data_format() == 'channels_first':
    input_shape_img = (3, None, None)
    input_shape_features = (num_features, None, None)
else:
    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, num_features)


img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

model_rpn = Model(img_input, rpn_layers)

model_classifier = Model([feature_map_input, roi_input], classifier)

print('Loading weights from {}'.format(os.path.join(C.best_model_path, "best_model_frcnn.hdf5")))
model_rpn.load_weights(os.path.join(C.best_model_path, "best_model_frcnn.hdf5"), by_name=True)
model_classifier.load_weights(os.path.join(C.best_model_path, "best_model_frcnn.hdf5"), by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')

#~~~~~~~~~~~~~~~~~~~~~YOLO~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
config_path = "YOLO/yolov4-custom.cfg"
best_weight_path = "YOLO/yolov4-custom_best.weights"
class_name_path = "YOLO/yolo.names"
YOLO_net = cv2.dnn.readNet(best_weight_path, config_path)

@app.route('/')
def index():
	model_list = ["Faster-RCNN", "YOLO"]
	return render_template("index.html", model_list=model_list)

@app.route('/prediction', methods=["POST"])
def prediction():
	time = strftime("%Y-%m-%d_%H-%M-%S", gmtime())

	img = request.files['img']
	img_name = "img_{}.jpg".format(time)
	img.save(os.path.join(src_folder, img_name))

	choose_model = request.form.get("choose_model")

	if choose_model == "Faster-RCNN":
		output_img, time_pred = FRCNN.predict(16, C, src_folder, img_name, class_mapping, class_to_color, model_rpn, model_classifier)
	elif choose_model == "YOLO":
		output_img, time_pred = YOLO.predict(class_name_path, src_folder, img_name, YOLO_net)
	output_img_name = "img_{}_out.jpg".format(time)
	output_img_path = os.path.join(output_folder, output_img_name)
	cv2.imwrite(output_img_path, output_img)

	return render_template("prediction.html", output_img_path=os.path.join(app.config['UPLOAD_FOLDER'], output_img_name), time_pred=time_pred)

if __name__ == "__main__":
	app.run(debug=True)
