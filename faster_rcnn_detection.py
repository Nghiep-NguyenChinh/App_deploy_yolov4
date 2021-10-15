import streamlit as st
import os, sys, requests, cv2, pickle
from time import gmtime, strftime
from PIL import Image
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

sys.path.append("Faster_RCNN")

from Faster_RCNN.keras_frcnn import config
import Faster_RCNN.test as FRCNN

def download(url, name):      
    if (os.path.exists(name)==False):
        #st.write("Đang lấy file %s..." % name)
        w = requests.get(url).content  # lấy nội dung url
        with open(name,'wb') as f:
            f.write(w)  # ghi ra file
        f.close()
#     else:
#         st.write("Đã tìm thấy file %s!" % name)

def main_rcnn():
	
	st.write("Đang lấy file weights...")
	download('https://archive.org/download/best_model_frcnn/Faster_RCNN/best_weights/best_model_frcnn.hdf5', 'best_model_frcnn.hdf5')
	download('https://archive.org/download/best_model_frcnn/Faster_RCNN/config.pickle', 'config.pickle')
	st.write("Trạng thái: Sẵn sàng")
	
	img_l = st.file_uploader("Upload Image",type=['jpg'])
	try:
		img = Image.open(img_l)
		image = np.array(img)
		st.image(image, "Ảnh gốc")
	except: pass

	btn = st.button("Bắt đầu nhận diện")

	if btn:
		#~~~~~~~~~~~~~~~~~~~~~Faster-RCNN~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		#sys.path.append('Faster_RCNN')
		config_output_filename = "config.pickle"
		with open(config_output_filename, 'rb') as f_in:
		    C = pickle.load(f_in)

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

		#print('Loading weights from {}'.format(os.path.join(C.best_model_path, "best_model_frcnn.hdf5")))
		model_rpn.load_weights("best_model_frcnn.hdf5", by_name=True)
		model_classifier.load_weights("best_model_frcnn.hdf5", by_name=True)

		model_rpn.compile(optimizer='sgd', loss='mse')
		model_classifier.compile(optimizer='sgd', loss='mse')


		output_img, time_pred = FRCNN.predict(16, C, image, class_mapping, class_to_color, model_rpn, model_classifier)

		st.write("F-RCNN Execution time: " + str(time_pred))
		st.image(output_img, "Ảnh đã nhận diện")

if __name__ == "__main__":
	main_rcnn()
