# Import libraries
import cv2
import numpy as np
import time
import streamlit as st
import requests, os
from PIL import Image
import streamlit.components.v1 as components


#########################################################
##### GIAO DIỆN
#########################################################
#st.title("**OBJECT DETECTION**")
st.markdown("<h1 style='text-align: center;'>OBJECT DETECTION</h1>", unsafe_allow_html=True)
st.markdown("""

|Menter   | Huỳnh Trọng Nghĩa  |
|:-------:|:------------------:|
| Mentees |Nguyễn Chính Nghiệp  |
|         |Hà Sơn Tùng         |
"""
            
            , unsafe_allow_html=True)


#Ham tải về
##  Kiểm tra file 'name' có tồn tại chưa || không thì kéo nội dung từ link url tạo thành name 
##
def download(url, name):      
    if (os.path.exists(name)==False):
        #st.write("Đang lấy file %s..." % name)
        w = requests.get(url).content  # lấy nội dung url
        with open(name,'wb') as f:
            st.write(f.write(w))   # in ra màn hình
        f.close()
#     else:
#         st.write("Đã tìm thấy file %s!" % name)

st.write("Đang lấy file weights...")
download('https://archive.org/download/yolov4-custom_best_202110/yolov4-custom_best.weights', 'yolov4-custom_best.weights')
download('https://archive.org/download/yolov4-custom_best_202110/yolov4-custom.cfg', 'yolov4-custom.cfg')
download('https://archive.org/download/yolov4-custom_best_202110/yolo.names', 'yolo.names')
st.write("Đã xong.")


option = st.selectbox('Chọn model',('Faster-RCNN', 'Yolov4', 'Mobile phone'))
#st.write('You selected:', option)

##################################################################
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), (255,0,0), 2)
    cv2.putText(img, label + "%0.2f" % confidence , (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)

def drawBox(image, points):
    height, width = image.shape[:2]
    for (label, xi,yi, wi, hi) in points:
        center_x = int(xi * width)
        center_y = int(yi * height)
        w = int(wi * width)
        h = int(hi * height)
        # Rectangle coordinates
        x = int(center_x - w / 2)
        y = int(center_y - h / 2)
        cv2.rectangle(image, (x, y), (x + w, y + h), black, 1)
    return
def savePredict(name, text):
    textName = name + '.txt'
    with open(textName, 'w+') as groundTruth:
        groundTruth.write(text)
        groundTruth.close()


  
#################
#### MAIN
################
img_l = st.file_uploader("Upload Image",type=['jpg'])
try:
    img = Image.open(img_l)
    image = np.array(img)
    st.image(image, "Ảnh gốc")
except: pass

btn = st.button("Băt đầu nhận diện")

if btn:
    if option=='Yolov4':        
        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392
        
        classes = None
        with open("yolo.names", 'r') as f: # Edit CLASS file
            classes = [line.strip() for line in f.readlines()]

        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

        net = cv2.dnn.readNet("yolov4-custom_best.weights", "yolov4-custom.cfg") # Edit WEIGHT and CONFIC file
        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(get_output_layers(net))
        #print(outs)
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.2
        nms_threshold = 0.4

        start = time.time()

        for out in outs:
            for detection in out:
                scores = detection[5:]
                #print(scores)
                class_id = np.argmax(scores)
                #print('b')
                #print(class_id)
                confidence = scores[class_id]
                #print(confidence)
                if confidence > 0.75:
                    #print(confidence)
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    #print(w,h,x,y)
                    class_ids.append(class_id)
    #                 """if confidence < 0.6:
    #                     class_ids.append(2)""" #change
                    confidences.append(float(confidence))
                    #print(confidence)
                    #print(class_ids)
                    boxes.append([x, y, w, h])
                    #print(boxes)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        Result = ""
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            textpredict = "{} {} {}\n".format(str(class_ids[i]), x+ w/2, y+h/2)
            #print(textpredict)
            draw_prediction(image, class_ids[i],confidences[i], round(x), round(y), round(x + w), round(y + h))
            Result += textpredict
            #print(Result)


        scale_percent = 100
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        image = cv2.resize(src=image, dsize=(width,height))


        end = time.time()
        st.write("YOLO Execution time: " + str(end-start))
        st.image(image, "Ảnh đã nhận diện")
    elif option=='Faster-RCNN':
        st.write("Oke Rcnn")
    

