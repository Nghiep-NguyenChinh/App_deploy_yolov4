# Import libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import time

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





image = cv2.imread("car_test2.jpg")
#image = cv2.imread("testing_images/vid_5_31700.jpg")



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
            """if confidence < 0.6:
                class_ids.append(2)""" #change
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

file = open("testt.txt","w+")
file.write(Result)
file.close()
#savePredict(pathSave, name, textPre) # Doi thanh con tro ve dia chi cua anh
    
scale_percent = 50
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
image = cv2.resize(src=image, dsize=(width,height))

cv2.imshow("object detection", image)


end = time.time()
print("YOLO Execution time: " + str(end-start))
cv2.waitKey()

cv2.imwrite("object-detection.jpg", image)
cv2.destroyAllWindows()
