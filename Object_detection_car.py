import streamlit as st
import sys, os
import importlib

#sys.path.append("pages")
#sys.path.append("yolo")

app = st.selectbox("Chọn App", ["YoloV4", "Faster-RCNN"])

if (app=="YoloV4"):
    import yolo_detection
    importlib.reload(yolo_detection)
    yolo_detection.yolo()
    
elif (app=="Faster-RCNN"):
    import faster_rcnn_detection
    importlib.reload(faster_rcnn_detection)
    faster_rcnn_detection.main_rcnn()
