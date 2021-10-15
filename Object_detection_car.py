import streamlit as st
import sys, os
import importlib

st.markdown("<h1 style='text-align: center;'>OBJECT DETECTION</h1>", unsafe_allow_html=True)
st.markdown("""

|Menter   | Huỳnh Trọng Nghĩa  |
|:-------:|:------------------:|
| Mentees |Nguyễn Chính Nghiệp  |
|         |Hà Sơn Tùng         |
"""
, unsafe_allow_html=True)

st.write("")

app = st.selectbox("Chọn mô hình", ["YoloV4", "Faster-RCNN"])

if (app=="YoloV4"):
    import yolo_detection
    importlib.reload(yolo_detection)
    yolo_detection.yolo()
    
elif (app=="Faster-RCNN"):
    import faster_rcnn_detection
    importlib.reload(faster_rcnn_detection)
    faster_rcnn_detection.main_rcnn()
