import streamlit as st
import sys, os
import importlib

sys.path.append("pages")
sys.path.append("yolo")

app = st.selectbox("Chọn App", ["Smart Steward", "YoloV4"])

if (app=="Smart Steward"):
    if (os.path.exists("loginok.log")==False):
        ten = st.text_input("Tên đăng nhập:")
        mk = st.text_input("Mật khẩu")
        check = st.button("Đăng nhập")

        if check:
            if (ten==st.secrets["db_username"] and mk==st.secrets["db_password"]):
                f = open("loginok.log","w")
                f.close()
                raise st.script_runner.RerunException(st.script_request_queue.RerunData(None))
            else: st.write("Tài khoản không đúng!")
    else:
        import pages
        importlib.reload(pages)
        pages.main()
elif (app=="YoloV4"):
    import detection
    importlib.reload(detection)
    detection.main()
