import cv2
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import streamlit as st
from keras.utils import img_to_array, load_img
from keras.models import load_model
from streamlit_option_menu import option_menu


model = load_model('genderdetect_final_3.h5')
model1 = load_model('facedetect_final_3.h5')

# class
classes = ['man', 'woman']
classes2 = ['Khuong', 'Phuc', 'Thanh', 'Thao', 'TuanKiet', 'VanTrung']

face_cascade = cv2.CascadeClassifier(r'haarcascade_frontalface_alt.xml')



def main():
    
    selected = option_menu(
        menu_title= None,
        options= ["Home", "Image", "RealTime", "Author"],
        icons=["house", "image", "camera", "list-task"],
        menu_icon= "cast",
        default_index= 0,
        orientation= "horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "blue", "font-size": "20px"},
            "nav-link": {"font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "#758078", "color":"black"},
            "nav-link-selected": {"background-color": "yellow"}
        }
    )

    if selected == "Home":
        st.title(":green[ỨNG DỤNG NHẬN DIỆN DANH TÍNH THỜI GIAN THỰC] :camera_with_flash:")
        st.subheader("MÔ TẢ:")
        st.write("""
                    Ứng dụng nhận diện danh tính sẽ nhận diện hai đối tượng.

                    1. Nhận diện giới tính.

                    2. Nhận diện tên đối tượng.

                    Nếu bạn muốn tìm hiểu thêm thông tin hãy nhấn vào nút xem thêm.

                    """)
        more = st.button("Xem Thêm")
        if more:
            st.subheader("Úng dụng nhận diện Danh tính sử dụng mô hình CNN")
            st.write("""Việc sử dụng nhận dạng khuôn mặt để xác minh danh tính là một lĩnh vực quan trọng trong xử lý ảnh và trí tuệ nhân tạo. Quá trình này thường bao gồm việc phát hiện khuôn mặt trong một hình ảnh, còn được gọi là Face Detection, và sau đó xác định xem khuôn mặt đó có phù hợp với danh tính đã biết hay không. Trong quá trình nhận dạng khuôn mặt để xác minh danh tính, công nghệ nhận dạng khuôn mặt được sử dụng để phát hiện và trích xuất các đặc trưng độc nhất của khuôn mặt, như các điểm mốc và cấu trúc khuôn mặt. Các đặc trưng này sau đó được so sánh với dữ liệu đã biết, chẳng hạn như hình ảnh đăng ký của người dùng trong cơ sở dữ liệu, để xác định xem khuôn mặt có phù hợp với danh tính đã biết hay không. Một số ứng dụng của việc sử dụng nhận dạng khuôn mặt để xác minh danh tính bao gồm quá trình đăng nhập an toàn và thuận tiện vào các hệ thống kỹ thuật số, quản lý danh tính trong các tổ chức và công ty, và xác minh danh tính trong các giao dịch trực tuyến và truy cập vào dịch vụ kỹ thuật số.
            """)

    
    if selected == "Image":
        st.markdown("""
        <style>
        .title {
            text-align: center;
            color: Green;
        }
        <style>
        """, unsafe_allow_html=True)

        st.markdown('<h1 class="title">VUI LÒNG CẬP NHẬP HÌNH ẢNH</h1', unsafe_allow_html= True)
        upload_file = st.file_uploader("choose an image file")

        if (upload_file is not None):

            file_byte = np.asarray(bytearray(upload_file.read()), dtype= np.uint8)
            img = cv2.imdecode(file_byte, 1)
            gray_face = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_face)

            img2 = cv2.resize(img, [256,256])
            st.image(img2, channels="BGR", caption="Ảnh gốc")
            
            Predict_button = st.button("Predict")
            
            if(Predict_button):
                
                with st.spinner("Predicting"):
                    for (x,y,w,h) in faces:
                        startX = x
                        startY = y
                        endX = x + w
                        endY = y + h 

                        rec = cv2.rectangle(img, (startX-10,startY-20), (endX+10,endY+10), (0,255,0),8)
                        face_crop = np.copy(img[startY:endY,startX:endX])
                        
                        face_crop = np.array(face_crop)
                        face_crop_iden = tf.image.resize(face_crop, [224,224])
                        face_crop_gender = tf.image.resize(face_crop, [100,100])
                        face_crop_iden = np.expand_dims(face_crop_iden, axis=0)
                        face_crop_gender = np.expand_dims(face_crop_gender, axis=0)

                        # apply gender detection on face
                        conf = model.predict(face_crop_gender)[0] # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]
                        iden = model1.predict(face_crop_iden)[0]

                        # get label with max accuracy
                        idx = np.argmax(conf)
                        idenx = np.argmax(iden)
                        label1 = classes[idx]
                        label2 = classes2[idenx]
                    
                    rec = cv2.resize(rec, [256,256])

                st.image(rec, channels= "BGR")
                st.subheader("Your Gender is : {} ".format(label1))
                if (label1 == 'man'):
                    st.subheader("Your Name is: {}".format(label2))

    if selected == "RealTime":
        
        st.markdown("""
        <style>
        .header {
            text-align: center;
            color: Green;
        }
        <style>
        """, unsafe_allow_html=True)

        st.markdown('<h2 class="header">SỬ DỤNG WEBCAM ĐỂ DỰ ĐOÁN</h2', unsafe_allow_html= True)
        st.subheader("VUI LÒNG NHẤN VÀO NÚT MỞ WEBCAM ĐỂ SỬ DỤNG WEBCAM")
        camera_button_open = st.button("OPEN WEBCAM")
        if camera_button_open:
            frame_window = st.image([])
            webcam = cv2.VideoCapture(0)
            camera_button_close = st.button("CLOSE WEBCAM")
            while webcam.isOpened():
                # read frame from webcam 
                status, frame = webcam.read()

                grayface = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(grayface)

                for (x,y,w,h) in faces:

                    # get corner points of face rectangle        
                    startX = x
                    startY = y
                    endX = x + w
                    endY = y + h

                    # draw rectangle over face
                    cv2.rectangle(frame, (startX-10,startY-20), (endX+10,endY+10), (0,255,0), 2)

                    # crop the detected face region
                    face_crop = np.copy(frame[startY:endY,startX:endX])

                    if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
                        continue
                    
                    face_crop = np.array(face_crop)
                    face_crop_iden = tf.image.resize(face_crop, [224,224])
                    face_crop_gender = tf.image.resize(face_crop, [100,100])
                    face_crop_iden = np.expand_dims(face_crop_iden, axis=0)
                    face_crop_gender = np.expand_dims(face_crop_gender, axis=0)
                    

                    # apply gender detection on face
                    conf = model.predict(face_crop_gender)[0] # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]
                    iden = model1.predict(face_crop_iden)[0]

                    # get label with max accuracy
                    idx = np.argmax(conf)
                    idenx = np.argmax(iden)
                    label1 = classes[idx]
                    label2 = classes2[idenx]

                    label_gender = "{}".format(label1)
                    label_iden = "{}".format(label2)
                    Y = startY - 10 if startY - 10 > 10 else startY + 10

                    # write label and confidence above face rectangle
                    cv2.putText(frame, label_gender, (startX, startY-30),  cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)
                    cv2.putText(frame, label_iden, (startX, endY+30),  cv2.FONT_HERSHEY_SIMPLEX,
                               0.7, (0, 255, 0), 2)

                imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_window.image(imgRGB)

                    
                if camera_button_close == True:
                    break

    if (selected == "Author"):
        st.title(':green[INFORMATION] :card_index:')
        st.subheader("""
                    Developed by : Nguyen Quang Phuc """)
        st.subheader("""
                    [Email] : 
                    (nqphuc2509002@gmail.com) """)
        st.subheader("""
                    [Youtube] : 
                    (https://www.youtube.com/channel/UCRXNzaalEEA3fLz0DvosRUA) """)
        st.subheader("""
                    [Github] : 
                    (https://github.com/quangphuc2509) """)
        st.subheader("""
                    [Facebook] : 
                    (https://www.facebook.com/profile.php?id=100010905552167) """)

if __name__ == "__main__":
    main()
