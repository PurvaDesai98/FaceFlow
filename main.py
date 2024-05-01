# In this program we will apply various ML algorithms to the built in datasets in scikit-learn

# Importing required Libraries
# Importing Numpy
import numpy as np
# For model deployment
import streamlit as st
import os
import cv2

from mtcnn.mtcnn import MTCNN

#mean values for the blob model
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# display variables
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']



# Giving Title
st.title("FaceFlow :sunglasses:")
st.header("Real Time Age and Gender Detection")

# Now we are making a select box for dataset
model_name=st.sidebar.selectbox("Select model",
                  ("Caffe","Caffe+Clahe"))

# loading age and gender detectors
#loading caffe model
ageProto = os.path.abspath("./models/age_deploy.prototxt")
ageModel = os.path.abspath("./models/age_net.caffemodel")

genderProto = os.path.abspath("./models/gender_deploy.prototxt")
genderModel = os.path.abspath("./models/gender_net.caffemodel")


ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)


# The Next is selecting algorithm
# We will display this in the sidebar
input_type=st.sidebar.selectbox("Select Input type",
                     ("Image","Video"))

st.caption("Powered by OpenCV, Streamlit")

if input_type == 'Image':
    uploaded_file = st.sidebar.file_uploader("Choose a file:")
    if uploaded_file is not None:

        frame_placeholder = st.empty()

        face_model = MTCNN()
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        #getting results yolo model
        faces = face_model.detect_faces(img)


        #looping over the faces detected from yolo
        for face in faces:
            top_left_x, top_left_y, width, height = face['box']

            bottom_right_x= top_left_x + width
            bottom_right_y= top_left_y + height

            cv2.rectangle(img,( top_left_x,top_left_y),(bottom_right_x,  bottom_right_y),(255,0,0),2)

            detected_face=img[top_left_y:bottom_right_y,  top_left_x:bottom_right_x]

            processed_img=cv2.resize(detected_face,(227,227),interpolation=cv2.INTER_AREA)

            detected_face_blob = cv2.dnn.blobFromImage(processed_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            genderNet.setInput(detected_face_blob)
            ageNet.setInput(detected_face_blob)

            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]


            agePreds=ageNet.forward()
            age = ageList[agePreds[0].argmax()]

            label = "{}{}".format(gender,age)

            cv2.putText(img, label, (top_left_x+30, top_left_y+30), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 7, cv2.LINE_AA)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # st.image(img, caption='Detected Faces', use_column_width=True)

        frame_placeholder.image(img,channels="RGB")

elif input_type == 'Video':

    start_button_pressed = st.sidebar.button("Start")

    stop_button_pressed = st.sidebar.button("Stop")

    face_cascade = cv2.CascadeClassifier(os.path.abspath('./models/haarcascade_frontalface_default.xml'))


    if start_button_pressed:

        cap = cv2.VideoCapture(0)
        frame_placeholder = st.empty()

        while cap.isOpened() and not stop_button_pressed:
            ret, img = cap.read()
            if not ret:
                st.write("Video Capture Ended")
                break

            #coverting image to graysacale.
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            #looping over the faces coordinate values
            for (x,y,w,h) in faces:
                #drwing rectangle on the face region
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                #detected face
                detected_face=img[y:y+h, x:x+w]

                if model_name =='Caffe+Clahe':
                    lab_img= cv2.cvtColor(detected_face, cv2.COLOR_BGR2LAB)

                    #Splitting the LAB image to L, A and B channels, respectively
                    l, a, b = cv2.split(lab_img)

                    #Apply histogram equalization to the L channel
                    equ = cv2.equalizeHist(l)

                    #Combine the Hist. equalized L-channel back with A and B channels
                    updated_lab_img1 = cv2.merge((equ,a,b))

                    #Convert LAB image back to color (RGB)
                    hist_eq_img = cv2.cvtColor(updated_lab_img1, cv2.COLOR_LAB2BGR)


                    #clahe equalization
                    #Apply CLAHE to L channel
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                    clahe_img = clahe.apply(l)


                    #Combine the CLAHE enhanced L-channel back with A and B channels
                    updated_lab_img2 = cv2.merge((clahe_img,a,b))

                    #Convert LAB image back to color (RGB)
                    processed_img = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)

                else:
                    #Bilinear image interpolation
                    processed_img=cv2.resize(detected_face,(227,227),interpolation=cv2.INTER_AREA)


                #preprocessing with blob
                detected_face_blob = cv2.dnn.blobFromImage(processed_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

                #inputing into the network
                genderNet.setInput(detected_face_blob)
                ageNet.setInput(detected_face_blob)

                genderPreds = genderNet.forward()
                gender = genderList[genderPreds[0].argmax()]


                agePreds=ageNet.forward()
                age = ageList[agePreds[0].argmax()]


                label = "{}{}".format(gender,age)

                #predictions on frame
                cv2.putText(img, label, (x+30, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(img,channels="RGB")
            if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
                break

        cap.release()
        cv2.destroyAllWindows()