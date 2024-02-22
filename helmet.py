
#from ultralytics import YOLO
#import cv2
#
#model = YOLO("best_results.pt")
#
#results = model.predict(source="0", show=True)
#
#cv2.imshow("image", results)
#cv2.waitKey(0) 
#print(results)
 



#import PIL
#
#import streamlit as st
#from ultralytics import YOLO
#
## Replace the relative path to your weight file
#model_path = 'best_results.pt'
#
## Setting page layout
#st.set_page_config(
#    page_title="Object Detection using YOLOv8",  # Setting page title
#    page_icon="🤖",     # Setting page icon
#    layout="wide",      # Setting layout to wide
#    initial_sidebar_state="expanded"    # Expanding sidebar by default
#)
#
## Creating sidebar
#with st.sidebar:
#
#    # Model Options
#    confidence = float(st.slider(
#        "Select Model Confidence", 25, 100, 40)) / 100
#
## Creating main page heading
#st.title("Object Detection using YOLOv8")
# 
## Creating two columns on the main page
#col1, col2 = st.columns(2)
#
## Adding image to the first column if image is uploaded
#with col1:
#    uploaded_image = st.camera_input("Take a picture")
#        
#
#try:
#    model = YOLO(model_path)
#except Exception as ex:
#    st.error(
#        f"Unable to load model. Check the specified path: {model_path}")
#    st.error(ex)
#
#if st.sidebar.button('Detect Objects'):
#    res = model.predict(uploaded_image,
#                        conf=confidence,
#                        show=True
#                        )
#    boxes = res[0].boxes
#    res_plotted = res[0].plot()[:, :, ::-1]
#    with col2:
#        st.image(res_plotted,
#                 caption='Detected Image',
#                 use_column_width=True
#                 )
#        try:
#            with st.expander("Detection Results"):
#                for box in boxes:
#                    st.write(box.xywh)
#        except Exception as ex:
#            st.write("No image is uploaded yet!")




#from ultralytics import YOLO
#import cv2
#import math 
## start webcam
#cap = cv2.VideoCapture(0)
#cap.set(3, 640)
#cap.set(4, 480)
#
## model
#model = YOLO("best_results.pt")
#
## object classes
#classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']
#
#
#while True:
#    success, img = cap.read()
#    results = model(img, stream=True)
#
#    # coordinates
#    for r in results:
#        boxes = r.boxes
#
#        for box in boxes:
#            # bounding box
#            x1, y1, x2, y2 = box.xyxy[0]
#            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
#
#            # put box in cam
#            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
#
#            # confidence
#            confidence = math.ceil((box.conf[0]*100))/100
#            print("Confidence --->",confidence)
#
#            # class name
#            cls = int(box.cls[0])
#            print("Class name -->", classNames[cls])
#
#            # object details
#            org = [x1, y1]
#            font = cv2.FONT_HERSHEY_SIMPLEX
#            fontScale = 1
#            color = (255, 0, 0)
#            thickness = 2
#
#            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
#
#    cv2.imshow('Webcam', img)
#    if cv2.waitKey(1) == ord('q'):
#        break
#
#cap.release()
#cv2.destroyAllWindows()

####

import math
import cv2 
import streamlit as st 
import numpy as np 
import tempfile
from ultralytics import YOLO

cap = cv2.VideoCapture(0)

model = YOLO("best_results.pt")

# object classes
classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']


st.title('Helmet Detection')

frame_placeholder = st.empty()

stop_button_pressed = st.button('Stop')

while cap.isOpened() and not stop_button_pressed:
    ret, frame = cap.read()
    results = model(frame, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(frame, classNames[cls], org, font, fontScale, color, thickness)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame_placeholder.image(frame, channels='RGB')
    

    if cv2.waitKey(1) & 0xFF == ord('q') or stop_button_pressed:
        break

cap.release()
cv2.destroyAllWindows()
