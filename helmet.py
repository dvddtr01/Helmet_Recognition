import math
import cv2 
import streamlit as st 
import numpy as np 
import tempfile
from ultralytics import YOLO

cap = cv2.VideoCapture(2)

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
