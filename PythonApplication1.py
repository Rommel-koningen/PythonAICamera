from asyncio.windows_events import NULL
from datetime import datetime
import cv2
import ultralytics
from ultralytics import YOLO
import requests
import json
import time
import socket
from CameraOakD import CameraOakD

import depthai as dai

model_path = 'weightsImproved/best.pt'
model = YOLO(model_path)



trashTypes = ['biodegradable', 'cardboard', 'glass', 'metal', 'paper', 'plastic']
testingSending = True

def detect_and_annotate2(model, frame):
    results = model.predict(frame, conf=0.6)
    
    if results != NULL:

        all_trash = []

        if testingSending:
            all_trash.append({
                "afval_Id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "afval_Type": "Test",
                "confidence": 0
            })

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label = trashTypes[cls_id]
                confi = float(box.conf)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                print("Categorie-ID: ", cls_id)
                print("Categorie-naam: ", label)

                all_trash.append({
                    "afval_Id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                    "afval_Type": label,
                    "confidence": confi
                })

        if all_trash != []:
            data_to_send = {
                "foto_Id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "datum_En_Tijd": "2025-06-17T10:12:13.790Z",
                "camera_Naam": "Cam1",
                "longitude": 0,
                "latitude": 0,
                "postcode": "",
                "windrichting": "",
                "temperatuur": 0,
                "weer_Omschrijving": "",
                "afvalData": all_trash
            }

            #send_to_api(data_to_send)

            print(data_to_send)

    return frame

def send_to_api(detected_object):
    url = "https://rommelkoningencontainer--rommelkoningencontainer.mangopebble-9c8f37e9.northeurope.azurecontainerapps.io/TrashdataSensoring"
    headers = {"Api-Key-Name": "G3vVhpsno58iXRV5pmtZKwhQqd160jSietseiseenlandverraderPlKI0o3v07lAKyX9tp8BqtoxVgBEkUrGWxvrrtMdYRcbECin45227w7XU82PfyuFjkGW13MGxkvKJLZZ9YEvFzBGRtsGXpIkyvxzWSWPPZXUS7orI73OIHN8NszwuZqS25siCTYl6XVpgAKGnta4LgOTwC9jSQOHpy8PU9dNNQ", "Content-Type": "application/json"}

    try:
        response = requests.post(url, headers=headers, json = detected_object)
        if response.status_code == 200 or response.status_code == 204:
            print("Sent to API:", detected_object)
        else:
            print("Failed to send. Status:", response.status_code, response.text)
    except Exception as e:
        print("Error sending to API:", e)

def webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam Error")
        exit

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame Read Error")
            exit

        annotated_frame = detect_and_annotate2(model, frame.copy())

        cv2.imshow("Frame: ", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(10)

    cap.release()
    cv2.destroyAllWindows()

    results = model(frame)

def oakDCam():
    camera = CameraOakD()
    with dai.Device(camera.pipeline) as device:
        q = device.getOutputQueue(name="video", maxSize=1, blocking=True)
        
        while True:
            frame = q.get().getCvFrame()
            annotated_frame = detect_and_annotate2(model, frame.copy())

            time.sleep(5)


#webcam()
oakDCam()
