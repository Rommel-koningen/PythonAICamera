from asyncio.windows_events import NULL
import cv2
import ultralytics
from ultralytics import YOLO
import requests
import json
import time

import depthai as dai
pipeline = dai.Pipeline()
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setPreviewSize(640, 480)

model_path = 'weights/best.pt'
model = YOLO(model_path)

trashTypes = ['biodegradable', 'cardboard', 'glass', 'metal', 'paper', 'plastic']

def detect_and_annotate(model, frame):
    results = model.predict(frame, conf=0.6)
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = trashTypes[cls_id]

            # Draw box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 2)

            # Create JSON data
            detected_data = {
                "label": label,
                "coordinates": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                }
            }

            data_to_send = {
                "foto_Id": "",
                "datum_En_Tijd": "2025-06-17T10:12:13.790Z",
                "camera_Naam": "Cam1",
                "longitude": 0,
                "latitude": 0,
                "postcode": "",
                "windrichting": "",
                "temperatuur": 0,
                "weer_Omschrijving": "",
                "afvalData": [
                    {
                        "afval_Id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                        "afval_Type": "cardboard",
                        "confidence": 0.70
                    },
                    {
                        "afval_Id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                        "afval_Type": "biodegradable",
                        "confidence": 0.78
                    }
            ]}

            # Send to API
            send_to_api(detected_data)

    return frame

def detect_and_annotate2(model, frame):
    results = model.predict(frame, conf=0.6)
    
    if results != NULL:

        all_trash = []

        all_trash.append({
            "afval_Id": "",
            "afval_Type": "Test",
            "confidence": 0
        })

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label = trashTypes[cls_id]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                print("Categorie-ID: ", cls_id)
                print("Categorie-naam: ", label)
                # confi = model.confidence

                all_trash.append({
                    "afval_Id": "",
                    "afval_Type": label,
                    "confidence": 0
                })

        if all_trash != []:
            data_to_send = {
                "foto_Id": "",
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

            send_to_api(data_to_send)

            print(data_to_send)

    return frame

def send_to_api(detected_object):
    url = "http://localhost:5120/TrashdataSensoring"
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, headers=headers, data=json.dumps(detected_object))
        if response.status_code == 200:
            print("Sent to API:", detected_object)
        else:
            print("Failed to send. Status:", response.status_code, response.text)
    except Exception as e:
        print("Error sending to API:", e)

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

    time.sleep(5)

cap.release()
cv2.destroyAllWindows()

results = model(frame)
