import cv2
import ultralytics
from ultralytics import YOLO

import depthai as dai
pipeline = dai.Pipeline()
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setPreviewSize(640, 480)

model_path = 'weights/best.pt'
model = YOLO(model_path)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Webcam Error")
    exit

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame Read Error")
        exit

    cv2.imshow("Frame: ", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

results = model(frame)

def send_to_api(detected_object):
    url = "http://localhost:5120/api/DetectedObjects"
    headers = {"Content-Type": "application/json"}

def detect_and_annotate(model, frame):
    results = model.predict(frame, conf=0.6)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{model.names[int(box.cls[0])]}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return frame