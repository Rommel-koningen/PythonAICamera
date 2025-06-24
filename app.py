import depthai as dai
import cv2

class CameraOakD:
    def __init__(self):
        self.pipeline = dai.Pipeline()
        cam = self.pipeline.createColorCamera()
        cam.setPreviewSize(640, 480)
        cam.setInterleaved(False)
        self.xout = self.pipeline.createXLinkOut()
        self.xout.setStreamName("video")
        cam.preview.link(self.xout.input)

    def capture_image(self, save_path="capture.jpg"):
        with dai.Device(self.pipeline) as device:
            q = device.getOutputQueue(name="video", maxSize=1, blocking=True)
            frame = q.get().getCvFrame()
            cv2.imwrite(save_path, frame)
            print(f"Frame opgeslagen als {save_path}")
            return save_path