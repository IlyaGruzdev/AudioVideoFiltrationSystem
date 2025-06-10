import cv2
import os
cur_path = os.path.dirname(os.path.abspath(__file__))
class FaceDetector:
    def __init__(self, model_path=cur_path+"/models/opencv_face_detector_uint8.pb", proto_path=cur_path+"/models/opencv_face_detector.pbtxt", conf_threshold=0.7):
        self.net = cv2.dnn.readNet(model_path, proto_path)
        self.conf_threshold = conf_threshold

    def detect_faces(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
        self.net.setInput(blob)
        detections = self.net.forward()
        boxes = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.conf_threshold:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                boxes.append((x1, y1, x2, y2))
        return boxes
