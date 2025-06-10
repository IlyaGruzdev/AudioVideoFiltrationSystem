import cv2
from collections import deque, defaultdict
from app.video_recognition.video_processed.face_detection import FaceDetector
from app.video_recognition.video_processed.age_detection import AgePredictor
import numpy as np

class VideoFrameProcessor:
    def __init__(self, min_age = 7):
        self.face_detector = FaceDetector()
        self.age_predictor = AgePredictor()

        self.face_data = defaultdict(lambda: {"coords": None, "ages": deque(maxlen=30)})
        self.face_id_counter = 0
        self.threshold_distance = 50
        self.min_age = min_age
    def process(self, frame):
            boxes = self.face_detector.detect_faces(frame)
            result_frame = frame.copy()

            for box in boxes:
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # ID assignment
                face_id = None
                min_dist = float('inf')
                for fid, data in self.face_data.items():
                    prev = data["coords"]
                    if prev:
                        pcx, pcy = (prev[0] + prev[2]) // 2, (prev[1] + prev[3]) // 2
                        dist = ((cx - pcx)**2 + (cy - pcy)**2)**0.5
                        if dist < min_dist:
                            min_dist = dist
                            face_id = fid
                if face_id is None or min_dist > self.threshold_distance:
                    self.face_id_counter += 1
                    face_id = self.face_id_counter

                self.face_data[face_id]["coords"] = (x1, y1, x2, y2)
                face_img = frame[y1:y2, x1:x2]

                if face_img.size == 0:
                    continue

                age = self.age_predictor.predict_age(face_img)
                self.face_data[face_id]["ages"].append(age)
                avg_age = round(np.mean(self.face_data[face_id]["ages"]))

                # 🧒 Замазываем лицо если < 7 лет по дефолту
                if avg_age < self.min_age:
                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
                else:
                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(result_frame, f"ID:{face_id}, Age:{avg_age}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            return result_frame
    
         