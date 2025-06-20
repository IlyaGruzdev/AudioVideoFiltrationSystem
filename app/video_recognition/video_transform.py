from aiortc import MediaStreamTrack
from app.video_recognition.main import VideoFrameProcessor
from av import  VideoFrame
import numpy as np
import cv2

faces = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
eyes = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_eye.xml")
smiles = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_smile.xml")

class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track, transform):
        super().__init__()
        self.track = track
        self.transform = transform
        self.frame_processor = None
        if(transform == "age_detection"):
            self.frame_processor = VideoFrameProcessor()

    async def recv(self):
        frame = await self.track.recv()

        # Явно указываем формат как BGR для OpenCV
        if self.transform == "age_detection":
            age_detection_transform_frame = self.cv_transform(frame)
            return age_detection_transform_frame
        if self.transform == "cartoon":
            cartoon_transform_frame = self.cv_transform(frame)
            return cartoon_transform_frame
        elif self.transform == "edges":
            edges_transform_frame = self.cv_transform(frame)
            return edges_transform_frame
        elif self.transform == "rotate":
            rotate_transform_frame = self.cv_transform(frame)
            return rotate_transform_frame
        elif self.transform == "cv":
            cv_transform_frame = self.cv_transform(frame)
            return cv_transform_frame
        else:
            return frame
    def age_detection_transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        processed = self.frame_processor.process(img)
        new_frame = VideoFrame.from_ndarray(processed, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame
    def cartoon_transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

            # prepare color
        img_color = cv2.pyrDown(cv2.pyrDown(img))
        for _ in range(6):
            img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
        img_color = cv2.pyrUp(cv2.pyrUp(img_color))

            # prepare edges
        img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_edges = cv2.adaptiveThreshold(
            cv2.medianBlur(img_edges, 7),
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            9,
            2,
        )
        img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

         # combine color and edges
        img = cv2.bitwise_and(img_color, img_edges)

         # rebuild a VideoFrame, preserving timing information
        new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame
    def edges_transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

            # rebuild a VideoFrame, preserving timing information
        new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame
    def rotate_transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rows, cols, _ = img.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), frame.time * 45, 1)
        img = cv2.warpAffine(img, M, (cols, rows))

        new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame
    def cv_transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        face = faces.detectMultiScale(img, 1.1, 19)
        for (x, y, w, h) in face:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        eye = eyes.detectMultiScale(img, 1.1, 19)
        for (x, y, w, h) in eye:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame    
