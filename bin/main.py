from ultralytics import YOLO
import cv2
import numpy as np

import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv

from threading import Thread


class LicensePlateRecognition:
    def __init__(self):
        self.tracker = Sort()
        self.license_plate_detector = YOLO('./models/license_plate_detector.pt')
        self.car_detector = YOLO('yolov8s.pt')

        self.vid_stream = cv2.VideoCapture('./sample.mp4')
        self.main_frame = "License Plate Recognition"

        self.eligible_vehicles = [2, 3, 5, 7]
        self.detections = []
        self.last_frame = None
        self.running = True

        self.video_stream_thread = Thread(target=self.video_stream)
        self.car_detection_thread = Thread(target=self.car_recognition)
        self.video_stream_thread.start()
        self.car_detection_thread.start()

    def video_stream(self):
        while self.running:
            ret, frame = self.vid_stream.read()
            self.last_frame = frame

            if not ret:
                self.running = False
                break

            for detection in self.detections:
                x1, y1, x2, y2, id = detection
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                frame = cv2.putText(
                    frame,
                    f"Id {int(id)}",
                    (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

            frame = cv2.resize(frame, (1280, 720))
            cv2.imshow(self.main_frame, frame)

            if cv2.waitKey(1) == ord('q'):
                self.running = False
                break

        print("Video stream ended")

    def car_recognition(self):
        while self.running:
            if self.last_frame is None:
                continue
            detections = self.car_detector(self.last_frame, verbose=False)[0]

            self.detections = []
            # boxes = detections[0].boxes.xyxy.cpu().numpy().astype(int)
            # ids = detections[0].boxes.id.cpu().numpy().astype(int)

            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in self.eligible_vehicles:
                    self.detections.append([x1, y1, x2, y2, class_id])


if __name__ == '__main__':
    rec = LicensePlateRecognition()
