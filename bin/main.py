import sys
import time

from ultralytics import YOLO
import cv2

# import numpy as np
# import util
# from sort.sort import *

from util import read_license_plate  # , write_csv, get_car

from threading import Thread


class LicensePlateRecognition:
    """
    Class that takes care of handling the video stream and recognising cars

    Attributes
    ----------

    Methods
    -------
    video_stream()
        Reads frames from a video stream, draws bounding boxes and displays the frame

    car_recognition()
        Gets the locations of the license plates in the image and reads from the plates

    --- Both methods run independently of one another, the car_recognition pulls frames when it is ready
    """

    def __init__(self):
        # Variables for models and tracker
        # self.tracker = Sort()
        self.license_plate_detector = YOLO('./models/license_plate_detector.pt')
        self.car_detector = YOLO('yolov8s.pt')

        # variables for display
        self.vid_stream = cv2.VideoCapture('./sample.mp4')
        self.main_frame = "License Plate Recognition"

        # self.eligible_vehicles = [2, 3, 5, 7]
        self.detections = []
        self.latest_frame = None

        # Variables for image collection and processing
        self.plate_image = None
        self.plate_images = []
        self.image_size_up = 50
        self.image_scale = 2

        # Variables for logging and benchmarking
        self.last_time = time.time()
        self.running = True

        # Variables for handling threads
        self.video_stream_thread = Thread(target=self.video_stream)
        self.car_detection_thread = Thread(target=self.car_recognition)
        self.video_stream_thread.start()
        self.car_detection_thread.start()

    def video_stream(self):
        while self.running:
            ret, frame = self.vid_stream.read()
            self.latest_frame = frame.copy()

            if not ret:
                self.running = False
                break

            frame = cv2.resize(frame, (640, 360))
            cv2.imshow(self.main_frame, frame)
            if self.plate_image is not None:
                cv2.imshow("temp Frame", self.plate_image)

            if cv2.waitKey(1) == ord('q'):
                self.running = False
                break

        print("Video stream ended")
        self.running = False
        sys.exit()

    def car_recognition(self):
        while self.running:
            if self.latest_frame is None:
                continue
            detections = self.license_plate_detector(self.latest_frame, verbose=False)[0]

            frame = self.latest_frame
            self.detections = []

            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection

                self.detections.append([x1, y1, x2, y2, class_id])
                try:
                    plate_image = frame[int(y1 - self.image_size_up):int(y2 + self.image_size_up),
                                        int(x1 - self.image_size_up):int(x2 + self.image_size_up)]
                except Exception as e:
                    print(f"Exception: {e}")
                    plate_image = frame[int(y1 - self.image_size_up/2):int(y2 + self.image_size_up/2),
                                        int(x1 - self.image_size_up/2):int(x2 + self.image_size_up/2)]

                # scale license plate
                # plate_image = cv2.resize(plate_image, (int(plate_image.shape[1] * self.image_scale),
                #                                        int(plate_image.shape[0] * self.image_scale)))

                # process license plate
                plate_img = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
                _, plate_img = cv2.threshold(plate_img, 64, 255, cv2.THRESH_BINARY_INV)

                # read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(plate_img)

                if license_plate_text is not None:
                    print(license_plate_text)
                    with open('license_plates.txt', 'r+') as file:
                        licenses = file.read()
                        # print(licenses)
                        if license_plate_text not in licenses:
                            file.write(license_plate_text + "\n")
                            print("Written to file")

            # now = time.time()
            # print(f"Time: {now - self.last_time} secs")
            # self.last_time = now


if __name__ == '__main__':
    rec = LicensePlateRecognition()
