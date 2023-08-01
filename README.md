# LicensePlateRecognition


License plate recognition intended to be used for Nigerian license plates. Works using a YOLO(You Only Look Once) 
model that was fine tuned to detect United Kingdom license plates.

## Requirements

* ultralytics
* cv2
* a video containing cars with visible license plates

## Installation

The major player in the code is a fine tuned YOLO model therefore the ultralytics library needs to be installed first 
there are no special dependencies.

## Usage

1. To use the model, you need to have a video prepared that contains cars with visible license plates
1. In the main.py file, in the initialisation function of the LicensePlateRecognition class, the path string of the video must be replaces with the path to your video

## Sources

The basis of the project was formed from the understanding of how to use YOLO models.
One of the modules making the project work was also from the github repository linked below
* [https://github.com/computervisioneng/automatic-number-plate-recognition-python-yolov8]
* [https://youtu.be/fyJB1t0o0ms]

## ToDo List

- [ ] Alter the util class to comply with the format of Nigerian plates
- [ ] Clean up the code, removing the parts that are no longer useful
- [ ] Add functionality to make the number plate readings more accurate
