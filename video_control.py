import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
import numpy as np
import matplotlib.pyplot as plt
import cv2


class MyWebcam(object):
    def __init__(self, camera_index=0) -> None:
        super().__init__()
        self.snap = cv2.VideoCapture(camera_index)
        _, self.frame = self.snap.read()


if __name__ == '__main__':

    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        cv2.imshow('Input', frame)

        c = cv2.waitKey(1)
        if c == 27:
            break

    cv2.destroyAllWindows()
