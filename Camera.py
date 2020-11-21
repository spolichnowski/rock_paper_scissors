from cv2 import cv2 as cv
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.backend import expand_dims
import numpy as np
import operator


lower_skin = np.array([0, 58, 30])
upper_skin = np.array([33, 255, 255])


class Camera:

    def __init__(self):
        # Initiate video capturing
        self.video = cv.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def take_picture(self, round):
        ret, frame = self.video.read()
        cv.imwrite('./choices/{}.jpeg'.format(round), frame)

    def get_video_capture(self):
        # Resizes frame, converts image into memory buffer and into bytes
        ret, frame = self.video.read()
        frame = cv.flip(frame, 1)
        h, w, l = frame.shape
        height = int(h/4)
        width = int(w/4)
        frame = cv.resize(frame, (width, height))
        pred_frame = cv.resize(frame, (300, 300))
        ret, jpeg = cv.imencode('.jpg', frame)
        return jpeg.tobytes()
