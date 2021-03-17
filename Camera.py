from cv2 import cv2 as cv
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.backend import expand_dims
import numpy as np
import operator

model_path = './model/'
model = load_model(model_path)


def predict(frame):
    frame = img_to_array(frame, dtype='float32')
    prediction = model.predict(expand_dims(frame,  axis=0))
    prediction = prediction[0]
    predictions = {}
    predictions['Rock!'] = prediction[0]
    predictions['Paper!'] = prediction[1]
    predictions['Scissors!'] = prediction[2]

    predicted = max(predictions.items(), key=operator.itemgetter(1))[0]
    return predicted


class Camera:
    '''
    Camera class. Uses openCV to capture frame
    from the camera
    '''
    def __init__(self):
        # Initiate video capturing. Change to other number
        # in case of using external camera
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
        predict(pred_frame)
        ret, jpeg = cv.imencode('.jpg', frame)
        return jpeg.tobytes()
