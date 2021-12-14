import h5py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
# from tensorflow import keras

# from keras.models import load_model

#from tensorflow import keras 

from picamera.array import PiRGBArray
from picamera import PiCamera
import time


model = load_model('/home/pi/Proyecto-Final-ITT/9-Project-ITT-Neural-Network-autonomous-movement/Step2-Training/model.h5', compile = False)


def preProcess(img):
    img = img[54:120, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img


if __name__ == '__main__':
    
    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    rawCapture = PiRGBArray(camera)
    # allow the camera to warmup
    time.sleep(2)
    # grab an image from the camera
    camera.capture(rawCapture, format="bgr")
    image = rawCapture.array
    camera.close()
    
    img = np.asarray(image)
    img = preProcess(img)
    img = np.array([img])
    direction = float(model.predict(img))
    
    print(direction)
    
    print(round(direction))
  

