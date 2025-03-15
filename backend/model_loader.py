import cv2
import numpy as np
from keras.models import model_from_json


# Load the trained model
def load_emotion_model():
    with open("../models/emotion_model.json", "r") as json_file:
        model = model_from_json(json_file.read())
    model.load_weights("../models/emotion_model.h5")
    return model


# Load face detector
def load_face_detector():
    haar_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(haar_file)


# Emotion labels
LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise",
}
