import cv2
from keras.models import model_from_json
import numpy as np
from tensorflow.keras.preprocessing.image import load_img

# Load the model
json_file = open("emotion_model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotion_model.h5")

# Load the face detector
haar_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_file)


def extract_feats(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255


webcam = cv2.VideoCapture(0)
labels = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise",
}

while True:
    i, im = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    try:
        for p, q, r, s in faces:
            image = gray[q : q + s, p : p + r]
            cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)
            image = cv2.resize(image, (48, 48))
            image = extract_feats(image)
            prediction = model.predict(image)
            pred_lbl = labels[np.argmax(prediction)]
            print("Prediction: ", pred_lbl)
            cv2.putText(
                im,
                pred_lbl,
                (p, q - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (36, 255, 12),
                2,
            )
        cv2.imshow("Emotion Detector", im)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    except cv2.error:
        pass
