from flask import Flask, request, jsonify
import cv2
import numpy as np
from keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
from mtcnn import MTCNN

app = Flask(__name__)

# Load the model
with open("../models/emotion_model.json", "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights("../models/emotion_model.h5")

# Load face detectors
haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
dnn_net = cv2.dnn.readNetFromCaffe(
    "models/deploy.prototxt", "models/res10_300x300_ssd_iter_140000.caffemodel"
)
detector = MTCNN()

LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise",
}


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Improve contrast
    return gray


def extract_feats(image):
    feature = cv2.GaussianBlur(image, (5, 5), 0)  # Reduce noise
    feature = img_to_array(feature)
    feature = feature.reshape(1, 48, 48, 1) / 255.0  # Normalize
    return feature


def detect_faces(image, method="dnn"):
    faces = []
    h, w = image.shape[:2]
    if method == "haar":
        gray = preprocess_image(image)
        faces = haar_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30)
        )
    elif method == "dnn":
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
        dnn_net.setInput(blob)
        detections = dnn_net.forward()
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                faces.append((x, y, x1 - x, y1 - y))
    elif method == "mtcnn":
        detected_faces = detector.detect_faces(image)
        for face in detected_faces:
            x, y, w, h = face["box"]
            faces.append((x, y, w, h))
    return faces


@app.route("/predict", methods=["POST"])
def make_pred():
    file = request.files["image"].read()
    npimg = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    faces = detect_faces(img, method="dnn")  # Change to "haar" or "mtcnn" for testing

    if not faces:
        return jsonify({"error": "No face detected"})

    for x, y, w, h in faces:
        face_img = img[y : y + h, x : x + w]
        face_img = cv2.resize(face_img, (48, 48))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = extract_feats(face_img)

        prediction = model.predict(face_img)
        pred_lbl = LABELS[np.argmax(prediction)]

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(
            img, pred_lbl, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2
        )

    cv2.imwrite("output.jpg", img)
    return jsonify({"prediction": pred_lbl})


if __name__ == "__main__":
    app.run(debug=True)
