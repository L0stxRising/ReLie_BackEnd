from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import mediapipe as mp
import cv2 as cv
from scipy.spatial import distance as dist
from collections import Counter
import os
import tempfile
import joblib
from fer import FER
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# Load your encoders and scalers
le1 = joblib.load("le_emotion.pkl")
ss = joblib.load("scaler.pkl")
model = load_model("relie_model.h5")  # Assuming this is your final model
fer_detector = FER()
haar_cascade = cv.CascadeClassifier(r"C:\PYTHON\OPENCV\HaarCascades\Haar_face.xml")

# FaceMesh init
facemesh = mp.solutions.face_mesh
faceMesh = facemesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                             refine_landmarks=True, min_detection_confidence=0.5,
                             min_tracking_confidence=0.5)


def extract_features(landmarks, shape, frame):
    def crop_face(f):
        faces = haar_cascade.detectMultiScale(f, 1.1, minNeighbors=3)
        for (x, y, w, h) in faces:
            return f[y:y + h, x:x + w]
        return f

    frame = crop_face(frame)
    h, w = shape

    def px(idx): return (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
    def eye_aspect_ratio(top, bottom, left, right):
        return dist.euclidean(top, bottom) / dist.euclidean(left, right)

    right_eye = eye_aspect_ratio(px(159), px(145), px(33), px(133))
    left_eye = eye_aspect_ratio(px(386), px(374), px(263), px(362))
    blink = int(((right_eye + left_eye) / 2) < 0.20)
    lip_gap = dist.euclidean(px(13), px(14))
    brow_gap = dist.euclidean(px(105), px(66))

    emotions = fer_detector.detect_emotions(frame)
    if emotions and "emotions" in emotions[0]:
        emotion = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
    else:
        emotion = "neutral"

    return emotion, blink, lip_gap, brow_gap


def summarize_window(features, emotion):
    features = np.array(features)
    return {
        "blink_sum": np.sum(features[:, 0]),
        "blink_mean": np.mean(features[:, 0]),
        "lipgap_mean": np.mean(features[:, 1]),
        "lipgap_max": np.max(features[:, 1]),
        "lipgap_std": np.std(features[:, 1]),
        "browgap_mean": np.mean(features[:, 2]),
        "browgap_std": np.std(features[:, 2]),
        "emotion": emotion
    }


@app.route("/")
def home():
    return "ReLie video API is live!"


@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    file.save(temp_path)

    vidcap = cv.VideoCapture(temp_path)
    fdata = []
    window, swindow = [], []

    while True:
        isTrue, frame = vidcap.read()
        if not isTrue:
            if len(window) > 0:
                emo = Counter(swindow).most_common(1)[0][0] if swindow else "neutral"
                summarized = summarize_window(window, emo)
                fdata.append(summarized)
            break

        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        coordsobj = faceMesh.process(frame_rgb)

        if coordsobj.multi_face_landmarks:
            for coords in coordsobj.multi_face_landmarks:
                emotion, *features = extract_features(coords.landmark, frame.shape[:2], frame_rgb)
                swindow.append(emotion)
                window.append(features)

                if len(window) == 30:
                    emo = Counter(swindow).most_common(1)[0][0] if swindow else "neutral"
                    summarized = summarize_window(window, emo)
                    fdata.append(summarized)
                    window, swindow = [], []
                break

    df = pd.DataFrame(fdata)

    if not df.empty and 'emotion' in df.columns:
        df['emotion'] = le1.transform(df['emotion'])
        df_scaled = ss.transform(df)

        pred = model.predict(df_scaled)
        pred_class = (pred > 0.5).astype(int)
        final = int(Counter(pred_class.flatten()).most_common(1)[0][0])

        return jsonify({
            "result": "Lie" if final == 1 else "Truth",
            "frames_analyzed": len(df),
            "raw_predictions": pred_class.flatten().tolist()
        })

    else:
        return jsonify({"error": "No face or emotion data detected"}), 400


if __name__ == "__main__":
    app.run(debug=True)