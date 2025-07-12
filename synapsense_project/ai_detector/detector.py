import os
import joblib
import numpy as np

BASE_DIR = os.path.dirname(__file__)
model_path = lambda name: os.path.join(BASE_DIR, 'models', name)

voice_model = joblib.load(model_path('voice_model.pkl'))
voice_encoder = joblib.load(model_path('voice_encoder.pkl'))

typing_model = joblib.load(model_path('typing_model.pkl'))
typing_encoder = joblib.load(model_path('typing_encoder.pkl'))

face_model = joblib.load(model_path('face_model.pkl'))
face_encoder = joblib.load(model_path('face_encoder.pkl'))

def run_synapsense(voice=None, face=None, typing=None):
    if voice is not None:
        pred = voice_model.predict([voice])[0]
        label = voice_encoder.inverse_transform([pred])[0]
        confidence = max(voice_model.predict_proba([voice])[0])
        return f"Voice model: {label}", round(confidence * 100, 2)
    elif face is not None:
        pred = face_model.predict([face])[0]
        label = face_encoder.inverse_transform([pred])[0]
        confidence = max(face_model.predict_proba([face])[0])
        return f"Face model: {label}", round(confidence * 100, 2)
    elif typing is not None:
        pred = typing_model.predict([typing])[0]
        label = typing_encoder.inverse_transform([pred])[0]
        confidence = max(typing_model.predict_proba([typing])[0])
        return f"Typing model: {label}", round(confidence * 100, 2)
    else:
        return "No valid input", 0.0
