import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), 'datasets')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Train VOICE model
voice_df = pd.read_csv(os.path.join(DATA_DIR, 'voice_data.csv'))
X_voice = voice_df.drop('label', axis=1)
y_voice = LabelEncoder().fit_transform(voice_df['label'])
joblib.dump(LabelEncoder().fit(voice_df['label']), os.path.join(MODEL_DIR, 'voice_encoder.pkl'))

Xv_train, Xv_test, yv_train, yv_test = train_test_split(X_voice, y_voice, test_size=0.2)
voice_model = RandomForestClassifier()
voice_model.fit(Xv_train, yv_train)
joblib.dump(voice_model, os.path.join(MODEL_DIR, 'voice_model.pkl'))
print("✅ Voice model saved")

# Train TYPING model
typing_df = pd.read_csv(os.path.join(DATA_DIR, 'typing_data.csv'))
X_typing = typing_df.drop('label', axis=1)
y_typing = LabelEncoder().fit_transform(typing_df['label'])
joblib.dump(LabelEncoder().fit(typing_df['label']), os.path.join(MODEL_DIR, 'typing_encoder.pkl'))

Xt_train, Xt_test, yt_train, yt_test = train_test_split(X_typing, y_typing, test_size=0.2)
typing_model = LogisticRegression(max_iter=1000)
typing_model.fit(Xt_train, yt_train)
joblib.dump(typing_model, os.path.join(MODEL_DIR, 'typing_model.pkl'))
print("✅ Typing model saved")

# Train FACE model
face_df = pd.read_csv(os.path.join(DATA_DIR, 'face_data.csv'))
X_face = face_df.drop('label', axis=1)
y_face = LabelEncoder().fit_transform(face_df['label'])
joblib.dump(LabelEncoder().fit(face_df['label']), os.path.join(MODEL_DIR, 'face_encoder.pkl'))

Xf_train, Xf_test, yf_train, yf_test = train_test_split(X_face, y_face, test_size=0.2)
face_model = DecisionTreeClassifier()
face_model.fit(Xf_train, yf_train)
joblib.dump(face_model, os.path.join(MODEL_DIR, 'face_model.pkl'))
print("✅ Face model saved")

print("🎉 All models and encoders are trained and saved!")
