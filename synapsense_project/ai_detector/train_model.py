# ai_detector/train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv('ai_detector/data/neurological_disorders_200.csv')

# Feature and label separation
X = df[['mean_pitch', 'jitter', 'shimmer', 'mfcc_1', 'mfcc_2', 'mfcc_3']]
y = LabelEncoder().fit_transform(df['diagnosis'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save model
joblib.dump(clf, 'ai_detector/model.pkl')
print("✅ Model trained and saved as model.pkl")
