import pandas as pd
import numpy as np

np.random.seed(42)

# Voice Dataset: 13 MFCC features
voice_data = []
for _ in range(30):
    row = np.round(np.random.rand(13), 4).tolist()
    label = np.random.choice(['healthy', 'parkinsons', 'dementia'])
    voice_data.append(row + [label])
voice_df = pd.DataFrame(voice_data, columns=[f'mfcc_{i}' for i in range(13)] + ['label'])
voice_df.to_csv("voice_data.csv", index=False)

# Typing Dataset: 5 features
typing_data = []
for _ in range(30):
    row = np.round(np.random.rand(5), 4).tolist()
    label = np.random.choice(['healthy', 'parkinsons', 'dementia'])
    typing_data.append(row + [label])
typing_df = pd.DataFrame(typing_data, columns=[f'typing_{i}' for i in range(5)] + ['label'])
typing_df.to_csv("typing_data.csv", index=False)

# Face Dataset: 4 features
face_data = []
for _ in range(30):
    row = np.round(np.random.rand(4), 4).tolist()
    label = np.random.choice(['healthy', 'parkinsons', 'dementia'])
    face_data.append(row + [label])
face_df = pd.DataFrame(face_data, columns=[f'face_{i}' for i in range(4)] + ['label'])
face_df.to_csv("face_data.csv", index=False)

print("✅ All datasets saved as CSV in current folder.")
