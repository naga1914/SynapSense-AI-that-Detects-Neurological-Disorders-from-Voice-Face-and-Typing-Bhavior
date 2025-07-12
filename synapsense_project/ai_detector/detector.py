import random

voice_predictions = [
    ("Voice indicates early Parkinson’s signs", 0.85),
    ("Voice shows signs of tremor instability", 0.81),
    ("Speech irregularities consistent with Parkinson's", 0.83),
    ("Voice pitch patterns suggest early dementia", 0.78),
    ("Normal voice modulation detected", 0.92)
]

face_predictions = [
    ("Facial analysis suggests neurological fatigue", 0.79),
    ("Eye movement irregularity detected", 0.76),
    ("Facial muscle stiffness observed", 0.82),
    ("Blink rate below normal range", 0.74),
    ("No signs of facial motor issues", 0.91)
]

typing_predictions = [
    ("Typing behavior shows possible motor issues", 0.82),
    ("Key latency abnormal — possible early Parkinson's", 0.84),
    ("Typing pattern matches healthy range", 0.90),
    ("Inconsistent keypress durations observed", 0.77),
    ("Typing rhythm suggests possible neurological disorder", 0.79)
]

def run_synapsense(voice_path=None, face_path=None, typing_data=None):
    if voice_path:
        return random.choice(voice_predictions)
    elif face_path:
        return random.choice(face_predictions)
    elif typing_data:
        return random.choice(typing_predictions)
    else:
        return "No valid input provided", 0.0
