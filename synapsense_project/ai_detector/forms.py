from django import forms
from .models import Detection

class DetectionForm(forms.ModelForm):
    class Meta:
        model = Detection
        fields = ['voice_file', 'face_image', 'typing_data']
