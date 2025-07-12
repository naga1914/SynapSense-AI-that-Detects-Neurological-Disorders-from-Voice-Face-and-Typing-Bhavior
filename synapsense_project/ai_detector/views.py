from django.shortcuts import render
from .detector import run_synapsense

def upload_detect(request):
    if request.method == 'POST':
        face_data = request.POST.get('face_image')
        voice_data = request.POST.get('voice_file')
        typing_text = request.POST.get('typing_data')

        voice_vector = [0.2] * 13 if voice_data else None
        face_vector = [0.3] * 4 if face_data else None
        typing_vector = [0.4] * 5 if typing_text else None

        prediction, confidence = run_synapsense(
            voice=voice_vector,
            face=face_vector,
            typing=typing_vector
        )

        return render(request, 'ai_detector/result.html', {
            'detection': {
                'result': prediction,
                'confidence': confidence * 100
            }
        })

    return render(request, 'ai_detector/upload.html')
