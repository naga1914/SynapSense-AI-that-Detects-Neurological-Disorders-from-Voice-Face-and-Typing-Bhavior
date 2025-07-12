from django.shortcuts import render, redirect, get_object_or_404
from .models import Detection
from .detector import run_synapsense
from django.core.files.base import ContentFile
import base64

def decode_base64_file(data, name):
    format, imgstr = data.split(';base64,')
    ext = format.split('/')[-1]
    return ContentFile(base64.b64decode(imgstr), name=f'{name}.{ext}')

def upload_detect(request):
    if request.method == 'POST':
        face_data = request.POST.get('face_image')
        voice_data = request.POST.get('voice_file')
        typing_data = request.POST.get('typing_data')

        if not (face_data or voice_data or typing_data):
            return render(request, 'ai_detector/upload.html', {
                'error': "Please provide at least one input."
            })

        det = Detection.objects.create(typing_data=typing_data)

        if face_data:
            det.face_image = decode_base64_file(face_data, 'face')
        if voice_data:
            det.voice_file = decode_base64_file(voice_data, 'voice')

        det.save()

        voice_path = det.voice_file.path if det.voice_file else None
        face_path = det.face_image.path if det.face_image else None

        prediction, confidence = run_synapsense(
            voice_path=voice_path,
            face_path=face_path,
            typing_data=typing_data
        )

        det.result = prediction
        det.confidence = confidence
        det.save()
        return redirect('result', det.id)

    return render(request, 'ai_detector/upload.html')

def view_result(request, pk):
    detection = get_object_or_404(Detection, pk=pk)
    return render(request, 'ai_detector/result.html', {'detection': detection})
