from django.db import models

class Detection(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    voice_file = models.FileField(upload_to='voices/', null=True, blank=True)
    face_image = models.ImageField(upload_to='faces/', null=True, blank=True)
    typing_data = models.TextField(null=True, blank=True)
    result = models.CharField(max_length=255, blank=True, null=True)
    confidence = models.FloatField(blank=True, null=True)
