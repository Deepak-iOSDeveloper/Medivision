import uuid
from django.db import models


class ScanResult(models.Model):
    SEVERITY_CHOICES = [
        ('safe',     'Safe'),
        ('moderate', 'Moderate'),
        ('high',     'High Risk'),
        ('critical', 'Critical'),
    ]

    id               = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    model_key        = models.CharField(max_length=64)
    scan_title       = models.CharField(max_length=200)
    image            = models.ImageField(upload_to='uploads/%Y/%m/%d/')
    predicted_class  = models.CharField(max_length=200)
    confidence       = models.FloatField()
    severity         = models.CharField(max_length=20, choices=SEVERITY_CHOICES)
    advice           = models.TextField()
    demo_mode        = models.BooleanField(default=False)
    created_at       = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.scan_title} → {self.predicted_class} ({self.confidence}%)"

    def severity_color(self):
        return {
            'safe':     '#22c55e',
            'moderate': '#f59e0b',
            'high':     '#ef4444',
            'critical': '#991b1b',
        }.get(self.severity, '#6b7280')

    def severity_label(self):
        return dict(self.SEVERITY_CHOICES).get(self.severity, 'Unknown')
