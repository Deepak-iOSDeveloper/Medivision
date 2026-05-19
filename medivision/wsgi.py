import os
from django.core.wsgi import get_wsgi_application
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'medivision.settings')
application = get_wsgi_application()

# Download models if not present
from scanner.model_downloader import download_models
download_models()
