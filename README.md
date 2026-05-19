# MediVision AI — Django Medical Diagnostic Platform

13 CNN scanners embedded into a full Django web application.

## Setup

---
<i style="color:red">First Run the image_analysis.ipynb file - it will train the model using our Medical Dataset Folder which contain all the image files</i>
## IMPORTANT: Saving Your Trained Models

At the end of your notebook, add these lines to save all 13 CNNs:

```python
import os
os.makedirs('ml_models', exist_ok=True)

cnn_brain_cancer_tumor.save('ml_models/brain_cancer_tumor.keras')
cnn_brain_tumor_not.save('ml_models/brain_tumor_not.keras')
cnn_brain_tumor_type.save('ml_models/brain_tumor_type.keras')
cnn_breast_tumor_type.save('ml_models/breast_tumor_type.keras')
cnn_heart_desease.save('ml_models/heart_disease.keras')
cnn_brain_cancer_2.save('ml_models/brain_cancer_2.keras')
cnn_breast_cancer_2.save('ml_models/breast_cancer_2.keras')
cnn_cervical_cancer.save('ml_models/cervical_cancer.keras')
cnn_kidney_cancer.save('ml_models/kidney_cancer.keras')
cnn_lung_cancer.save('ml_models/lung_cancer.keras')
cnn_lymphoma_cancer.save('ml_models/lymphoma_cancer.keras')
cnn_oral_cancer.save('ml_models/oral_cancer.keras')
cnn_pneumonia_tuborculosis.save('ml_models/pneumonia_tuberculosis.keras')
```

Copy the `ml_models/` folder into the root of this Django project.

```bash
unzip medivision.zip && cd medivision
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

Visit: http://127.0.0.1:8000

---

## Project Structure
```
medivision/
├── manage.py
├── requirements.txt
├── ml_models/          ← place your .h5 files here
├── medivision/         ← Django config
│   ├── settings.py
│   └── urls.py
└── scanner/            ← main app
    ├── predictor.py    ← loads all 13 models + runs inference
    ├── models.py       ← ScanResult DB model
    ├── views.py        ← all views
    ├── urls.py
    └── templates/scanner/
        ├── base.html
        ├── dashboard.html
        ├── scan.html       ← upload + live result
        ├── result.html     ← full report
        └── history.html
```

## Demo Mode
If `.keras` files are not found, the system runs in **Demo Mode** — all 13 scanners work with simulated random predictions so you can preview the UI.

## All 13 Scanners
| # | Scanner | Classes |
|---|---------|---------|
| 1 | Brain Cancer / Tumor | 3 |
| 2 | Brain Tumor Yes/No | 2 (binary) |
| 3 | Brain Tumor Type | 4 (Glioma, Meningioma, Pituitary, None) |
| 4 | Breast Tumor (Ultrasound) | 3 |
| 5 | Heart / Chest Disease | 4 |
| 6 | Brain Cancer Advanced | 3 |
| 7 | Breast Cancer Detection | 2 (binary) |
| 8 | Cervical Cancer Cells | 5 |
| 9 | Kidney Cancer | 2 (binary) |
| 10 | Lung & Colon Cancer | 5 |
| 11 | Lymphoma Type | 3 |
| 12 | Oral Cancer | 2 (binary) |
| 13 | Pneumonia / Tuberculosis | 3 |
