# MediVision AI — Django Medical Diagnostic Platform (live on - https://medivision-7gaa.onrender.com/)

13 CNN scanners embedded into a full Django web application.

## Setup

```bash
unzip medivision.zip && cd medivision
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

THEN RUN THIS URL IN YOUR LOCAL MACHINE : http://127.0.0.1:8000

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
