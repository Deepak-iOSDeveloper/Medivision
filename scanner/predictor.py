"""
predictor.py
Lazy-loads all 13 trained CNN models and exposes a single predict() function.

IMPORTANT: After training in your notebook, save each model like this:
    cnn_brain_cancer_tumor.save('ml_models/brain_cancer_tumor.h5')
    cnn_brain_tumor_not.save('ml_models/brain_tumor_not.h5')
    cnn_brain_tumor_type.save('ml_models/brain_tumor_type.h5')
    cnn_breast_tumor_type.save('ml_models/breast_tumor_type.h5')
    cnn_heart_desease.save('ml_models/heart_disease.h5')
    cnn_brain_cancer_2.save('ml_models/brain_cancer_2.h5')
    cnn_breast_cancer_2.save('ml_models/breast_cancer_2.h5')
    cnn_cervical_cancer.save('ml_models/cervical_cancer.h5')
    cnn_kidney_cancer.save('ml_models/kidney_cancer.h5')
    cnn_lung_cancer.save('ml_models/lung_cancer.h5')
    cnn_lymphoma_cancer.save('ml_models/lymphoma_cancer.h5')
    cnn_oral_cancer.save('ml_models/oral_cancer.h5')
    cnn_pneumonia_tuborculosis.save('ml_models/pneumonia_tuberculosis.h5')
Place all .h5 files inside the ml_models/ folder at project root.
"""

import os
import logging
import numpy as np
from threading import Lock
from django.conf import settings

logger = logging.getLogger(__name__)

# ── Model registry ─────────────────────────────────────────────────────────────
# key           → filename in MODELS_DIR
# mode          → 'binary' or 'categorical'
# classes       → list of class labels (must match folder order from ImageDataGenerator)
# severity      → per-class severity: 'safe', 'moderate', 'high', 'critical'
# description   → short clinical description shown to user

MODEL_REGISTRY = {
    'brain_cancer_tumor': {
        'file':      'brain_cancer_tumor.keras',
        'mode':      'categorical',
        'classes':   ['Brain Cancer', 'Brain Tumor', 'Normal'],
        'severity':  {'Brain Cancer': 'critical', 'Brain Tumor': 'high', 'Normal': 'safe'},
        'title':     'Brain Cancer / Tumor Detection',
        'organ':     'Brain',
        'icon':      '🧠',
        'desc':      'Detects presence of brain cancer or tumor from MRI scan.',
        'advice':    {
            'Brain Cancer': 'Immediate oncology consultation required. Do not delay.',
            'Brain Tumor':  'Neurological evaluation needed. Schedule MRI with contrast.',
            'Normal':       'No abnormality detected. Routine follow-up recommended.',
        },
    },
    'brain_tumor_not': {
        'file':      'brain_tumor_not.keras',
        'mode':      'binary',
        'classes':   ['No Tumor', 'Tumor Present'],
        'severity':  {'No Tumor': 'safe', 'Tumor Present': 'critical'},
        'title':     'Brain Tumor Screening (Yes / No)',
        'organ':     'Brain',
        'icon':      '🧠',
        'desc':      'Binary screening — determines whether a tumor is present.',
        'advice':    {
            'No Tumor':      'No tumor detected. Maintain regular check-ups.',
            'Tumor Present': 'Tumor detected. Urgent neurology referral required.',
        },
    },
    'brain_tumor_type': {
        'file':      'brain_tumor_type.keras',
        'mode':      'categorical',
        'classes':   ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'],
        'severity':  {'Glioma': 'critical', 'Meningioma': 'high', 'No Tumor': 'safe', 'Pituitary': 'high'},
        'title':     'Brain Tumor Type Classification',
        'organ':     'Brain',
        'icon':      '🧠',
        'desc':      'Classifies brain tumor type: Glioma, Meningioma, or Pituitary.',
        'advice':    {
            'Glioma':     'Gliomas are aggressive. Immediate neuro-oncology consultation needed.',
            'Meningioma': 'Usually benign but requires monitoring. Neurosurgery consultation advised.',
            'No Tumor':   'No tumor found in this scan.',
            'Pituitary':  'Pituitary adenoma detected. Endocrinology and neurology workup required.',
        },
    },
    'breast_tumor_type': {
        'file':      'breast_tumor_type.keras',
        'mode':      'categorical',
        'classes':   ['Benign', 'Malignant', 'Normal'],
        'severity':  {'Benign': 'moderate', 'Malignant': 'critical', 'Normal': 'safe'},
        'title':     'Breast Tumor Classification (Ultrasound)',
        'organ':     'Breast',
        'icon':      '🎗️',
        'desc':      'Classifies breast ultrasound as Benign, Malignant, or Normal.',
        'advice':    {
            'Benign':    'Benign mass detected. Follow-up ultrasound in 6 months recommended.',
            'Malignant': 'Malignant features detected. Immediate biopsy and oncology referral needed.',
            'Normal':    'No suspicious findings on ultrasound.',
        },
    },
    'heart_disease': {
        'file':      'heart_disease.keras',
        'mode':      'categorical',
        'classes':   ['COVID-19', 'Normal', 'Pneumonia', 'Viral Pneumonia'],
        'severity':  {'COVID-19': 'high', 'Normal': 'safe', 'Pneumonia': 'high', 'Viral Pneumonia': 'moderate'},
        'title':     'Heart / Chest Disease Detection',
        'organ':     'Heart & Chest',
        'icon':      '❤️',
        'desc':      'Analyses chest X-ray for COVID-19, Pneumonia, or Normal findings.',
        'advice':    {
            'COVID-19':        'COVID-19 findings on chest X-ray. Isolation and pulmonology consult needed.',
            'Normal':          'Chest X-ray appears normal.',
            'Pneumonia':       'Bacterial pneumonia suspected. Antibiotic therapy and follow-up required.',
            'Viral Pneumonia': 'Viral pneumonia pattern. Supportive care and monitoring recommended.',
        },
    },
    'brain_cancer_2': {
        'file':      'brain_cancer_2.keras',
        'mode':      'categorical',
        'classes':   ['Glioma', 'Meningioma', 'Pituitary'],
        'severity':  {'Glioma': 'critical', 'Meningioma': 'high', 'Pituitary': 'high'},
        'title':     'Brain Cancer Advanced Classification',
        'organ':     'Brain',
        'icon':      '🧠',
        'desc':      'Advanced multi-class brain cancer detection from large dataset (15K images).',
        'advice':    {
            'Glioma':     'High-grade glioma suspected. Immediate neuro-oncology referral.',
            'Meningioma': 'Meningioma identified. Surgical consultation may be required.',
            'Pituitary':  'Pituitary tumor detected. Hormonal workup and MRI with contrast advised.',
        },
    },
    'breast_cancer_2': {
        'file':      'breast_cancer_2.keras',
        'mode':      'binary',
        'classes':   ['Benign', 'Malignant'],
        'severity':  {'Benign': 'moderate', 'Malignant': 'critical'},
        'title':     'Breast Cancer Detection',
        'organ':     'Breast',
        'icon':      '🎗️',
        'desc':      'Binary breast cancer detection from histopathology images.',
        'advice':    {
            'Benign':    'Benign tissue. Continue regular mammogram screening.',
            'Malignant': 'Malignant tissue detected. Urgent surgical oncology consultation required.',
        },
    },
    'cervical_cancer': {
        'file':      'cervical_cancer.keras',
        'mode':      'categorical',
        'classes':   ['Dyskeratotic', 'Koilocytotic', 'Metaplastic', 'Parabasal', 'Superficial-Intermediate'],
        'severity':  {
            'Dyskeratotic':            'high',
            'Koilocytotic':            'high',
            'Metaplastic':             'moderate',
            'Parabasal':               'moderate',
            'Superficial-Intermediate':'safe',
        },
        'title':     'Cervical Cancer Cell Classification',
        'organ':     'Cervix',
        'icon':      '🔬',
        'desc':      'Classifies cervical cell types from Pap smear images (5 classes).',
        'advice':    {
            'Dyskeratotic':            'Dyskeratotic cells found. Colposcopy and biopsy recommended.',
            'Koilocytotic':            'HPV-associated changes detected. Gynaecology follow-up essential.',
            'Metaplastic':             'Metaplastic cells: normal transformation zone. Annual Pap smear advised.',
            'Parabasal':               'Parabasal cells: may indicate atrophy. Clinical correlation needed.',
            'Superficial-Intermediate':'Normal superficial cells. Routine annual screening.',
        },
    },
    'kidney_cancer': {
        'file':      'kidney_cancer.keras',
        'mode':      'binary',
        'classes':   ['Normal', 'Tumor'],
        'severity':  {'Normal': 'safe', 'Tumor': 'critical'},
        'title':     'Kidney Cancer / Tumor Detection',
        'organ':     'Kidney',
        'icon':      '🫘',
        'desc':      'Detects kidney tumour from CT or MRI scan images.',
        'advice':    {
            'Normal': 'No renal mass detected. Continue routine monitoring.',
            'Tumor':  'Renal mass detected. Urgent urology and oncology referral required.',
        },
    },
    'lung_cancer': {
        'file':      'lung_cancer.keras',
        'mode':      'categorical',
        'classes':   ['Colon Adenocarcinoma', 'Colon Benign', 'Lung Adenocarcinoma', 'Lung Benign', 'Lung Squamous'],
        'severity':  {
            'Colon Adenocarcinoma': 'critical',
            'Colon Benign':        'safe',
            'Lung Adenocarcinoma': 'critical',
            'Lung Benign':         'safe',
            'Lung Squamous':       'critical',
        },
        'title':     'Lung & Colon Cancer Classification',
        'organ':     'Lung & Colon',
        'icon':      '🫁',
        'desc':      'Histopathology-based classification of lung and colon tissue (5 classes).',
        'advice':    {
            'Colon Adenocarcinoma': 'Colon malignancy detected. Gastroenterology and oncology consultation.',
            'Colon Benign':        'Benign colon tissue. Routine colonoscopy follow-up.',
            'Lung Adenocarcinoma': 'Lung adenocarcinoma detected. Thoracic oncology referral urgently needed.',
            'Lung Benign':         'Benign lung tissue found.',
            'Lung Squamous':       'Squamous cell carcinoma. Chest physician and oncologist referral required.',
        },
    },
    'lymphoma_cancer': {
        'file':      'lymphoma_cancer.keras',
        'mode':      'categorical',
        'classes':   ['Chronic Lymphocytic Leukemia', 'Follicular Lymphoma', 'Mantle Cell Lymphoma'],
        'severity':  {
            'Chronic Lymphocytic Leukemia': 'high',
            'Follicular Lymphoma':          'high',
            'Mantle Cell Lymphoma':         'critical',
        },
        'title':     'Lymphoma Type Classification',
        'organ':     'Lymph Nodes',
        'icon':      '🔴',
        'desc':      'Classifies lymphoma subtype from histopathology images.',
        'advice':    {
            'Chronic Lymphocytic Leukemia': 'CLL detected. Haematology referral required for staging and treatment.',
            'Follicular Lymphoma':          'Follicular lymphoma identified. Oncology evaluation needed.',
            'Mantle Cell Lymphoma':         'Aggressive lymphoma subtype. Immediate haematology-oncology referral.',
        },
    },
    'oral_cancer': {
        'file':      'oral_cancer.keras',
        'mode':      'binary',
        'classes':   ['Normal', 'Oral Cancer'],
        'severity':  {'Normal': 'safe', 'Oral Cancer': 'critical'},
        'title':     'Oral Cancer Detection',
        'organ':     'Oral Cavity',
        'icon':      '🦷',
        'desc':      'Detects oral cancer from oral cavity images.',
        'advice':    {
            'Normal':      'No oral malignancy detected. Continue regular dental check-ups.',
            'Oral Cancer': 'Suspicious oral lesion detected. ENT/maxillofacial surgery referral required.',
        },
    },
    'pneumonia_tuberculosis': {
        'file':      'pneumonia_tuberculosis.h5',
        'mode':      'categorical',
        'classes':   ['Normal', 'Pneumonia', 'Tuberculosis'],
        'severity':  {'Normal': 'safe', 'Pneumonia': 'high', 'Tuberculosis': 'high'},
        'title':     'Pneumonia / Tuberculosis Detection',
        'organ':     'Lungs',
        'icon':      '🫁',
        'desc':      'Detects Pneumonia or Tuberculosis from chest X-ray.',
        'advice':    {
            'Normal':       'Chest X-ray appears clear.',
            'Pneumonia':    'Pneumonia pattern detected. Antibiotic therapy recommended.',
            'Tuberculosis': 'TB findings on X-ray. Pulmonology referral and DOTS therapy evaluation needed.',
        },
    },
}

# ── Thread-safe model cache ────────────────────────────────────────────────────
_model_cache = {}
_cache_lock  = Lock()


def _load_model(model_key: str):
    """Load a single .h5 model from MODELS_DIR. Returns model or None."""
    with _cache_lock:
        if model_key in _model_cache:
            return _model_cache[model_key]

        import tensorflow as tf
        info     = MODEL_REGISTRY[model_key]
        path     = os.path.join(settings.MODELS_DIR, info['file'])

        if not os.path.exists(path):
            logger.warning(f"Model file not found: {path}  (running in DEMO mode)")
            _model_cache[model_key] = None
            return None

        try:
            model = tf.keras.models.load_model(path)
            model.compile()          # suppress re-compile warnings
            _model_cache[model_key] = model
            logger.info(f"Loaded model: {model_key}")
            return model
        except Exception as e:
            logger.error(f"Failed to load {model_key}: {e}")
            _model_cache[model_key] = None
            return None


def predict(model_key: str, image_path: str) -> dict:
    """
    Run inference on a single image.

    Returns dict:
        predicted_class  : str
        confidence       : float (0–100)
        severity         : str  ('safe'|'moderate'|'high'|'critical')
        advice           : str
        all_scores       : list of (class_name, confidence_pct)
        demo_mode        : bool  (True when .h5 not found)
    """
    import numpy as np
    from tensorflow.keras.preprocessing import image as keras_image

    info    = MODEL_REGISTRY[model_key]
    classes = info['classes']
    model   = _load_model(model_key)

    # ── DEMO mode — random scores when model file not yet saved ───────────────
    if model is None:
        import random
        scores     = np.array([random.random() for _ in classes])
        scores     = scores / scores.sum()
        best_idx   = int(np.argmax(scores))
        best_class = classes[best_idx]
        return {
            'predicted_class': best_class,
            'confidence':      round(float(scores[best_idx]) * 100, 1),
            'severity':        info['severity'][best_class],
            'advice':          info['advice'][best_class],
            'all_scores':      [(c, round(float(s)*100,1)) for c, s in zip(classes, scores)],
            'demo_mode':       True,
        }

    # ── Real inference ─────────────────────────────────────────────────────────
    img    = keras_image.load_img(image_path, target_size=(64, 64))
    arr    = keras_image.img_to_array(img) / 255.0
    arr    = np.expand_dims(arr, axis=0)

    preds  = model.predict(arr, verbose=0)

    if info['mode'] == 'binary':
        prob_pos = float(preds[0][0])
        scores   = np.array([1 - prob_pos, prob_pos])
    else:
        scores   = preds[0]

    best_idx   = int(np.argmax(scores))
    best_class = classes[best_idx]

    return {
        'predicted_class': best_class,
        'confidence':      round(float(scores[best_idx]) * 100, 1),
        'severity':        info['severity'][best_class],
        'advice':          info['advice'][best_class],
        'all_scores':      [(c, round(float(s)*100,1)) for c, s in zip(classes, scores)],
        'demo_mode':       False,
    }
