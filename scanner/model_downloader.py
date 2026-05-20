import os
from django.conf import settings

HF_REPO   = os.getenv("HF_REPO", "DeepakLPU/medivision-models")
HF_TOKEN  = os.getenv("HF_TOKEN")   # set this as env variable on Railway

MODEL_FILES = [
    "brain_cancer_tumor.keras",
    "brain_tumor_not.keras",
    "brain_tumor_type.keras",
    "breast_tumor_type.keras",
    "heart_disease.keras",
    "brain_cancer_2.keras",
    "breast_cancer_2.keras",
    "cervical_cancer.keras",
    "kidney_cancer.keras",
    "lung_cancer.keras",
    "lymphoma_cancer.keras",
    "oral_cancer.keras",
    "pneumonia_tuberculosis.keras",
]



def download_models():
    from huggingface_hub import hf_hub_download
    import sys

    os.makedirs(settings.MODELS_DIR, exist_ok=True)
    print("=== Starting model download check ===", flush=True)
    print(f"MODELS_DIR: {settings.MODELS_DIR}", flush=True)
    print(f"HF_REPO: {HF_REPO}", flush=True)
    print(f"HF_TOKEN set: {bool(HF_TOKEN)}", flush=True)

    for filename in MODEL_FILES:
        dest = os.path.join(settings.MODELS_DIR, filename)
        if os.path.exists(dest):
            print(f"✅ Already exists: {filename}", flush=True)
            continue
        print(f"⬇️  Downloading: {filename}", flush=True)
        hf_hub_download(
            repo_id=HF_REPO,
            filename=filename,
            repo_type="model",
            token=HF_TOKEN,
            local_dir=settings.MODELS_DIR,
        )
        print(f"✅ Done: {filename}", flush=True)

    print("=== Model download check complete ===", flush=True)