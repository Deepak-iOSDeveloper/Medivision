from huggingface_hub import HfApi, login
import os

login(token=os.getenv("HF_TOKEN"))  # enter your HF token from huggingface.co/settings/tokens

api = HfApi()

# Create a private repo
api.create_repo("DeepakLPU/medivision-models", private=True)

# Upload all 13 models
import os
for f in os.listdir("ml_models"):
    api.upload_file(
        path_or_fileobj=f"ml_models/{f}",
        path_in_repo=f,
        repo_id="DeepakLPU/medivision-models",
        repo_type="model",
    )
    print(f"Uploaded: {f}")