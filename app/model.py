import os
import wandb
from loadotenv import load_env

load_env(file_loc='/workspaces/fruit-classifier-mlops-class/.env')
os.getenv('WANDB_API_KEY')

MODELS_DIR = 'models'

os.makedirs(MODELS_DIR, exist_ok=True)

def download_artifact():
    assert 'WANDB_API_KEY' in os.environ, "Please enter the wandb API key"

    wandb_org = os.environ.get('WANDB_ORG')
    wandb_project = os.environ.get('WANDB_PROJECT')
    wandb_model_name = os.environ.get('WANDB_MODEL_NAME')
    wandb_model_version = os.environ.get('WANDB_MODEL_VERSION')

    artifact_path = f"{wandb_org}/{wandb_project}/{wandb_model_name}:{wandb_model_version}"

    print(artifact_path)

download_artifact()