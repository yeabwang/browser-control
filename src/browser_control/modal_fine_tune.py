"""Optional Modal launcher for the GPU-agnostic fine-tuning function."""

import os

import modal

from .config import FineTuningConfig
from .fine_tune import fine_tune
from .modal_infra import get_docker_image, get_retries, get_secrets, get_volume

app = modal.App("browser-control-fine-tune-with-grpo")
image = get_docker_image().env({
    "HF_HOME": "/hf_model_cache",
    "MODEL_CHECKPOINTS_DIR": "/model_checkpoints",
})
hf_models_volume = get_volume("hf-model-cache")
model_checkpoints_volume = get_volume("browser-control-fine-tune-with-grpo")


@app.function(
    image=image,
    gpu=os.environ.get("MODAL_GPU", "A100"),
    volumes={
        "/hf_model_cache": hf_models_volume,
        "/model_checkpoints": model_checkpoints_volume,
    },
    secrets=get_secrets(),
    timeout=2 * 60 * 60,
    retries=get_retries(max_retries=1),
    max_inputs=1,
)
def run_fine_tune(config: FineTuningConfig) -> None:
    fine_tune(config)


@app.local_entrypoint()
def main(config_file_name: str) -> None:
    config = FineTuningConfig.from_yaml(file_name=config_file_name)
    run_fine_tune.remote(config=config)
    print("Fine-tuning job completed successfully!")
