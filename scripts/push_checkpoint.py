import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload a saved model checkpoint to Hugging Face Hub."
    )
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument(
        "--repo-id", required=True, help="Hugging Face repo, e.g. username/model-name"
    )
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    if not checkpoint_dir.is_dir():
        parser.error(f"checkpoint directory does not exist: {checkpoint_dir}")

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(args.repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(checkpoint_dir),
        commit_message=f"Upload checkpoint from {checkpoint_dir.name}",
    )
    print(f"Uploaded {checkpoint_dir} to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
