"""Upload the trained model folder to the Hugging Face Hub.
Usage: python src/upload_to_hub.py --folder models/afriberta_base --repo YOUR_USERNAME/kinyarwanda-news-classifier
(Log in first: run `huggingface-cli login` or set HF_TOKEN.)
"""
import argparse
from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("--folder", required=True)
ap.add_argument("--repo", required=True)
args = ap.parse_args()

api = HfApi()
api.create_repo(args.repo, exist_ok=True)
api.upload_folder(folder_path=args.folder, repo_id=args.repo)
print(f"Uploaded. Model page: https://huggingface.co/{args.repo}")
