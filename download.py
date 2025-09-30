from huggingface_hub import snapshot_download

target_dir = r"D:\AIModels"
snapshot_download(repo_id="Qwen/Qwen-Image-Edit-2509", local_dir=target_dir)
print(f"Downloaded to {target_dir}")