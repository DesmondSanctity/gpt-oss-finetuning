"""
Environment check and dependency installation
"""
import torch

def check_environment():
    # Check GPU
    gpu_stats = torch.cuda.get_device_properties(0)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    if max_memory < 15:
        print("⚠️ Warning: You need at least 16GB GPU memory. Switch to T4 or better.")
    else:
        print("✅ GPU memory sufficient for GPT-OSS-20B fine-tuning!")
