"""
Start the actual fine-tuning process
"""
import torch

def train_model(trainer):
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    print(f"GPU memory reserved before training: {start_gpu_memory} GB")
    print("\n🚀 Starting training...")
    print("This will take about 5-10 minutes for 30 steps.")
    print("For full training, set max_steps=None and num_train_epochs=1\n")
    trainer_stats = trainer.train()
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    print("\n✅ Training completed!")
    print(f"Time: {trainer_stats.metrics['train_runtime']/60:.1f} minutes")
    print(f"Final loss: {trainer_stats.metrics['train_loss']:.4f}")
    print(f"Peak memory for training: {used_memory_for_lora} GB")
    print(f"Total peak memory: {used_memory} GB ({used_memory_for_lora} GB for LoRA)")
    return trainer_stats
