#!/usr/bin/env python3
"""
GPT-OSS Fine-tuning Pipeline
Complete pipeline for fine-tuning GPT-OSS with Unsloth and Bright Data
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def main():
    parser = argparse.ArgumentParser(description='Fine-tune GPT-OSS with Unsloth')
    parser.add_argument('--skip-data-collection', action='store_true', 
                       help='Skip data collection (use existing data)')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training (test existing model)')
    parser.add_argument('--max-steps', type=int, default=60,
                       help='Maximum training steps')
    args = parser.parse_args()

    print("="*60)
    print("GPT-OSS FINE-TUNING PIPELINE")
    print("="*60)
    
    # Step 1: Setup
    print("\n[Step 1/9] Setting up environment...")
    from src.setup import check_environment
    check_environment()
    
    # Step 2: Load Model
    print("\n[Step 2/9] Loading GPT-OSS model...")
    from src.model_loader import load_model
    model, tokenizer = load_model()
    
    # Step 3: Apply LoRA
    print("\n[Step 3/9] Applying LoRA adapters...")
    from src.lora_config import apply_lora
    model = apply_lora(model)
    
    # Step 4: Collect Data
    if not args.skip_data_collection:
        print("\n[Step 4/9] Collecting training data...")
        from src.data_collector import collect_training_data
        training_data = collect_training_data()
    else:
        print("\n[Step 4/9] Loading existing training data...")
        from src.utils import load_saved_data
        training_data = load_saved_data()
    
    # Step 5: Format Data
    print("\n[Step 5/9] Formatting dataset...")
    from src.data_formatter import prepare_dataset
    dataset = prepare_dataset(training_data, tokenizer)
    
    # Step 6: Setup Trainer
    print("\n[Step 6/9] Setting up trainer...")
    from src.trainer_setup import setup_trainer
    trainer = setup_trainer(model, tokenizer, dataset, args.max_steps)
    
    # Step 7: Train
    if not args.skip_training:
        print("\n[Step 7/9] Training model...")
        from src.training import train_model
        trainer_stats = train_model(trainer)
    else:
        print("\n[Step 7/9] Skipping training...")
    
    # Step 8: Test Model
    print("\n[Step 8/9] Testing model...")
    from src.model_tester import test_model_comprehensive
    test_model_comprehensive(model, tokenizer)
    
    # Step 9: Save Model
    print("\n[Step 9/9] Saving model...")
    from src.model_saver import save_model
    save_model(model, tokenizer)
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
