# GPT-OSS Fine-tuning with Unsloth

Fine-tune GPT-OSS-20B 2x faster using Unsloth and Bright Data. Runs on free Google Colab T4 GPU.

## Quick Start
```bash
# Clone repo
git clone https://github.com/yourusername/gpt-oss-finetuning
cd gpt-oss-finetuning

# Install dependencies
pip install -r requirements.txt

# Install Unsloth (special installation)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Setup environment
cp .env.example .env
# Edit .env with your API keys

# Run complete pipeline
python main.py

# Or run with options
python main.py --skip-data-collection  # Use cached data
python main.py --max-steps 100        # Train longer
```

Requirements

GPU: 16GB+ VRAM (T4, V100, A100)
Python: 3.8+
CUDA: 11.8+

Features

✅ 2x faster training with Unsloth
✅ Runs on free Colab T4 GPU
✅ LoRA fine-tuning (only 1% parameters)
✅ Reasoning effort control (low/medium/high)
✅ Automatic data collection with Bright Data

Project Structure
src/
├── 01_setup.py         # Environment check
├── 02_model_loader.py  # Load GPT-OSS
├── 03_lora_config.py   # LoRA setup
├── 04_data_collector.py # Bright Data scraping
├── 05_data_formatter.py # Format for training
├── 06_trainer_setup.py  # Configure trainer
├── 07_training.py      # Train model
├── 08_model_tester.py  # Test with reasoning levels
└── 09_model_saver.py   # Save to disk/HF Hub

API Keys Required

Bright Data: Get API token
Hugging Face: Create token

Run Individual Steps
```python
# Test model only
python src/08_model_tester.py

# Save model only
python src/09_model_saver.py
```

Notebook
Full Colab notebook available: notebooks/GPT_OSS_Fine_tuning_Complete.ipynb

Performance
| Metric         | Value                |
|---------------|----------------------|
| Training Speed| ~30 tokens/sec       |
| Memory Usage  | 12-14GB              |
| Training Time | 10-15 min (1000 ex.) |
| Parameters    | <1% (LoRA)           |

Links
- Unsloth Documentation
- Bright Data
- Original Article

License
MIT

This structure provides:

1. **Modular design**: Each step is a separate file that can run independently
2. **main.py**: Orchestrates the complete pipeline with command-line options
3. **config.py**: Centralized configuration
4. **Environment variables**: Secure API key management
5. **Reasoning level testing**: Included in the model tester
6. **Clean README**: Direct and actionable

You can run the complete pipeline with `python main.py` or individual steps by importing them. The notebook is preserved for Colab users who prefer that interface.
