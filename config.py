import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
BRIGHTDATA_API_TOKEN = os.getenv('BRIGHTDATA_API_TOKEN')
HF_TOKEN = os.getenv('HF_TOKEN')

# Model Configuration
MODEL_NAME = os.getenv('MODEL_NAME', 'unsloth/gpt-oss-20b')
MAX_SEQ_LENGTH = 1024
LOAD_IN_4BIT = True

# LoRA Configuration
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0

# Training Configuration
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
MAX_STEPS = 60
LEARNING_RATE = 2e-4
OUTPUT_DIR = os.getenv('OUTPUT_DIR', './outputs')

# Data Collection
SCRAPING_URLS = [
    "https://docs.python.org/3/tutorial/introduction.html",
    "https://docs.python.org/3/tutorial/controlflow.html",
    "https://docs.python.org/3/tutorial/datastructures.html",
]
