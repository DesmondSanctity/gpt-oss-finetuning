"""
Load GPT-OSS-20B model with Unsloth
"""
from unsloth import FastLanguageModel
from config import MODEL_NAME, MAX_SEQ_LENGTH, LOAD_IN_4BIT

def load_model():
    print("Loading GPT-OSS-20B with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        dtype=None,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        full_finetuning=False,
    )
    print("✅ Model loaded successfully!")
    return model, tokenizer
