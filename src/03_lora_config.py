"""
Add LoRA adapters for parameter-efficient fine-tuning
"""
from unsloth import FastLanguageModel
from config import LORA_R, LORA_ALPHA, LORA_DROPOUT

def apply_lora(model):
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj",],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"✅ LoRA applied!")
    print(f"Training {trainable_params:,} / {all_params:,} params")
    print(f"That's {100 * trainable_params / all_params:.2f}% of all parameters")
    return model
