"""
Save your fine-tuned model locally or push to Hugging Face Hub
"""
def save_model(model, tokenizer):
    print("💾 Saving model...")
    model.save_pretrained("gpt-oss-python-expert-lora")
    tokenizer.save_pretrained("gpt-oss-python-expert-lora")
    print("✅ LoRA adapters saved to 'gpt-oss-python-expert-lora'")
    
    # Optional: Push to Hugging Face Hub
    model.push_to_hub("your-username/gpt-oss-python-expert-lora", token="your-hf-token")
    model.push_to_hub_merged("your-username/gpt-oss-python-expert", tokenizer, save_method="mxfp4", token="your-hf-token")
    print("✅ Model pushed to Hugging Face Hub!")
