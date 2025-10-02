"""
Format the collected data for GPT-OSS training
"""
from datasets import Dataset
from unsloth.chat_templates import standardize_sharegpt
from config import MAX_SEQ_LENGTH

def prepare_dataset(raw_data, tokenizer):
    formatted_data = []
    for item in raw_data:
        formatted_data.append({
            "messages": [
                {"role": "user", "content": item["instruction"]},
                {"role": "assistant", "content": item["response"]}
            ]
        })
    dataset = Dataset.from_list(formatted_data)
    dataset = standardize_sharegpt(dataset)
    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = []
        for convo in convos:
            text = tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}
    dataset = dataset.map(formatting_prompts_func, batched=True)
    print(f"✅ Dataset ready with {len(dataset)} examples")
    print("\nExample formatted text (first 500 chars):")
    print(dataset[0]['text'][:500])
    if "<|channel|>" not in dataset[0]['text']:
        print("\n⚠️ Warning: Missing channel in format. Adding explicit channel...")
        def fix_formatting(examples):
            fixed_texts = []
            for text in examples["text"]:
                text = text.replace("<|start|>assistant<|message|>", "<|start|>assistant<|channel|>final<|message|>")
                fixed_texts.append(text)
            return {"text": fixed_texts}
        dataset = dataset.map(fix_formatting, batched=True)
        print("✅ Fixed formatting with channel")
        print("\nFixed example (first 500 chars):")
        print(dataset[0]['text'][:500])
    return dataset
