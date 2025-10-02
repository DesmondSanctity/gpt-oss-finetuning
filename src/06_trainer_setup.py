"""
Configure the training parameters and setup trainer
"""
from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import train_on_responses_only
from config import BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS, MAX_STEPS, LEARNING_RATE, OUTPUT_DIR

def setup_trainer(model, tokenizer, dataset, max_steps):
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=5,
            max_steps=max_steps,
            learning_rate=LEARNING_RATE,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=OUTPUT_DIR,
            report_to="none",
        ),
    )
    gpt_oss_kwargs = dict(
        instruction_part = "<|start|>user<|message|>",
        response_part = "<|start|>assistant<|channel|>final<|message|>"
    )
    trainer = train_on_responses_only(trainer, **gpt_oss_kwargs)
    print("✅ Trainer configured!")
    sample = trainer.train_dataset[0]
    decoded_labels = tokenizer.decode([
        tokenizer.pad_token_id if x == -100 else x for x in sample["labels"]
    ]).replace(tokenizer.pad_token, " ")
    print("\nVerifying we only train on assistant responses:")
    print(f"Training on: {decoded_labels[:200]}...")
    return trainer
