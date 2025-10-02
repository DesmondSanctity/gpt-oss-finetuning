"""
Model testing with different reasoning effort levels
"""
from transformers import TextStreamer
import torch

def test_single_question(model, tokenizer, prompt, reasoning_effort="medium", max_length=256):
    messages = [
        {"role": "system", "content": "You are a Python expert assistant."},
        {"role": "user", "content": prompt}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        reasoning_effort=reasoning_effort,
    ).to("cuda")
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        streamer=streamer,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def test_reasoning_levels(model, tokenizer):
    complex_question = "Write a Python function that finds all prime numbers up to n using the Sieve of Eratosthenes"
    print("\n" + "="*60)
    print("TESTING REASONING EFFORT LEVELS")
    print("="*60)
    for effort in ["low", "medium", "high"]:
        print(f"\n[Reasoning: {effort.upper()}]")
        print("-"*40)
        test_single_question(model, tokenizer, complex_question, reasoning_effort=effort, max_length=300)
        print()

def test_model_comprehensive(model, tokenizer):
    test_questions = [
        "What is a Python generator?",
        "How do I read a CSV file in Python?",
        "Explain async/await in Python"
    ]
    print("\nTesting on Python questions:")
    print("="*60)
    for i, question in enumerate(test_questions, 1):
        print(f"\nQ{i}: {question}")
        print("-"*40)
        test_single_question(model, tokenizer, question)
    test_reasoning_levels(model, tokenizer)
