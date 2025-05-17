from datasets import load_dataset
from transformers import AutoTokenizer

# Load dataset
dataset = load_dataset("wanhin/DEEPCAD-completion-sft", split="train")

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    use_fast=True,
    padding_side="left",
    model_max_length=8192
)

# Combine prompt and completion
combined_texts = [f"{item['prompt']}\n{item['completion']}" for item in dataset]

# Tokenize all texts
tokenized_lengths = [len(tokenizer.encode(text)) for text in combined_texts]

# Find min and max lengths with their indices
min_length = min(tokenized_lengths)
max_length = max(tokenized_lengths)
min_index = tokenized_lengths.index(min_length)
max_index = tokenized_lengths.index(max_length)

print("\nSample with minimum length:")
print(f"Text: {combined_texts[min_index]}")
print(f"Tokenized length: {min_length}")

print("\nSample with maximum length:")
print(f"Text: {combined_texts[max_index]}")
print(f"Tokenized length: {max_length}")

print(f"Total number of samples: {len(dataset)}")
print(f"Minimum length: {min_length} (index: {min_index})")
print(f"Maximum length: {max_length} (index: {max_index})")