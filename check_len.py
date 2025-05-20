from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from trl import apply_chat_template

# Load dataset
raw_dataset = load_dataset("wanhin/DEEPCAD-completion-sft", split="train")

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    use_fast=True,
    padding_side="left",
    model_max_length=6000
)

# Convert to chat format
dataset_dict = {
    "prompt": [[{"role": "user", "content": item["prompt"]}] for item in raw_dataset],
    "completion": [[{"role": "assistant", "content": item["completion"]}] for item in raw_dataset]
}

dataset = Dataset.from_dict(dataset_dict)
dataset = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})

# Get tokenized lengths
tokenized_lengths = [len(tokenizer.encode(text)) for text in dataset["prompt"]]

# Find min and max lengths with their indices
min_length = min(tokenized_lengths)
max_length = max(tokenized_lengths)
min_index = tokenized_lengths.index(min_length)
max_index = tokenized_lengths.index(max_length)

print(f"Total number of samples: {len(dataset)}")
print(f"Minimum length: {min_length} (index: {min_index})")
print(f"Maximum length: {max_length} (index: {max_index})")