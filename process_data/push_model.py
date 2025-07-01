import os
from dotenv import load_dotenv
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load HF token và login
load_dotenv()
hf_token = os.getenv('HF_TOKEN')
if not hf_token:
    raise ValueError("Please set HF_TOKEN environment variable in .env file")
login(token=hf_token)
print("Successfully logged in to Hugging Face")

# Path to local checkpoint
checkpoint_path = "train_results/Qwen2.5-7B-Instruct_1e_fullfinetune_2epoch_stage2_24-06/checkpoint-2678"

# Load tokenizer from local checkpoint
print(f"Loading tokenizer from {checkpoint_path}...")
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=True)
print("Tokenizer loaded successfully!")

# Load model from local checkpoint
print(f"Loading model from {checkpoint_path}...")
model = AutoModelForCausalLM.from_pretrained(
    checkpoint_path,
    torch_dtype="auto",
    device_map="auto"
)
print("Model loaded successfully!")

# Push model and tokenizer to hub
hub_model_name = "wanhin/cad_reasoning_1_2e"
print(f"Pushing model and tokenizer to {hub_model_name}...")

# Push tokenizer
print("Pushing tokenizer...")
tokenizer.push_to_hub(hub_model_name)

# Push model
print("Pushing model...")
model.push_to_hub(hub_model_name)

print(f"Successfully pushed model and tokenizer to {hub_model_name}!") 