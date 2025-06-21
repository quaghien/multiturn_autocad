from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
import numpy as np
import os
import json
import re
from collections import defaultdict
from dotenv import load_dotenv
from huggingface_hub import login

def count_tokens(texts, tokenizer):
    """Count tokens for multiple texts using simple loop"""
    lengths = []
    for text in texts:
        lengths.append(len(tokenizer.encode(text)))
    return lengths

def extract_description_content(prompt_text):
    """Extract content from <description> tags only"""
    # Find content between <description> and </description> tags
    pattern = r'<description>(.*?)</description>'
    match = re.search(pattern, prompt_text, re.DOTALL)
    
    if match:
        # Return the content with the tags (giữ nguyên cả tag)
        return f"<description>{match.group(1).strip()}</description>"
    else:
        # If no description tags found, return original text
        return prompt_text

def clean_completion_json(completion_text):
    """Remove 'final_shape' and 'description' fields from JSON completion and wrap with <json> tags"""
    try:
        # Parse JSON
        data = json.loads(completion_text)
        
        # Remove 'final_shape' field if it exists
        if 'final_shape' in data:
            del data['final_shape']
        
        # Remove 'description' field from each part if it exists
        if 'parts' in data:
            for part_name, part_data in data['parts'].items():
                if 'description' in part_data:
                    del part_data['description']
        
        # Convert back to JSON string and wrap with <json> tags
        json_content = json.dumps(data, separators=(',', ':'))
        return f"<json>\n{json_content}\n</json>"
    except json.JSONDecodeError:
        # If JSON parsing fails, return None to indicate invalid data
        return None

def classify_samples_by_token_count(token_lengths, token_range=500):
    """
    Classify samples into ranges based on token count
    token_range: size of each range (e.g., 500 tokens)
    """
    classified_samples = defaultdict(list)
    
    for i, token_count in enumerate(token_lengths):
        # Calculate which range this sample belongs to
        range_start = (token_count // token_range) * token_range
        range_end = range_start + token_range
        
        # Create range key
        range_key = f"range_{range_start}_{range_end}"
        classified_samples[range_key].append(i)
    
    return classified_samples

def create_dataset_splits(raw_dataset, classified_samples, language_suffix):
    """Create dataset splits based on classified samples"""
    splits = {}
    
    for range_name, sample_indices in classified_samples.items():
        if sample_indices:  # Only create split if there are samples
            # Add language suffix to split name
            split_name = f"{range_name}_{language_suffix}"
            
            # Extract samples for this split
            split_data = {
                "prompt": [raw_dataset[i]["prompt"] for i in sample_indices],
                "completion": [raw_dataset[i]["completion"] for i in sample_indices]
            }
            
            splits[split_name] = Dataset.from_dict(split_data)
            print(f"Created split '{split_name}' with {len(sample_indices)} samples")
    
    return splits

def main():
    # Login to Hugging Face using .env file
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        raise ValueError("Please set HF_TOKEN environment variable in .env file")
    login(token=hf_token)
    print("Successfully logged in to Hugging Face")
    
    # Load dataset
    print("Loading dataset...")
    raw_dataset = load_dataset("wanhin/DEEPCAD-stage1", split="train_en_vi")
    
    # Split dataset into English and Vietnamese
    print("Splitting dataset into English and Vietnamese...")
    en_indices = list(range(0, len(raw_dataset), 2))  # Even indices (0, 2, 4, ...)
    vi_indices = list(range(1, len(raw_dataset), 2))  # Odd indices (1, 3, 5, ...)
    
    en_dataset = raw_dataset.select(en_indices)
    vi_dataset = raw_dataset.select(vi_indices)
    
    print(f"English dataset: {len(en_dataset)} samples")
    print(f"Vietnamese dataset: {len(vi_dataset)} samples")
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        use_fast=True
    )
    
    # Process English dataset
    print("\nProcessing English dataset...")
    en_processed_data = []
    for item in en_dataset:
        # Process prompt: extract only description content
        processed_prompt = extract_description_content(item["prompt"])
        
        # Process completion: remove final_shape and description fields
        processed_completion = clean_completion_json(item["completion"])
        
        # Only add data if completion is valid JSON
        if processed_completion is not None:
            en_processed_data.append({
                "prompt": processed_prompt,
                "completion": processed_completion
            })
    
    # Process Vietnamese dataset
    print("Processing Vietnamese dataset...")
    vi_processed_data = []
    for item in vi_dataset:
        # Process prompt: extract only description content
        processed_prompt = extract_description_content(item["prompt"])
        
        # Process completion: remove final_shape and description fields
        processed_completion = clean_completion_json(item["completion"])
        
        # Only add data if completion is valid JSON
        if processed_completion is not None:
            vi_processed_data.append({
                "prompt": processed_prompt,
                "completion": processed_completion
            })
    
    print(f"English dataset after filtering: {len(en_processed_data)} samples (removed {len(en_dataset) - len(en_processed_data)} invalid samples)")
    print(f"Vietnamese dataset after filtering: {len(vi_processed_data)} samples (removed {len(vi_dataset) - len(vi_processed_data)} invalid samples)")
    
    # Create processed datasets
    en_processed_dataset = Dataset.from_list(en_processed_data)
    vi_processed_dataset = Dataset.from_list(vi_processed_data)
    
    # Count tokens for processed prompts
    print("\nCounting tokens for processed prompts...")
    en_prompt_texts = [item["prompt"] for item in en_processed_data]
    en_prompt_lengths = count_tokens(en_prompt_texts, tokenizer)
    
    vi_prompt_texts = [item["prompt"] for item in vi_processed_data]
    vi_prompt_lengths = count_tokens(vi_prompt_texts, tokenizer)
    
    # Print statistics
    print(f"\nEnglish dataset - Prompt lengths - Min: {min(en_prompt_lengths)}, Max: {max(en_prompt_lengths)}")
    print(f"Vietnamese dataset - Prompt lengths - Min: {min(vi_prompt_lengths)}, Max: {max(vi_prompt_lengths)}")
    
    # Classify samples by token count (500 token ranges)
    print("\nClassifying samples by token count...")
    en_classified = classify_samples_by_token_count(en_prompt_lengths, token_range=500)
    vi_classified = classify_samples_by_token_count(vi_prompt_lengths, token_range=500)
    
    # Create dataset splits
    print("\nCreating dataset splits...")
    en_splits = create_dataset_splits(en_processed_dataset, en_classified, "en")
    vi_splits = create_dataset_splits(vi_processed_dataset, vi_classified, "vi")
    
    # Combine all splits
    all_splits = {**en_splits, **vi_splits}
    
    # Create DatasetDict
    dataset_dict = DatasetDict(all_splits)
    
    # Print final statistics
    print("\nFinal dataset statistics:")
    for split_name, dataset in dataset_dict.items():
        print(f"  {split_name}: {len(dataset)} samples")
    
    # Upload to Hugging Face
    print("\nUploading to Hugging Face...")
    try:
        dataset_dict.push_to_hub("wanhin/cad_hqh1")
        print("Successfully uploaded to wanhin/cad_hqh1")
    except Exception as e:
        print(f"Error uploading to Hugging Face: {e}")

if __name__ == "__main__":
    main() 