import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset
import os
from dotenv import load_dotenv
from huggingface_hub import login
import pandas as pd
from tqdm import tqdm
import re

def load_trained_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        use_safetensors=True,
        use_cache=False
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
    )
    print(f'Loaded model and tokenizer from {model_path}')
    return model, tokenizer

def extract_tags(text, tag_name):
    """Extract content from XML-like tags"""
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[0].strip() if matches else ""

def generate_response(model, tokenizer, prompt, do_sample=True):
    # Create complete prompt template
    formatted_input = f'''<objective>
    Generate a JSON file describing the sketching and extrusion steps needed to construct a 3D CAD model based on the description provided. The output should include a reasoning section within the <think> tag and the corresponding JSON in the <json> tag. Do not provide any additional text outside of the tags.
    </objective>

    <instruction>
    You will be given a natural language description of a CAD design task enclosed within <description> </description>. Your task is to:
    1. Analyze the description and extract the relevant geometric and extrusion information.
    2. In the <think> tag, explain how you derived each field and value in the JSON from the description. This includes the geometric properties (e.g., coordinates, shapes) and extrusion operations. The reasoning should clarify how the geometry is mapped to the JSON structure and the chosen extrusion operation.
    3. Based on the reasoning in the <think> tag, generate the corresponding JSON structure for the CAD model in the <json> tag.

    The extrusion <operation> must be one of the following:
    1. <NewBodyFeatureOperation>: Creates a new solid body.
    2. <JoinFeatureOperation>: Fuses the shape with an existing body.
    3. <CutFeatureOperation>: Subtracts the shape from an existing body.
    4. <IntersectFeatureOperation>: Keeps only the overlapping volume between the new shape and existing body.
    </instruction>

    <description>
    {prompt}
    </description>'''
    
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant that generates CAD model descriptions in JSON format."},
        {"role": "user", "content": formatted_input}
    ]
    
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([formatted_prompt], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=10000,
        do_sample=do_sample,
        temperature=0.7 if do_sample else None,
        top_p=0.9 if do_sample else None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def process_dataset(model, tokenizer, dataset, split_name, do_sample=True):
    print(f"Generating predictions for {split_name}...")
    predictions = []
    reasonings = []
    
    # Take only first 1000 samples
    # dataset = dataset.select(range(min(500, len(dataset))))
    for item in tqdm(dataset, desc=f"Processing {split_name}"):
        response = generate_response(model, tokenizer, item["input"], do_sample=do_sample)
        
        # Extract reasoning from <think> tag
        reasoning = extract_tags(response, "think")
        reasonings.append(reasoning)
        
        # Extract JSON content from <json> tag (without the tags)
        json_content = extract_tags(response, "json")
        predictions.append(json_content)

    processed_dataset = Dataset.from_dict({
        "uid": dataset["uid"],
        "input": dataset["input"],
        "reasoning": reasonings,
        "predicted_output": predictions,
        "ground_truth_output": dataset["output"]
    })

    print(f"Pushing {split_name} dataset to hub...")
    processed_dataset.push_to_hub("wanhin/test_reasoning_1_2e", split=split_name)
    print(f"{split_name} dataset pushed successfully!")

def main():
    # Load environment variables and login to HF
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        raise ValueError("Please set HF_TOKEN environment variable")
    login(token=hf_token)

    # Load the trained model
    model_path = "wanhin/cad_reasoning_1_2e"
    model, tokenizer = load_trained_model(model_path)

    # Load English and Vietnamese test datasets separately
    test_en = load_dataset("TruongSinhAI/DEEPCAD-Text2Json-EnVi", split="test_en")
    test_vi = load_dataset("TruongSinhAI/DEEPCAD-Text2Json-EnVi", split="test_vi")

    # Process English dataset without sampling
    # process_dataset(model, tokenizer, test_en, "no_sampling_en", do_sample=False)

    # Process English dataset with sampling
    # process_dataset(model, tokenizer, test_en, "sampling_en", do_sample=True)

    # # Process Vietnamese dataset without sampling
    process_dataset(model, tokenizer, test_vi, "no_sampling_vi", do_sample=False)

    # # Process Vietnamese dataset with sampling
    # process_dataset(model, tokenizer, test_vi, "sampling_vi", do_sample=True)

if __name__ == "__main__":
    main()
