from datasets import load_dataset, interleave_datasets, concatenate_datasets
from huggingface_hub import login
from dotenv import load_dotenv
import os
import torch

load_dotenv()
hf_token = os.getenv('HF_TOKEN')
if not hf_token:
    raise ValueError("Please set HF_TOKEN environment variable")
login(token=hf_token)

# Load new datasets
train_en = load_dataset("TruongSinhAI/DEEPCAD-Text2Json-EnVi", split="val_en")
train_vi = load_dataset("TruongSinhAI/DEEPCAD-Text2Json-EnVi", split="val_vi")

# Interleave new datasets
new_dataset = interleave_datasets([train_en, train_vi])

# Load old dataset
old_dataset = load_dataset("wanhin/DEEPCAD-completion-sft", split="train")

messages_prompt ='''<objective>
Generate a JSON file describing the sketching and extrusion steps needed to construct a 3D CAD model. Generate only the JSON file, no other text.
</objective>

<instruction>
You will be given a natural language description of a CAD design task. Your goal is to convert it into a structured JSON representation, which includes sketch geometry and extrusion operations.

The JSON must follow the structure defined in the <template> section, and the extrusion <operation> must be one of the following:

1. <NewBodyFeatureOperation>: Creates a new solid body.
2. <JoinFeatureOperation>: Fuses the shape with an existing body.
3. <CutFeatureOperation>: Subtracts the shape from an existing body.
4. <IntersectFeatureOperation>: Keeps only the overlapping volume between the new shape and existing body.

Ensure all coordinates, geometry, and extrusion depths are extracted accurately from the input.
</instruction>'''

def process_example(example):
    # Convert input list to string if it's a list
    input_text = example["input"]
    if isinstance(input_text, list):
        input_text = "\n".join(input_text)
    
    return {
        "prompt": messages_prompt + "\n\n<description>\n" + input_text + "\n</description>",
        "completion": example["output"]
    }

# Process new dataset with GPU acceleration
device = "cuda" if torch.cuda.is_available() else "cpu"
processed_new_dataset = new_dataset.map(
    process_example, 
    batched=False, 
    num_proc=30,
    remove_columns=new_dataset.column_names,
    load_from_cache_file=False
)

# Concatenate old and new datasets
combined_dataset = concatenate_datasets([old_dataset, processed_new_dataset])

print(f"Old dataset size: {len(old_dataset)}")
print(f"New dataset size: {len(processed_new_dataset)}")
print(f"Combined dataset size: {len(combined_dataset)}")

# Push to hub
combined_dataset.push_to_hub("wanhin/DEEPCAD-completion-sft")


