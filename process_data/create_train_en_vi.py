import os
from dotenv import load_dotenv
from datasets import load_dataset, Dataset
from huggingface_hub import login
from transformers import AutoTokenizer

# Load HF token và login
load_dotenv()
hf_token = os.getenv('HF_TOKEN')
if not hf_token:
    raise ValueError("Please set HF_TOKEN environment variable in .env file")
login(token=hf_token)
print("Successfully logged in to Hugging Face")

# Load dataset
raw = load_dataset('wanhin/reason_s1_full', split="en_vi")
raw_test = load_dataset('wanhin/reason_s1_full', split="test")

# Instruction template
instruction = '''<objective>
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
</instruction>'''

# Process train dataset
print("Processing train dataset...")
processed_train_data = []
for item in raw:
    # Tạo prompt mới
    prompt = instruction + "\n\n" + item['description']
    
    # Gộp reasoning + completion thành completion mới
    completion = item['reasoning'] + "\n\n" + item['completion']
    
    processed_train_data.append({
        'prompt': prompt,
        'completion': completion
    })

# Process test dataset
print("Processing test dataset...")
processed_test_data = []
for item in raw_test:
    # Tạo prompt mới
    prompt = instruction + "\n\n" + item['description']
    
    # Gộp reasoning + completion thành completion mới
    completion = item['reasoning'] + "\n\n" + item['completion']
    
    processed_test_data.append({
        'prompt': prompt,
        'completion': completion
    })

# Tạo datasets mới
train_dataset = Dataset.from_list(processed_train_data)
test_dataset = Dataset.from_list(processed_test_data)

# In thông tin datasets
print(f"Train dataset features: {train_dataset.features}")
print(f"Train dataset columns: {train_dataset.column_names}")
print(f"Train sample data:")
print(train_dataset[0])
print(f"Train dataset size: {len(train_dataset)}")

print(f"\nTest dataset features: {test_dataset.features}")
print(f"Test dataset columns: {test_dataset.column_names}")
print(f"Test sample data:")
print(test_dataset[0])
print(f"Test dataset size: {len(test_dataset)}")

# Khảo sát max token length
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", use_fast=True)

def count_tokens(text):
    return len(tokenizer.encode(text))

print("\nKhảo sát token length...")
max_length_train = 0
max_length_test = 0

for item in processed_train_data:
    full_text = item['prompt'] + item['completion']
    token_count = count_tokens(full_text)
    max_length_train = max(max_length_train, token_count)

for item in processed_test_data:
    full_text = item['prompt'] + item['completion']
    token_count = count_tokens(full_text)
    max_length_test = max(max_length_test, token_count)

print(f"Max token length train: {max_length_train}")
print(f"Max token length test: {max_length_test}")

# Push lên HF - tạo repo mới để tránh conflict
repo_id = 'wanhin/reason_s1_full_train'
train_dataset.push_to_hub(repo_id, split="train_en_vi")
test_dataset.push_to_hub(repo_id, split="val")

print(f"Đã push lên {repo_id} với splits: train_en_vi, val") 